#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
import hashlib
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import hivemind
from hivemind import DHT, get_logger
import numpy as np
from data_acquisition import DataAcquisition

logger = get_logger(__name__)

class LoRATrainer:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1", 
                 data_cache_dir: str = "./data_cache", model_cache_dir: str = "./model_cache"):
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.data_acquisition = DataAcquisition(cache_dir=data_cache_dir)
        
        logger.info(f"Initializing LoRATrainer with model: {model_name}")
        self._load_model()
        self._setup_lora()
        self._setup_dht()
        
    def _detect_hardware_capabilities(self):
        """Detect available hardware and determine optimal model loading configuration"""
        self.use_gpu = False
        self.gpu_info = None
        
        if torch.cuda.is_available():
            try:
                self.gpu_info = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU detected: {self.gpu_info} with {gpu_memory:.1f}GB memory")
                
                # Check if GPU supports the required features
                if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported'):
                    self.use_gpu = True
                    logger.info("GPU supports required features, will use GPU acceleration")
                else:
                    self.use_gpu = True
                    logger.info("GPU detected, will use GPU acceleration with float16")
                    
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
                self.use_gpu = False
        else:
            logger.info("CUDA not available, will use CPU")
            
        if not self.use_gpu:
            logger.info("Using CPU-only mode with optimized settings")
            
    def _get_model_kwargs(self, is_cached=False):
        """Get model loading parameters based on hardware capabilities"""
        base_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        if not is_cached:
            base_kwargs["cache_dir"] = self.model_cache_dir
            
        if self.use_gpu:
            base_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": "auto",
            })
        else:
            base_kwargs.update({
                "torch_dtype": torch.float32,
                "device_map": "cpu",
                "use_cache": False,
                "quantization_config": None,
                "attn_implementation": "eager",
                "load_in_8bit": False,
                "load_in_4bit": False,
            })
            
        return base_kwargs
    
    def _load_model(self):
        self._detect_hardware_capabilities()
        
        tokenizer_cache_path = f"{self.model_cache_dir}/tokenizer"
        model_cache_path = f"{self.model_cache_dir}/model"
        
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        try:
            if os.path.exists(tokenizer_cache_path):
                logger.info("Loading cached tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache_path)
            else:
                logger.info(f"Tokenizer cache not found, downloading from HuggingFace: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.model_cache_dir,
                    trust_remote_code=True
                )
                logger.info("Saving tokenizer to cache...")
                self.tokenizer.save_pretrained(tokenizer_cache_path)
                
            if os.path.exists(model_cache_path):
                logger.info("Loading cached model...")
                model_kwargs = self._get_model_kwargs(is_cached=True)
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_cache_path,
                    **model_kwargs
                )
            else:
                logger.info(f"Model cache not found, downloading from HuggingFace: {self.model_name}")
                logger.warning("This may take several minutes for large models like DeepSeek-R1...")
                
                model_kwargs = self._get_model_kwargs(is_cached=False)
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                logger.info("Saving model to cache...")
                self.base_model.save_pretrained(model_cache_path)
                
            logger.info(f"Model loaded successfully in {'GPU' if self.use_gpu else 'CPU'} mode")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Please check your internet connection and HuggingFace access")
            sys.exit(1)
            
    def _setup_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.train()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.0
        )
        
        logger.info("LoRA configuration applied")
        
    def _setup_dht(self):
        initial_peers = []
        genesis_host = os.getenv("GENESIS_HOST")
        if genesis_host and genesis_host != "localhost":
            initial_peers = [f"/ip4/{genesis_host}/tcp/13337"]
            
        self.dht = DHT(
            start=True,
            initial_peers=initial_peers,
            client_mode=False,
            host_maddrs=["/ip4/0.0.0.0/tcp/0"],
            record_validators={}
        )
        
        self.peer_id = str(self.dht.peer_id)
        logger.info(f"DHT initialized with peers: {initial_peers}")
        logger.info(f"Local peer ID: {self.peer_id}")
        
    def extract_lora_delta(self) -> bytes:
        delta_weights = {}
        
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                delta_weights[name] = param.data.cpu().half()
                
        delta_bytes = torch.save(delta_weights, f=None)
        
        if len(delta_bytes) > 150 * 1024:
            logger.warning(f"LoRA delta size {len(delta_bytes)} bytes exceeds 150KB limit")
            
        return delta_bytes
        
    async def publish_delta(self, delta_bytes: bytes) -> str:
        raw_id = hashlib.sha256(delta_bytes).hexdigest()
        dht_key = f"raw:{raw_id}"
        
        success = await self.dht.store(
            key=dht_key,
            value=delta_bytes,
            expiration_time=hivemind.get_dht_time() + 3600
        )
        
        if success:
            logger.info(f"Published LoRA delta to DHT key: {dht_key}")
        else:
            logger.error(f"Failed to publish delta to DHT")
            
        return raw_id
        
    def train_step(self, text_batch: list) -> float:
        total_loss = 0.0
        batch_size = len(text_batch)
        
        for text in text_batch:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / batch_size
        logger.info(f"Training step completed, average loss: {avg_loss:.4f}")
        return avg_loss
        
    async def training_loop(self, training_texts: list, steps: int = 100):
        logger.info(f"Starting training loop for {steps} steps")
        
        for step in range(steps):
            batch = np.random.choice(training_texts, size=min(4, len(training_texts)), replace=False)
            loss = self.train_step(batch.tolist())
            
            if step % 10 == 0:
                delta_bytes = self.extract_lora_delta()
                raw_id = await self.publish_delta(delta_bytes)
                logger.info(f"Step {step}: loss={loss:.4f}, published delta {raw_id[:16]}")
                
        logger.info("Training loop completed")
        
    def acquire_training_data(self, dataset_keys: List[str], 
                             samples_per_dataset: Optional[int] = None,
                             total_samples: int = 1000) -> List[str]:
        
        logger.info(f"Acquiring training data from datasets: {dataset_keys}")
        
        return self.data_acquisition.load_datasets(
            dataset_keys=dataset_keys,
            samples_per_dataset=samples_per_dataset,
            total_samples=total_samples
        )
            
    def list_available_datasets(self) -> None:
        self.data_acquisition.list_available_datasets()
        
    def save_training_data(self, samples: List[str], filename: str) -> None:
        self.data_acquisition.save_dataset_to_file(samples, filename)
        
    def load_training_data_from_file(self, filename: str) -> List[str]:
        return self.data_acquisition.load_dataset_from_file(filename)
        
    async def auto_training_pipeline(self, dataset_keys: List[str], 
                                     samples_per_dataset: Optional[int] = None,
                                     total_samples: int = 1000, 
                                     steps: int = 100) -> None:
        logger.info("Starting automatic training pipeline")
        
        training_data = self.acquire_training_data(
            dataset_keys=dataset_keys,
            samples_per_dataset=samples_per_dataset,
            total_samples=total_samples
        )
        
        if not training_data:
            logger.error("No training data acquired")
            return
            
        logger.info(f"Acquired {len(training_data)} training samples")
        await self.training_loop(training_data, steps)

def main():
    parser = argparse.ArgumentParser(description="POL-AI LoRA Trainer")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1", help="Model name")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--training-data", type=str, help="Path to training data file")
    parser.add_argument("--data-cache-dir", type=str, default="./data_cache", help="Data cache directory")
    parser.add_argument("--model-cache-dir", type=str, default="./model_cache", help="Model cache directory")
    
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--auto-data", action="store_true", help="Automatically acquire training data")
    parser.add_argument("--datasets", nargs="+", help="Dataset keys to use (uses all datasets if not specified)")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to acquire")
    parser.add_argument("--samples-per-dataset", type=int, help="Samples per dataset")
    parser.add_argument("--save-data", type=str, help="Save acquired data to file")
    
    args = parser.parse_args()
    
    trainer = LoRATrainer(
        model_name=args.model, 
        data_cache_dir=args.data_cache_dir,
        model_cache_dir=args.model_cache_dir
    )
    
    if args.list_datasets:
        trainer.list_available_datasets()
        return
    
    training_texts = []
    
    if args.auto_data:
        datasets_to_use = args.datasets
        if not datasets_to_use:
            datasets_to_use = list(trainer.data_acquisition.available_datasets.keys())
            logger.info(f"No datasets specified, using all available datasets: {len(datasets_to_use)} datasets")
            
        try:
            training_texts = trainer.acquire_training_data(
                dataset_keys=datasets_to_use,
                samples_per_dataset=args.samples_per_dataset,
                total_samples=args.num_samples
            )
            logger.info(f"Acquired {len(training_texts)} training samples")
            
            if args.save_data:
                trainer.save_training_data(training_texts, args.save_data)
                logger.info(f"Saved training data to {args.save_data}")
                
        except Exception as e:
            logger.error(f"Failed to acquire training data: {e}")
            sys.exit(1)
            
    elif args.training_data and os.path.exists(args.training_data):
        training_texts = trainer.load_training_data_from_file(args.training_data)
        
    else:
        logger.info("No training data specified. Using Wikipedia Simple dataset...")
        training_texts = trainer.acquire_training_data(
            dataset_keys=["wikipedia_simple"],
            total_samples=1000
        )
    
    if not training_texts:
        logger.error("No training data available")
        sys.exit(1)
    
    logger.info(f"Starting training with {len(training_texts)} samples")
    asyncio.run(trainer.training_loop(training_texts, steps=args.steps))

if __name__ == "__main__":
    main()

 