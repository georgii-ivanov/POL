#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
import hashlib
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import hivemind
from hivemind import DHT, get_logger
import numpy as np

logger = get_logger(__name__)

class LoRATrainer:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1", peer_id: Optional[str] = None):
        self.model_name = model_name
        self.peer_id = peer_id
        self.model_cache_dir = "/model_cache"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing LoRATrainer with model: {model_name}")
        self._load_model()
        self._setup_lora()
        self._setup_dht()
        
    def _load_model(self):
        try:
            if os.path.exists(f"{self.model_cache_dir}/tokenizer"):
                logger.info("Loading cached tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_cache_dir}/tokenizer")
            else:
                logger.error("Tokenizer cache not found and HuggingFace unreachable")
                sys.exit(1)
                
            if os.path.exists(f"{self.model_cache_dir}/model"):
                logger.info("Loading cached model...")
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    f"{self.model_cache_dir}/model",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                logger.error("Model cache not found and HuggingFace unreachable")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Model not found: {e}")
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
        logger.info(f"DHT initialized with peers: {initial_peers}")
        
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

def main():
    parser = argparse.ArgumentParser(description="POL-AI LoRA Trainer")
    parser.add_argument("--peer-id", type=str, help="Peer ID for DHT")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1", help="Model name")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--training-data", type=str, help="Path to training data file")
    
    args = parser.parse_args()
    
    training_texts = []
    if args.training_data and os.path.exists(args.training_data):
        with open(args.training_data, 'r') as f:
            training_texts = [line.strip() for line in f if line.strip()]
    
    if not training_texts:
        logger.error("No training data provided. Use --training-data argument.")
        sys.exit(1)
    
    trainer = LoRATrainer(model_name=args.model, peer_id=args.peer_id)
    asyncio.run(trainer.training_loop(training_texts, steps=args.steps))

if __name__ == "__main__":
    main()

 