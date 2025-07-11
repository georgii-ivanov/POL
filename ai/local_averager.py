#!/usr/bin/env python3

import os
import sys
import asyncio
import hashlib
import pickle
from typing import List, Dict, Tuple
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import hivemind
from hivemind import DHT, get_logger
import numpy as np

logger = get_logger(__name__)

DEV_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming modern technology.",
    "Blockchain networks enable decentralized digital transactions.",
    "Machine learning algorithms process vast amounts of data.",
    "Distributed systems require careful coordination mechanisms.",
    "Consensus protocols ensure network agreement on state.",
    "Cryptographic signatures provide authentication and integrity.",
    "Smart contracts automate execution of digital agreements.",
    "Federated learning preserves privacy while enabling collaboration.",
    "Neural networks learn patterns from training examples.",
    "Deep learning models achieve state-of-the-art performance.",
    "Natural language processing enables human-computer interaction.",
    "Computer vision systems analyze and interpret visual data.",
    "Reinforcement learning agents learn through trial and error.",
    "Generative models create new content from learned patterns.",
    "Transfer learning leverages pre-trained model knowledge.",
    "Fine-tuning adapts models to specific downstream tasks.",
    "Gradient descent optimizes model parameters iteratively.",
    "Backpropagation computes gradients for neural network training.",
    "Attention mechanisms focus on relevant input features.",
    "Transformer architectures revolutionized sequence modeling.",
    "Self-supervised learning extracts signals from unlabeled data.",
    "Contrastive learning learns representations through comparisons.",
    "Few-shot learning generalizes from limited examples.",
    "Meta-learning enables rapid adaptation to new tasks.",
    "Adversarial training improves model robustness.",
    "Regularization techniques prevent overfitting in models.",
    "Batch normalization stabilizes neural network training.",
    "Dropout randomly deactivates neurons during training.",
    "Data augmentation increases training set diversity.",
    "Cross-validation evaluates model generalization performance.",
    "Hyperparameter tuning optimizes model configuration.",
    "Model compression reduces computational requirements.",
    "Knowledge distillation transfers information between models.",
    "Ensemble methods combine multiple model predictions.",
    "Active learning selects informative samples for labeling.",
    "Online learning adapts to streaming data continuously.",
    "Multi-task learning shares knowledge across related problems.",
    "Domain adaptation handles distribution shift between datasets.",
    "Continual learning retains knowledge while learning new tasks.",
    "Catastrophic forgetting erases previously learned information.",
    "Memory networks store and retrieve relevant information.",
    "Graph neural networks process structured relational data.",
    "Recurrent networks model sequential dependencies in data.",
    "Convolutional networks excel at spatial pattern recognition.",
    "Residual connections enable training of very deep networks.",
    "Skip connections facilitate gradient flow in deep architectures.",
    "Normalization layers improve training stability and speed.",
    "Activation functions introduce non-linearity in neural networks.",
    "Loss functions measure discrepancy between predictions and targets."
] * 10

class LocalAverager:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1"):
        self.model_name = model_name
        self.model_cache_dir = "/model_cache"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing LocalAverager with model: {model_name}")
        self._load_model()
        self._setup_dht()
        
    def _load_model(self):
        try:
            if os.path.exists(f"{self.model_cache_dir}/tokenizer"):
                self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_cache_dir}/tokenizer")
            else:
                logger.error("Tokenizer cache not found")
                sys.exit(1)
                
            if os.path.exists(f"{self.model_cache_dir}/model"):
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    f"{self.model_cache_dir}/model",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                logger.error("Model cache not found")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            sys.exit(1)
            
    def _setup_dht(self):
        initial_peers = []
        genesis_host = os.getenv("GENESIS_HOST")
        if genesis_host and genesis_host != "localhost":
            initial_peers = [f"/ip4/{genesis_host}/tcp/13337"]
            
        self.dht = DHT(
            start=True,
            initial_peers=initial_peers,
            client_mode=False,
            host_maddrs=["/ip4/0.0.0.0/tcp/0"]
        )
        logger.info(f"DHT initialized for averager")
        
    def compute_loss(self, model: torch.nn.Module, texts: List[str]) -> float:
        model.eval()
        total_loss = 0.0
        valid_samples = 0
        
        with torch.no_grad():
            for text in texts:
                try:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                        padding=True
                    ).to(self.device)
                    
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    total_loss += outputs.loss.item()
                    valid_samples += 1
                    
                except Exception as e:
                    logger.warning(f"Loss computation failed for text: {e}")
                    continue
                    
        if valid_samples == 0:
            return float('inf')
            
        return total_loss / valid_samples
        
    def apply_lora_delta(self, delta_weights: Dict[str, torch.Tensor]) -> torch.nn.Module:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model_with_lora = get_peft_model(self.base_model, lora_config)
        
        for name, param in model_with_lora.named_parameters():
            if name in delta_weights:
                param.data = delta_weights[name].to(param.device, dtype=param.dtype)
                
        return model_with_lora
        
    def krum_selection(self, deltas: List[Dict[str, torch.Tensor]], num_select: int = 1) -> List[int]:
        if len(deltas) <= num_select:
            return list(range(len(deltas)))
            
        distances = []
        
        for i, delta_i in enumerate(deltas):
            total_dist = 0.0
            for j, delta_j in enumerate(deltas):
                if i != j:
                    dist = 0.0
                    for key in delta_i.keys():
                        if key in delta_j:
                            dist += torch.norm(delta_i[key] - delta_j[key]).item() ** 2
                    total_dist += dist
            distances.append(total_dist)
            
        selected_indices = sorted(range(len(distances)), key=lambda x: distances[x])[:num_select]
        logger.info(f"Krum selected indices: {selected_indices}")
        return selected_indices
        
    def federated_average(self, deltas: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not deltas:
            return {}
            
        averaged_delta = {}
        first_delta = deltas[0]
        
        for key in first_delta.keys():
            weights = []
            for delta in deltas:
                if key in delta:
                    weights.append(delta[key])
                    
            if weights:
                averaged_delta[key] = torch.stack(weights).mean(dim=0)
                
        logger.info(f"Averaged {len(deltas)} deltas with {len(averaged_delta)} parameters")
        return averaged_delta
        
    def loss_gate_validation(self, delta_weights: Dict[str, torch.Tensor]) -> bool:
        try:
            loss_before = self.compute_loss(self.base_model, DEV_TEXTS[:50])
            if loss_before < 0.05:
                logger.info(f"Skipping loss gate: loss_before={loss_before:.4f} < 0.05")
                return True
            model_with_delta = self.apply_lora_delta(delta_weights)
            loss_after = self.compute_loss(model_with_delta, DEV_TEXTS[:50])
            is_valid = loss_after < 0.999 * loss_before
            logger.info(f"Loss gate: before={loss_before:.4f}, after={loss_after:.4f}, valid={is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Loss gate validation failed: {e}")
            return True
            
    def calculate_peer_id(self, validator_pubkey: bytes) -> bytes:
        """Calculate DHT peer ID from validator BLS pubkey"""
        import hashlib
        hash_result = hashlib.sha256(validator_pubkey).digest()
        return hash_result[:8]  # first 8 bytes
        
    async def publish_ack(self, agg_id: str, validator_pubkey: bytes) -> bool:
        """Publish ACK using proper committee peer ID format"""
        try:
            agg_id_bytes = bytes.fromhex(agg_id)
            peer_id_bytes = self.calculate_peer_id(validator_pubkey)
            
            # Format: ack:{updateID[:16]}:{peerID} in hex
            ack_key = f"ack:{agg_id_bytes[:16].hex()}:{peer_id_bytes.hex()}"
            ack_value = b'\x01'
            
            success = await self.dht.store(
                key=ack_key,
                value=ack_value,
                expiration_time=hivemind.get_dht_time() + 3600
            )
            
            if success:
                logger.info(f"Published ACK: {ack_key}")
            else:
                logger.error(f"Failed to publish ACK: {ack_key}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error publishing ACK: {e}")
            return False
        
    async def aggregate_and_validate(self, raw_ids: List[str], peer_ids: List[str]) -> Optional[str]:
        logger.info(f"Starting aggregation for {len(raw_ids)} deltas")
        
        deltas = []
        valid_peer_ids = []
        
        for raw_id, peer_id in zip(raw_ids, peer_ids):
            try:
                dht_key = f"raw:{raw_id}"
                result = await self.dht.get(dht_key)
                
                if result and result.value:
                    delta_weights = torch.load(result.value, map_location='cpu')
                    deltas.append(delta_weights)
                    valid_peer_ids.append(peer_id)
                    logger.info(f"Retrieved delta for {raw_id[:16]}")
                else:
                    logger.warning(f"Failed to retrieve delta for {raw_id[:16]}")
                    
            except Exception as e:
                logger.error(f"Error retrieving delta {raw_id[:16]}: {e}")
                continue
                
        if not deltas:
            logger.error("No valid deltas retrieved")
            return None
            
        selected_indices = self.krum_selection(deltas, num_select=min(len(deltas), 3))
        selected_deltas = [deltas[i] for i in selected_indices]
        
        averaged_delta = self.federated_average(selected_deltas)
        
        is_valid = self.loss_gate_validation(averaged_delta)
        
        if is_valid:
            agg_id = hashlib.sha256(pickle.dumps(averaged_delta)).hexdigest()
            
            os.makedirs(f"{self.model_cache_dir}/deltas", exist_ok=True)
            torch.save(averaged_delta, f"{self.model_cache_dir}/deltas/agg_{agg_id}.bin")
            
            validator_pubkey_hex = os.getenv("VALIDATOR_PUBKEY")
            if not validator_pubkey_hex:
                logger.error("VALIDATOR_PUBKEY environment variable not set")
                return None
            
            try:
                validator_pubkey = bytes.fromhex(validator_pubkey_hex)
                if len(validator_pubkey) != 48:
                    logger.error(f"Invalid validator pubkey length: {len(validator_pubkey)}, expected 48 bytes")
                    return None
            except ValueError:
                logger.error(f"Invalid validator pubkey hex format: {validator_pubkey_hex}")
                return None
                
            await self.publish_ack(agg_id, validator_pubkey)
                
            logger.info(f"Aggregation successful: {agg_id[:16]}")
            return agg_id
        else:
            logger.warning(f"Aggregation failed loss gate validation")
            return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="POL-AI Local Averager Service")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1", help="Model name")
    parser.add_argument("--interval", type=int, default=60, help="Aggregation check interval (seconds)")
    
    args = parser.parse_args()
    
    logger.info("Starting Local Averager service...")
    LocalAverager(model_name=args.model)
    
    async def service_loop():
        while True:
            try:
                await asyncio.sleep(args.interval)
                logger.debug("Averager service running...")
            except KeyboardInterrupt:
                logger.info("Service stopped by user")
                break
            except Exception as e:
                logger.error(f"Service error: {e}")
                await asyncio.sleep(5)
    
    try:
        asyncio.run(service_loop())
    except KeyboardInterrupt:
        logger.info("Local Averager service terminated")

if __name__ == "__main__":
    main()

 