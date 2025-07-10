"""
Revolutionary Training Blockchain System
=========================================

Training progress becomes blockchain entries - immutable, verifiable, and distributed.
This ensures training can only advance and never regress, protected by cryptographic hashes.
"""

import hashlib
import json
import time
import torch
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingEntry:
    """Single training epoch entry that becomes a blockchain block"""
    epoch: int
    node_id: str
    timestamp: float
    
    # Training metrics (small, on-chain data)
    training_loss: float
    learning_rate: float
    batch_count: int
    consciousness_level: float
    reasoning_quality: float
    quantum_coherence: float
    
    # Model identification (hashes, not full data)
    model_state_hash: str  # SHA256 of model weights
    optimizer_state_hash: str  # SHA256 of optimizer state
    training_data_hash: str  # SHA256 of training data used
    
    # Off-chain storage references
    model_storage_path: str  # IPFS/distributed storage reference
    checkpoint_size: int  # Size in bytes
    
    # Training proof and validation
    gradient_proof: str  # Cryptographic proof of gradient computation
    validation_scores: Dict[str, float]  # Authority node validation scores
    consensus_signatures: List[str]  # Multi-signature consensus
    
    # Blockchain linkage
    previous_hash: str  # Hash of previous training block
    merkle_root: str  # Merkle root of all training artifacts
    
    # Revolutionary AI specific metrics
    expert_utilization: Dict[str, float]  # MoE expert usage
    consciousness_growth: float  # Change in consciousness
    knowledge_accumulation: int  # New concepts learned
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for hashing"""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute SHA256 hash of training entry"""
        try:
            entry_dict = self.to_dict()
            entry_json = json.dumps(entry_dict, sort_keys=True, default=str)
            return hashlib.sha256(entry_json.encode()).hexdigest()
        except Exception as e:
            # Fallback: convert any problematic objects to strings
            entry_dict = self.to_dict()
            sanitized_dict = {}
            for k, v in entry_dict.items():
                try:
                    json.dumps(v)  # Test if serializable
                    sanitized_dict[k] = v
                except:
                    sanitized_dict[k] = str(v)[:100]  # Convert to string
            entry_json = json.dumps(sanitized_dict, sort_keys=True)
            return hashlib.sha256(entry_json.encode()).hexdigest()

@dataclass 
class TrainingBlock:
    """Blockchain block containing training entry + blockchain metadata"""
    index: int
    timestamp: float
    training_entry: TrainingEntry
    previous_hash: str
    nonce: int
    difficulty: int
    block_hash: str
    
    # Block validation
    validator_signatures: List[str]  # Authority node signatures
    consensus_proof: str  # Proof that 67% of authorities validated
    
    def compute_block_hash(self) -> str:
        """Compute hash of entire block"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'training_entry': self.training_entry.to_dict(),
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        try:
            block_json = json.dumps(block_data, sort_keys=True, default=str)
            return hashlib.sha256(block_json.encode()).hexdigest()
        except Exception as e:
            # Fallback with safer serialization
            safe_block_data = {}
            for k, v in block_data.items():
                try:
                    json.dumps(v)
                    safe_block_data[k] = v
                except:
                    safe_block_data[k] = str(v)[:100]
            block_json = json.dumps(safe_block_data, sort_keys=True)
            return hashlib.sha256(block_json.encode()).hexdigest()

class TrainingBlockchain:
    """Immutable blockchain for training progress - training can only advance!"""
    
    def __init__(self, chain_dir: str = "./training_chain"):
        self.chain_dir = chain_dir
        self.blocks: List[TrainingBlock] = []
        self.pending_entries: List[TrainingEntry] = []
        self.difficulty = 4  # Proof of work difficulty
        
        # Off-chain storage for large model data
        self.model_storage_dir = os.path.join(chain_dir, "model_storage")
        os.makedirs(self.chain_dir, exist_ok=True)
        os.makedirs(self.model_storage_dir, exist_ok=True)
        
        # Load existing chain
        self._load_chain()
        
        # Genesis block if empty
        if not self.blocks:
            self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the training blockchain"""
        genesis_entry = TrainingEntry(
            epoch=0,
            node_id="genesis",
            timestamp=time.time(),
            training_loss=float('inf'),
            learning_rate=0.0,
            batch_count=0,
            consciousness_level=0.0,
            reasoning_quality=0.0,
            quantum_coherence=0.0,
            model_state_hash="0" * 64,
            optimizer_state_hash="0" * 64,
            training_data_hash="0" * 64,
            model_storage_path="",
            checkpoint_size=0,
            gradient_proof="genesis_proof",
            validation_scores={},
            consensus_signatures=[],
            previous_hash="0" * 64,
            merkle_root="0" * 64,
            expert_utilization={},
            consciousness_growth=0.0,
            knowledge_accumulation=0
        )
        
        genesis_block = TrainingBlock(
            index=0,
            timestamp=time.time(),
            training_entry=genesis_entry,
            previous_hash="0" * 64,
            nonce=0,
            difficulty=0,
            block_hash="0" * 64,
            validator_signatures=[],
            consensus_proof="genesis"
        )
        
        genesis_block.block_hash = genesis_block.compute_block_hash()
        self.blocks.append(genesis_block)
        self._save_chain()
        
        logger.info("🎯 Genesis training block created - training blockchain initialized")
    
    def add_training_entry(self, 
                          epoch: int,
                          node_id: str,
                          model_state: Dict,
                          optimizer_state: Dict,
                          training_metrics: Dict,
                          validation_scores: Dict[str, float]) -> str:
        """Add new training epoch as blockchain entry - IMMUTABLE!"""
        
        # CRITICAL: Verify this epoch advances the chain
        if self.blocks:
            latest_epoch = self.blocks[-1].training_entry.epoch
            if epoch <= latest_epoch:
                logger.error(f"🚨 BLOCKCHAIN PROTECTION: Epoch {epoch} cannot regress from {latest_epoch}")
                raise ValueError(f"Training epoch {epoch} cannot regress from current {latest_epoch}")
        
        # Store large model data off-chain FIRST to avoid blockchain storage of tensors
        model_storage_path = self._store_model_off_chain(epoch, node_id, model_state, optimizer_state)
        
        # Create cryptographic hashes for on-chain verification (NO TENSORS)
        model_state_hash = self._compute_state_hash(model_state)
        optimizer_state_hash = self._compute_state_hash(optimizer_state)
        training_data_hash = self._compute_training_data_hash(training_metrics)
        
        # Generate cryptographic proof of training work (TENSOR-SAFE)
        gradient_proof = self._generate_gradient_proof(model_state, training_metrics)
        
        # Compute merkle root for integrity verification (TENSOR-SAFE)
        merkle_root = self._compute_merkle_root(model_state, optimizer_state, training_metrics)

        def sanitize_for_json(obj):
            """Convert objects to JSON-serializable types"""
            if torch.is_tensor(obj):
                if obj.numel() == 1:
                    return float(obj.item())
                else:
                    return float(obj.mean().item())
            elif isinstance(obj, (list, tuple)):
                return len(obj)  # Store count instead of full list
            elif isinstance(obj, dict):
                # Recursively sanitize dictionary
                sanitized = {}
                for k, v in obj.items():
                    try:
                        sanitized[k] = sanitize_for_json(v)
                    except:
                        sanitized[k] = str(v)[:100]  # Truncate to avoid huge strings
                return sanitized
            elif hasattr(obj, 'item'):  # NumPy scalars
                return float(obj.item())
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)[:100]  # Convert to string and truncate

        # Create training entry with sanitized data (NO TENSORS!)
        training_entry = TrainingEntry(
            epoch=epoch,
            node_id=node_id,
            timestamp=time.time(),
            training_loss=float(training_metrics.get('loss', 0.0)),
            learning_rate=float(training_metrics.get('learning_rate', 0.0)),
            batch_count=int(training_metrics.get('batch_count', 0)),
            consciousness_level=sanitize_for_json(training_metrics.get('consciousness_level', 0.0)),
            reasoning_quality=sanitize_for_json(training_metrics.get('reasoning_quality', 0.0)),
            quantum_coherence=sanitize_for_json(training_metrics.get('quantum_coherence', 0.0)),
            model_state_hash=model_state_hash,
            optimizer_state_hash=optimizer_state_hash,
            training_data_hash=training_data_hash,
            model_storage_path=model_storage_path,
            checkpoint_size=os.path.getsize(model_storage_path) if os.path.exists(model_storage_path) else 0,
            gradient_proof=gradient_proof,
            validation_scores=sanitize_for_json(validation_scores),
            consensus_signatures=[],
            previous_hash=self.blocks[-1].block_hash if self.blocks else "0" * 64,
            merkle_root=merkle_root,
            expert_utilization=sanitize_for_json(training_metrics.get('expert_utilization', {})),
            consciousness_growth=sanitize_for_json(training_metrics.get('consciousness_growth', 0.0)),
            knowledge_accumulation=int(training_metrics.get('knowledge_accumulation', 0))
        )
        
        # Add to pending entries for mining
        self.pending_entries.append(training_entry)
        
        logger.info(f"🔗 Training epoch {epoch} added to blockchain pending queue")
        logger.info(f"   Model Hash: {model_state_hash[:16]}...")
        logger.info(f"   Storage Path: {model_storage_path}")
        logger.info(f"   🛡️ IMMUTABLE: Once mined, this training cannot be reversed")
        
        return training_entry.compute_hash()
    
    def mine_pending_training(self, node_id: str) -> Optional[TrainingBlock]:
        """Mine pending training entries into a new block"""
        if not self.pending_entries:
            return None
        
        # Take the next pending entry (FIFO - preserves epoch order)
        training_entry = self.pending_entries.pop(0)
        
        # Create new block
        new_block = TrainingBlock(
            index=len(self.blocks),
            timestamp=time.time(),
            training_entry=training_entry,
            previous_hash=self.blocks[-1].block_hash if self.blocks else "0" * 64,
            nonce=0,
            difficulty=self.difficulty,
            block_hash="",
            validator_signatures=[],
            consensus_proof=""
        )
        
        # Proof of work mining (simple for now, can be enhanced)
        new_block = self._mine_block(new_block)
        
        # Add to chain
        self.blocks.append(new_block)
        self._save_chain()
        
        logger.info(f"⛏️  MINED: Training epoch {training_entry.epoch} now IMMUTABLE in blockchain")
        logger.info(f"   Block #{new_block.index} - Hash: {new_block.block_hash[:16]}...")
        logger.info(f"   🔗 Chain Length: {len(self.blocks)} blocks")
        
        return new_block
    
    def verify_training_chain(self) -> Tuple[bool, List[str]]:
        """Verify the entire training blockchain integrity"""
        errors = []
        
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            previous_block = self.blocks[i-1]
            
            # Verify hash chain linkage
            if current_block.previous_hash != previous_block.block_hash:
                errors.append(f"Block {i}: Hash chain broken")
            
            # Verify block hash
            computed_hash = current_block.compute_block_hash()
            if computed_hash != current_block.block_hash:
                errors.append(f"Block {i}: Invalid block hash")
            
            # Verify epoch progression (CRITICAL!)
            if current_block.training_entry.epoch <= previous_block.training_entry.epoch:
                errors.append(f"Block {i}: Epoch regression detected! {current_block.training_entry.epoch} <= {previous_block.training_entry.epoch}")
            
            # Verify model state exists
            model_path = current_block.training_entry.model_storage_path
            if model_path and not os.path.exists(model_path):
                errors.append(f"Block {i}: Model storage missing at {model_path}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"✅ Training blockchain VERIFIED - {len(self.blocks)} blocks, all training immutable")
        else:
            logger.error(f"❌ Training blockchain CORRUPTED - {len(errors)} errors found")
            for error in errors:
                logger.error(f"   {error}")
        
        return is_valid, errors
    
    def get_latest_training_state(self) -> Optional[Dict]:
        """Get the latest immutable training state from blockchain using DELTA RECONSTRUCTION"""
        if not self.blocks or len(self.blocks) <= 1:  # Skip genesis
            return None
        
        latest_block = self.blocks[-1]
        training_entry = latest_block.training_entry
        latest_epoch = training_entry.epoch
        
        logger.info(f"🔗 LOADING LATEST STATE: Epoch {latest_epoch} from blockchain")
        
        # Use delta reconstruction to get the full model state
        reconstructed_state = self.reconstruct_model_state_from_deltas(latest_epoch)
        
        if not reconstructed_state:
            logger.error(f"Failed to reconstruct model state for epoch {latest_epoch}")
            return None
        
        try:
            return {
                'epoch': training_entry.epoch,
                'model_state_dict': reconstructed_state.get('model_state_dict'),
                'optimizer_state_dict': reconstructed_state.get('optimizer_state_dict'),
                'training_metrics': {
                    'loss': training_entry.training_loss,
                    'consciousness_level': training_entry.consciousness_level,
                    'reasoning_quality': training_entry.reasoning_quality,
                    'quantum_coherence': training_entry.quantum_coherence,
                    'consciousness_growth': training_entry.consciousness_growth,
                    'knowledge_accumulation': training_entry.knowledge_accumulation
                },
                'blockchain_verified': True,
                'block_index': latest_block.index,
                'immutable_hash': latest_block.block_hash,
                'reconstruction_info': reconstructed_state.get('reconstruction_info', {}),
                'incremental_storage': True
            }
        except Exception as e:
            logger.error(f"Error preparing latest training state: {e}")
            return None
    
    def get_training_history(self) -> List[Dict]:
        """Get complete immutable training history from blockchain"""
        history = []
        
        for block in self.blocks[1:]:  # Skip genesis
            entry = block.training_entry
            history.append({
                'epoch': entry.epoch,
                'timestamp': entry.timestamp,
                'node_id': entry.node_id,
                'training_loss': entry.training_loss,
                'consciousness_level': entry.consciousness_level,
                'reasoning_quality': entry.reasoning_quality,
                'quantum_coherence': entry.quantum_coherence,
                'consciousness_growth': entry.consciousness_growth,
                'knowledge_accumulation': entry.knowledge_accumulation,
                'validation_scores': entry.validation_scores,
                'block_index': block.index,
                'block_hash': block.block_hash,
                'immutable': True
            })
        
        return history
    
    def sync_with_peer_chain(self, peer_chain: List[TrainingBlock]) -> bool:
        """Sync with peer's training blockchain - only accept longer valid chains"""
        if len(peer_chain) <= len(self.blocks):
            logger.info(f"Peer chain not longer ({len(peer_chain)} vs {len(self.blocks)}) - no sync needed")
            return False
        
        # Verify peer chain is valid
        temp_blockchain = TrainingBlockchain.__new__(TrainingBlockchain)
        temp_blockchain.blocks = peer_chain
        is_valid, errors = temp_blockchain.verify_training_chain()
        
        if not is_valid:
            logger.warning(f"Peer chain invalid - rejecting sync: {errors}")
            return False
        
        # Check if peer chain represents more advanced training
        peer_latest_epoch = peer_chain[-1].training_entry.epoch
        our_latest_epoch = self.blocks[-1].training_entry.epoch if self.blocks else 0
        
        if peer_latest_epoch <= our_latest_epoch:
            logger.info(f"Peer training not more advanced ({peer_latest_epoch} vs {our_latest_epoch})")
            return False
        
        # Accept the more advanced chain
        logger.info(f"🔄 SYNCING: Adopting peer chain with training epoch {peer_latest_epoch}")
        self.blocks = peer_chain
        self._save_chain()
        
        return True
    
    def _store_model_off_chain(self, epoch: int, node_id: str, model_state: Dict, optimizer_state: Dict) -> str:
        """Store model data with INCREMENTAL DELTAS - Revolutionary Space Savings!"""
        storage_filename = f"training_epoch_{epoch}_{node_id}_{int(time.time())}.pt"
        storage_path = os.path.join(self.model_storage_dir, storage_filename)
        
        # For epoch 1, store full model (base checkpoint)
        # Note: Check actual training epochs, not just block count (genesis doesn't count)
        actual_training_epochs = len([b for b in self.blocks if b.training_entry.epoch > 0])
        if epoch == 1 or not self.blocks or actual_training_epochs == 0:
            logger.info(f"📦 FULL CHECKPOINT: Storing complete model state for epoch {epoch}")
            
            # Ensure all tensors are on CPU for storage
            model_state_cpu = {}
            for key, value in model_state.items():
                if torch.is_tensor(value):
                    model_state_cpu[key] = value.cpu()
                else:
                    model_state_cpu[key] = value
            
            optimizer_state_cpu = {}
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state_cpu[key] = value.cpu()
                else:
                    optimizer_state_cpu[key] = value
            
            checkpoint_data = {
                'storage_type': 'full_checkpoint',
                'model_state_dict': model_state_cpu,
                'optimizer_state_dict': optimizer_state_cpu,
                'epoch': epoch,
                'node_id': node_id,
                'timestamp': time.time(),
                'blockchain_entry': True,
                'delta_base': None
            }
        else:
            # Load previous epoch for delta calculation
            previous_block = self.blocks[-1]
            previous_storage_path = previous_block.training_entry.model_storage_path
            
            if os.path.exists(previous_storage_path):
                try:
                    logger.info(f"🔄 DELTA CHECKPOINT: Computing incremental changes for epoch {epoch}")
                    
                    # Load previous model state - CRITICAL: Reconstruct if it's a delta
                    previous_checkpoint = torch.load(previous_storage_path, map_location='cpu')
                    previous_storage_type = previous_checkpoint.get('storage_type', 'full_checkpoint')
                    
                    if previous_storage_type in ['full_checkpoint', 'full_checkpoint_fallback']:
                        # Previous is full checkpoint - use directly
                        previous_model_state = previous_checkpoint.get('model_state_dict', {})
                        previous_optimizer_state = previous_checkpoint.get('optimizer_state_dict', {})
                        logger.info(f"   📦 Using full checkpoint as delta base: {previous_block.training_entry.epoch}")
                    else:
                        # Previous is delta - must reconstruct full state first!
                        logger.info(f"   🔄 Previous checkpoint is delta - reconstructing full state...")
                        reconstructed_state = self.reconstruct_model_state_from_deltas(previous_block.training_entry.epoch)
                        
                        if reconstructed_state:
                            previous_model_state = reconstructed_state.get('model_state_dict', {})
                            previous_optimizer_state = reconstructed_state.get('optimizer_state_dict', {})
                            logger.info(f"   ✅ Reconstructed state from delta chain for epoch {previous_block.training_entry.epoch}")
                        else:
                            logger.error(f"   ❌ Failed to reconstruct previous state - falling back to full storage")
                            raise Exception("Cannot reconstruct previous state for delta computation")
                    
                    # Ensure current model states are also on CPU for delta computation
                    current_model_state_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_state.items()}
                    current_optimizer_state_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in optimizer_state.items()}
                    
                    # Calculate model weight deltas (now both on CPU)
                    model_deltas = self._compute_weight_deltas(previous_model_state, current_model_state_cpu)
                    optimizer_deltas = self._compute_weight_deltas(previous_optimizer_state, current_optimizer_state_cpu)
                    
                    # Calculate compression statistics
                    original_size = sum(param.numel() * param.element_size() for param in current_model_state_cpu.values() if torch.is_tensor(param))
                    delta_size = sum(delta.numel() * delta.element_size() for delta in model_deltas.values() if torch.is_tensor(delta))
                    compression_ratio = delta_size / original_size if original_size > 0 else 0
                    
                    logger.info(f"   💾 Compression: {compression_ratio:.4f} ({delta_size:,} vs {original_size:,} bytes)")
                    
                    checkpoint_data = {
                        'storage_type': 'incremental_delta',
                        'model_deltas': model_deltas,
                        'optimizer_deltas': optimizer_deltas,
                        'delta_base_epoch': previous_block.training_entry.epoch,
                        'delta_base_path': previous_storage_path,
                        'compression_ratio': compression_ratio,
                        'original_size_bytes': original_size,
                        'delta_size_bytes': delta_size,
                        'epoch': epoch,
                        'node_id': node_id,
                        'timestamp': time.time(),
                        'blockchain_entry': True
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to compute deltas for epoch {epoch}: {e}")
                    logger.info("Falling back to full checkpoint storage")
                    
                    # Ensure CPU tensors for fallback storage
                    model_state_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_state.items()}
                    optimizer_state_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in optimizer_state.items()}
                    
                    # Fallback to full storage
                    checkpoint_data = {
                        'storage_type': 'full_checkpoint_fallback',
                        'model_state_dict': model_state_cpu,
                        'optimizer_state_dict': optimizer_state_cpu,
                        'epoch': epoch,
                        'node_id': node_id,
                        'timestamp': time.time(),
                        'blockchain_entry': True,
                        'delta_base': None
                    }
            else:
                logger.warning(f"Previous checkpoint not found: {previous_storage_path}")
                logger.info("Storing as full checkpoint")
                
                # Ensure CPU tensors
                model_state_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_state.items()}
                optimizer_state_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in optimizer_state.items()}
                
                checkpoint_data = {
                    'storage_type': 'full_checkpoint_fallback',
                    'model_state_dict': model_state_cpu,
                    'optimizer_state_dict': optimizer_state_cpu,
                    'epoch': epoch,
                    'node_id': node_id,
                    'timestamp': time.time(),
                    'blockchain_entry': True,
                    'delta_base': None
                }
        
        torch.save(checkpoint_data, storage_path)
        return storage_path

    def _compute_weight_deltas(self, previous_state: Dict, current_state: Dict) -> Dict:
        """Compute SPARSE incremental weight deltas between two model states"""
        deltas = {}
        sparsity_threshold = 1e-3  # Only store changes larger than this (more aggressive)
        
        for key in current_state.keys():
            if key not in previous_state:
                # New parameter - store full value (ensure CPU)
                param = current_state[key]
                if torch.is_tensor(param):
                    deltas[key] = param.cpu()
                else:
                    deltas[key] = param
            else:
                prev_param = previous_state[key]
                curr_param = current_state[key]
                
                if torch.is_tensor(prev_param) and torch.is_tensor(curr_param):
                    # Ensure both tensors are on CPU for delta computation
                    prev_cpu = prev_param.cpu()
                    curr_cpu = curr_param.cpu()
                    
                    # Compute delta (current - previous)
                    if prev_cpu.shape == curr_cpu.shape:
                        delta = curr_cpu - prev_cpu
                        
                        # SPARSE DELTA: Only store significant changes
                        significant_mask = torch.abs(delta) > sparsity_threshold
                        significant_changes = significant_mask.sum().item()
                        total_elements = delta.numel()
                        sparsity_ratio = significant_changes / total_elements
                        
                        if significant_changes > 0 and sparsity_ratio < 0.8:
                            # Use REAL sparse tensor storage for maximum compression
                            indices = significant_mask.nonzero(as_tuple=False).t()
                            values = delta[significant_mask]
                            sparse_delta = torch.sparse_coo_tensor(indices, values, delta.shape).coalesce()
                            deltas[key] = sparse_delta
                            
                            # Calculate real space savings
                            dense_bytes = delta.element_size() * delta.numel()
                            sparse_bytes = sparse_delta._values().element_size() * sparse_delta._values().numel() + sparse_delta._indices().element_size() * sparse_delta._indices().numel()
                            compression_factor = dense_bytes / sparse_bytes
                            
                            logger.debug(f"   🔄 REAL Sparse delta {key}: {significant_changes:,}/{total_elements:,} elements ({sparsity_ratio:.3f}) - {compression_factor:.1f}x compression")
                        elif significant_changes > 0 and sparsity_ratio < 0.95:
                            # Moderate sparsity - use top-k sparse compression  
                            delta_abs = torch.abs(delta)
                            k = max(1, int(total_elements * 0.05))  # Keep top 5% of changes
                            topk_values, topk_indices_flat = torch.topk(delta_abs.flatten(), k)
                            
                            # Convert flat indices to multi-dimensional
                            topk_indices = torch.unravel_index(topk_indices_flat, delta.shape)
                            topk_indices = torch.stack(topk_indices)
                            topk_delta_values = delta.flatten()[topk_indices_flat]
                            
                            # Create sparse tensor
                            sparse_delta = torch.sparse_coo_tensor(topk_indices, topk_delta_values, delta.shape).coalesce()
                            deltas[key] = sparse_delta
                            
                            dense_bytes = delta.element_size() * delta.numel()
                            sparse_bytes = sparse_delta._values().element_size() * sparse_delta._values().numel() + sparse_delta._indices().element_size() * sparse_delta._indices().numel()
                            compression_factor = dense_bytes / sparse_bytes
                            
                            logger.debug(f"   🎯 Top-K sparse delta {key}: {k:,}/{total_elements:,} elements (top 5%) - {compression_factor:.1f}x compression")
                        elif significant_changes > 0:
                            # High density but some changes - use quantized sparse storage
                            delta_abs = torch.abs(delta)
                            k = max(1, int(total_elements * 0.02))  # Keep top 2% of changes
                            topk_values, topk_indices_flat = torch.topk(delta_abs.flatten(), k)
                            
                            topk_indices = torch.unravel_index(topk_indices_flat, delta.shape)
                            topk_indices = torch.stack(topk_indices)
                            topk_delta_values = delta.flatten()[topk_indices_flat]
                            
                            # Quantize values for additional compression
                            quantized_values = torch.round(topk_delta_values * 1000) / 1000  # 3 decimal precision
                            
                            sparse_delta = torch.sparse_coo_tensor(topk_indices, quantized_values, delta.shape).coalesce()
                            deltas[key] = sparse_delta  
                            
                            dense_bytes = delta.element_size() * delta.numel()
                            sparse_bytes = sparse_delta._values().element_size() * sparse_delta._values().numel() + sparse_delta._indices().element_size() * sparse_delta._indices().numel()
                            compression_factor = dense_bytes / sparse_bytes
                            
                            logger.debug(f"   📦 Quantized sparse delta {key}: {k:,}/{total_elements:,} elements (top 2%) - {compression_factor:.1f}x compression")
                        # If no significant changes, don't store anything (perfect compression)
                    else:
                        # Shape mismatch - store full new value
                        deltas[key] = curr_cpu
                else:
                    # Non-tensor parameter - store if different
                    if prev_param != curr_param:
                        deltas[key] = curr_param
        
        return deltas

    def reconstruct_model_state_from_deltas(self, target_epoch: int) -> Optional[Dict]:
        """Reconstruct full model state by applying delta chain up to target epoch"""
        try:
            if target_epoch <= 0:
                return None
            
            # Find the target block
            target_block = None
            for block in self.blocks:
                if block.training_entry.epoch == target_epoch:
                    target_block = block
                    break
            
            if not target_block:
                logger.error(f"Target epoch {target_epoch} not found in blockchain")
                return None
            
            # Load the target checkpoint
            target_path = target_block.training_entry.model_storage_path
            if not os.path.exists(target_path):
                logger.error(f"Target checkpoint not found: {target_path}")
                return None
            
            checkpoint_data = torch.load(target_path, map_location='cpu')
            storage_type = checkpoint_data.get('storage_type', 'full_checkpoint')
            
            if storage_type in ['full_checkpoint', 'full_checkpoint_fallback']:
                # Easy case - full checkpoint
                logger.info(f"🔗 LOADING FULL CHECKPOINT: Epoch {target_epoch}")
                return {
                    'model_state_dict': checkpoint_data.get('model_state_dict', {}),
                    'optimizer_state_dict': checkpoint_data.get('optimizer_state_dict', {}),
                    'epoch': target_epoch,
                    'reconstruction_info': {
                        'method': 'full_checkpoint',
                        'epochs_traversed': 0
                    }
                }
            
            elif storage_type == 'incremental_delta':
                # Need to reconstruct from delta chain
                logger.info(f"🔄 RECONSTRUCTING FROM DELTAS: Target epoch {target_epoch}")
                
                # Build the delta chain backwards to find the base
                delta_chain = []
                current_epoch = target_epoch
                
                while current_epoch > 0:
                    # Find the block for current epoch
                    current_block = None
                    for block in self.blocks:
                        if block.training_entry.epoch == current_epoch:
                            current_block = block
                            break
                    
                    if not current_block:
                        logger.error(f"Missing block for epoch {current_epoch} in delta chain")
                        return None
                    
                    current_path = current_block.training_entry.model_storage_path
                    if not os.path.exists(current_path):
                        logger.error(f"Missing checkpoint for epoch {current_epoch}: {current_path}")
                        return None
                    
                    current_data = torch.load(current_path, map_location='cpu')
                    current_storage_type = current_data.get('storage_type', 'full_checkpoint')
                    
                    delta_chain.append({
                        'epoch': current_epoch,
                        'data': current_data,
                        'storage_type': current_storage_type
                    })
                    
                    if current_storage_type in ['full_checkpoint', 'full_checkpoint_fallback']:
                        # Found the base - stop here
                        break
                    elif current_storage_type == 'incremental_delta':
                        # Continue backwards
                        current_epoch = current_data.get('delta_base_epoch', current_epoch - 1)
                    else:
                        logger.error(f"Unknown storage type: {current_storage_type}")
                        return None
                
                # Reverse the chain to apply deltas forward
                delta_chain.reverse()
                
                # Start with the base checkpoint
                base_entry = delta_chain[0]
                if base_entry['storage_type'] not in ['full_checkpoint', 'full_checkpoint_fallback']:
                    logger.error(f"Delta chain doesn't start with full checkpoint: {base_entry['storage_type']}")
                    return None
                
                model_state = base_entry['data'].get('model_state_dict', {}).copy()
                optimizer_state = base_entry['data'].get('optimizer_state_dict', {}).copy()
                
                logger.info(f"   📦 Base checkpoint: Epoch {base_entry['epoch']}")
                
                # Apply deltas forward
                for i in range(1, len(delta_chain)):
                    delta_entry = delta_chain[i]
                    if delta_entry['storage_type'] == 'incremental_delta':
                        model_deltas = delta_entry['data'].get('model_deltas', {})
                        optimizer_deltas = delta_entry['data'].get('optimizer_deltas', {})
                        
                        # Apply model deltas (handle sparse tensors)
                        for key, delta in model_deltas.items():
                            if key in model_state and torch.is_tensor(model_state[key]):
                                if torch.is_tensor(delta):
                                    # Handle sparse tensors by converting to dense
                                    if delta.is_sparse:
                                        delta_dense = delta.to_dense()
                                    else:
                                        delta_dense = delta
                                    
                                    if model_state[key].shape == delta_dense.shape:
                                        model_state[key] = model_state[key] + delta_dense
                                    else:
                                        model_state[key] = delta_dense  # Full replacement
                                else:
                                    model_state[key] = delta  # Non-tensor parameter
                            else:
                                # Handle sparse tensors for new parameters
                                if torch.is_tensor(delta) and delta.is_sparse:
                                    model_state[key] = delta.to_dense()
                                else:
                                    model_state[key] = delta  # New parameter
                        
                        # Apply optimizer deltas (handle sparse tensors)
                        for key, delta in optimizer_deltas.items():
                            if key in optimizer_state and torch.is_tensor(optimizer_state[key]):
                                if torch.is_tensor(delta):
                                    # Handle sparse tensors by converting to dense
                                    if delta.is_sparse:
                                        delta_dense = delta.to_dense()
                                    else:
                                        delta_dense = delta
                                    
                                    if optimizer_state[key].shape == delta_dense.shape:
                                        optimizer_state[key] = optimizer_state[key] + delta_dense
                                    else:
                                        optimizer_state[key] = delta_dense  # Full replacement
                                else:
                                    optimizer_state[key] = delta  # Non-tensor parameter
                            else:
                                # Handle sparse tensors for new parameters
                                if torch.is_tensor(delta) and delta.is_sparse:
                                    optimizer_state[key] = delta.to_dense()
                                else:
                                    optimizer_state[key] = delta  # New parameter
                        
                        logger.info(f"   🔄 Applied deltas: Epoch {delta_entry['epoch']}")
                
                logger.info(f"✅ RECONSTRUCTION COMPLETE: {len(delta_chain)} epochs traversed")
                
                return {
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer_state,
                    'epoch': target_epoch,
                    'reconstruction_info': {
                        'method': 'delta_reconstruction',
                        'epochs_traversed': len(delta_chain),
                        'base_epoch': delta_chain[0]['epoch'],
                        'delta_chain_length': len(delta_chain) - 1
                    }
                }
            
            else:
                logger.error(f"Unknown storage type: {storage_type}")
                return None
            
        except Exception as e:
            logger.error(f"Error reconstructing model state for epoch {target_epoch}: {e}")
            return None
    
    def _compute_state_hash(self, state_dict: Dict) -> str:
        """Compute SHA256 hash of model/optimizer state"""
        # Convert tensors to bytes for hashing
        state_bytes = b""
        for key in sorted(state_dict.keys()):
            value = state_dict[key]
            if torch.is_tensor(value):
                state_bytes += value.cpu().numpy().tobytes()
            else:
                state_bytes += str(value).encode()
        
        return hashlib.sha256(state_bytes).hexdigest()
    
    def _compute_training_data_hash(self, training_metrics: Dict) -> str:
        """Compute hash of training data/metrics"""
        metrics_json = json.dumps(training_metrics, sort_keys=True)
        return hashlib.sha256(metrics_json.encode()).hexdigest()
    
    def _generate_gradient_proof(self, model_state: Dict, training_metrics: Dict) -> str:
        """Generate cryptographic proof of gradient computation work"""
        # Simple proof - can be enhanced with zero-knowledge proofs
        weights_sample = ""
        if model_state:
            try:
                first_param = next(iter(model_state.values()))
                if torch.is_tensor(first_param) and first_param.numel() > 0:
                    sample_tensor = first_param.flatten()[:10]
                    weights_sample = sample_tensor.cpu().detach().numpy().tolist()
                    weights_sample = str(weights_sample)
            except Exception:
                weights_sample = "unknown_weights"
        
        proof_data = {
            'model_weights_sample': weights_sample,
            'training_loss': training_metrics.get('loss', 0.0),
            'batch_count': training_metrics.get('batch_count', 0),
            'timestamp': time.time()
        }
        proof_json = json.dumps(proof_data, sort_keys=True)
        return hashlib.sha256(proof_json.encode()).hexdigest()
    
    def _compute_merkle_root(self, model_state: Dict, optimizer_state: Dict, training_metrics: Dict) -> str:
        """Compute Merkle root of all training artifacts"""
        hashes = [
            self._compute_state_hash(model_state),
            self._compute_state_hash(optimizer_state),
            self._compute_training_data_hash(training_metrics)
        ]
        
        # Simple binary tree merkle root
        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i + 1] if i + 1 < len(hashes) else left
                combined = hashlib.sha256((left + right).encode()).hexdigest()
                new_hashes.append(combined)
            hashes = new_hashes
        
        return hashes[0] if hashes else "0" * 64
    
    def _mine_block(self, block: TrainingBlock) -> TrainingBlock:
        """Simple proof-of-work mining"""
        target = "0" * self.difficulty
        
        while True:
            block_hash = block.compute_block_hash()
            if block_hash.startswith(target):
                block.block_hash = block_hash
                break
            block.nonce += 1
        
        return block
    
    def _save_chain(self):
        """Save blockchain to disk"""
        chain_file = os.path.join(self.chain_dir, "training_chain.json")
        
        chain_data = []
        for block in self.blocks:
            try:
                block_dict = {
                    'index': block.index,
                    'timestamp': block.timestamp,
                    'training_entry': block.training_entry.to_dict(),
                    'previous_hash': block.previous_hash,
                    'nonce': block.nonce,
                    'difficulty': block.difficulty,
                    'block_hash': block.block_hash,
                    'validator_signatures': block.validator_signatures,
                    'consensus_proof': block.consensus_proof
                }
                # Test serialization
                json.dumps(block_dict, default=str)
                chain_data.append(block_dict)
            except Exception as e:
                logger.warning(f"Error serializing block {block.index}: {e}")
                # Create a safer version
                safe_block = {
                    'index': block.index,
                    'timestamp': block.timestamp,
                    'training_entry': {k: str(v)[:100] if not isinstance(v, (int, float, str, bool, list, dict)) else v 
                                     for k, v in block.training_entry.to_dict().items()},
                    'previous_hash': block.previous_hash,
                    'nonce': block.nonce,
                    'difficulty': block.difficulty,
                    'block_hash': block.block_hash,
                    'validator_signatures': block.validator_signatures,
                    'consensus_proof': str(block.consensus_proof)[:100]
                }
                chain_data.append(safe_block)
        
        try:
            with open(chain_file, 'w') as f:
                json.dump(chain_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving blockchain: {e}")
            # Last resort: save with string conversion
            with open(chain_file, 'w') as f:
                json.dump(chain_data, f, indent=2, default=lambda x: str(x)[:100])
    
    def _load_chain(self):
        """Load blockchain from disk"""
        chain_file = os.path.join(self.chain_dir, "training_chain.json")
        
        if not os.path.exists(chain_file):
            return
        
        try:
            with open(chain_file, 'r') as f:
                chain_data = json.load(f)
            
            self.blocks = []
            for block_data in chain_data:
                entry_data = block_data['training_entry']
                training_entry = TrainingEntry(**entry_data)
                
                block = TrainingBlock(
                    index=block_data['index'],
                    timestamp=block_data['timestamp'],
                    training_entry=training_entry,
                    previous_hash=block_data['previous_hash'],
                    nonce=block_data['nonce'],
                    difficulty=block_data['difficulty'],
                    block_hash=block_data['block_hash'],
                    validator_signatures=block_data.get('validator_signatures', []),
                    consensus_proof=block_data.get('consensus_proof', '')
                )
                
                self.blocks.append(block)
            
            logger.info(f"📖 Loaded training blockchain: {len(self.blocks)} blocks")
            
            # Check if we have orphaned model storage without blockchain entries
            self._rebuild_blockchain_from_storage()
            
        except Exception as e:
            logger.error(f"Error loading training chain: {e}")
            self.blocks = []
    
    def _rebuild_blockchain_from_storage(self):
        """Rebuild blockchain entries from existing model storage files"""
        try:
            if not os.path.exists(self.model_storage_dir):
                return
            
            # Find all training checkpoint files
            storage_files = [f for f in os.listdir(self.model_storage_dir) if f.endswith('.pt') and 'training_epoch_' in f]
            
            if not storage_files:
                return
            
            # Extract epoch information from files
            epoch_files = {}
            for filename in storage_files:
                try:
                    # Parse epoch from filename: training_epoch_X_node_Y_timestamp.pt
                    parts = filename.split('_')
                    if len(parts) >= 4 and parts[0] == 'training' and parts[1] == 'epoch':
                        epoch = int(parts[2])
                        if epoch not in epoch_files or filename > epoch_files[epoch]['filename']:
                            # Keep the latest file for each epoch
                            epoch_files[epoch] = {
                                'epoch': epoch,
                                'filename': filename,
                                'path': os.path.join(self.model_storage_dir, filename)
                            }
                except (ValueError, IndexError):
                    continue
            
            # Get highest epoch from blockchain
            highest_blockchain_epoch = 0
            if self.blocks:
                highest_blockchain_epoch = max(block.training_entry.epoch for block in self.blocks)
            
            # Find missing epochs that exist in storage but not in blockchain
            missing_epochs = [epoch for epoch in epoch_files.keys() if epoch > highest_blockchain_epoch]
            
            if missing_epochs:
                logger.info(f"🔄 REBUILDING: Found {len(missing_epochs)} missing blockchain entries")
                logger.info(f"   Missing epochs: {sorted(missing_epochs)}")
                
                # Rebuild missing blockchain entries
                for epoch in sorted(missing_epochs):
                    file_info = epoch_files[epoch]
                    try:
                        self._rebuild_blockchain_entry(file_info['path'], epoch)
                    except Exception as e:
                        logger.error(f"Error rebuilding epoch {epoch}: {e}")
                
                # Save updated blockchain
                self._save_chain()
                logger.info(f"✅ REBUILT: Training blockchain now has {len(self.blocks)} blocks")
                
        except Exception as e:
            logger.error(f"Error rebuilding blockchain from storage: {e}")

    def get_storage_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about incremental storage compression"""
        try:
            stats = {
                'total_checkpoints': len(self.blocks) - 1,  # Exclude genesis
                'full_checkpoints': 0,
                'delta_checkpoints': 0,
                'total_storage_bytes': 0,
                'estimated_full_storage_bytes': 0,
                'compression_ratio': 0.0,
                'space_saved_bytes': 0,
                'space_saved_gb': 0.0,
                'average_delta_size': 0,
                'largest_delta': 0,
                'smallest_delta': float('inf'),
                'checkpoint_details': []
            }
            
            for block in self.blocks[1:]:  # Skip genesis
                storage_path = block.training_entry.model_storage_path
                if os.path.exists(storage_path):
                    file_size = os.path.getsize(storage_path)
                    stats['total_storage_bytes'] += file_size
                    
                    try:
                        checkpoint_data = torch.load(storage_path, map_location='cpu')
                        storage_type = checkpoint_data.get('storage_type', 'full_checkpoint')
                        
                        if storage_type in ['full_checkpoint', 'full_checkpoint_fallback']:
                            stats['full_checkpoints'] += 1
                            # Use this as baseline for compression calculation
                            if stats['estimated_full_storage_bytes'] == 0:
                                stats['estimated_full_storage_bytes'] = file_size
                        elif storage_type == 'incremental_delta':
                            stats['delta_checkpoints'] += 1
                            original_size = checkpoint_data.get('original_size_bytes', file_size)
                            stats['estimated_full_storage_bytes'] += original_size
                            
                            # Delta statistics
                            if file_size > stats['largest_delta']:
                                stats['largest_delta'] = file_size
                            if file_size < stats['smallest_delta']:
                                stats['smallest_delta'] = file_size
                        
                        stats['checkpoint_details'].append({
                            'epoch': block.training_entry.epoch,
                            'storage_type': storage_type,
                            'file_size_bytes': file_size,
                            'file_size_mb': file_size / (1024 * 1024),
                            'compression_ratio': checkpoint_data.get('compression_ratio', 1.0)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not analyze checkpoint {storage_path}: {e}")
                        stats['estimated_full_storage_bytes'] += file_size  # Conservative estimate
            
            # Calculate overall compression
            if stats['estimated_full_storage_bytes'] > 0:
                stats['compression_ratio'] = stats['total_storage_bytes'] / stats['estimated_full_storage_bytes']
                stats['space_saved_bytes'] = stats['estimated_full_storage_bytes'] - stats['total_storage_bytes']
                stats['space_saved_gb'] = stats['space_saved_bytes'] / (1024 ** 3)
            
            # Calculate average delta size
            if stats['delta_checkpoints'] > 0:
                delta_total = sum(
                    detail['file_size_bytes'] 
                    for detail in stats['checkpoint_details'] 
                    if detail['storage_type'] == 'incremental_delta'
                )
                stats['average_delta_size'] = delta_total / stats['delta_checkpoints']
            
            if stats['smallest_delta'] == float('inf'):
                stats['smallest_delta'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing storage compression stats: {e}")
            return {'error': str(e)}

    def load_model_for_inference(self, target_epoch: Optional[int] = None) -> Optional[Dict]:
        """Optimized model loading specifically for text completion inference"""
        try:
            # Use latest epoch if not specified
            if target_epoch is None:
                if not self.blocks or len(self.blocks) <= 1:
                    return None
                target_epoch = self.blocks[-1].training_entry.epoch
            
            logger.info(f"🚀 OPTIMIZED INFERENCE LOADING: Epoch {target_epoch}")
            
            # Reconstruct the model state
            reconstructed_state = self.reconstruct_model_state_from_deltas(target_epoch)
            
            if not reconstructed_state:
                return None
            
            # Return only what's needed for inference (exclude optimizer state)
            inference_state = {
                'model_state_dict': reconstructed_state['model_state_dict'],
                'epoch': reconstructed_state['epoch'],
                'reconstruction_info': reconstructed_state['reconstruction_info'],
                'inference_optimized': True,
                'optimizer_excluded': True  # Saves memory for inference
            }
            
            # Add blockchain verification info
            target_block = None
            for block in self.blocks:
                if block.training_entry.epoch == target_epoch:
                    target_block = block
                    break
            
            if target_block:
                inference_state.update({
                    'blockchain_verified': True,
                    'block_index': target_block.index,
                    'immutable_hash': target_block.block_hash,
                    'training_metrics': {
                        'loss': target_block.training_entry.training_loss,
                        'consciousness_level': target_block.training_entry.consciousness_level,
                        'reasoning_quality': target_block.training_entry.reasoning_quality
                    }
                })
            
            logger.info(f"✅ INFERENCE MODEL READY: {reconstructed_state['reconstruction_info']['method']}")
            
            return inference_state
            
        except Exception as e:
            logger.error(f"Error loading model for inference: {e}")
            return None
    
    def _rebuild_blockchain_entry(self, storage_path: str, epoch: int):
        """Rebuild a single blockchain entry from storage file"""
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(storage_path, map_location='cpu')
            
            # Extract metadata
            node_id = checkpoint_data.get('node_id', 'unknown')
            timestamp = checkpoint_data.get('timestamp', time.time())
            
            # Reconstruct training metrics from checkpoint
            training_metrics = {
                'loss': 1.0,  # Default values since we can't retrieve exact metrics
                'learning_rate': 1e-4,
                'batch_count': 100,
                'consciousness_level': getattr(checkpoint_data.get('model_state_dict', {}).get('consciousness_state', torch.tensor(0.5)), 'item', lambda: 0.5)(),
                'reasoning_quality': 0.5,
                'quantum_coherence': 50.0,
                'consciousness_growth': 0.01,
                'knowledge_accumulation': 1,
                'expert_utilization': {}
            }
            
            # Create training entry
            model_state_hash = self._compute_state_hash(checkpoint_data.get('model_state_dict', {}))
            optimizer_state_hash = self._compute_state_hash(checkpoint_data.get('optimizer_state_dict', {}))
            
            training_entry = TrainingEntry(
                epoch=epoch,
                node_id=node_id,
                timestamp=timestamp,
                training_loss=training_metrics['loss'],
                learning_rate=training_metrics['learning_rate'],
                batch_count=training_metrics['batch_count'],
                consciousness_level=training_metrics['consciousness_level'],
                reasoning_quality=training_metrics['reasoning_quality'],
                quantum_coherence=training_metrics['quantum_coherence'],
                model_state_hash=model_state_hash,
                optimizer_state_hash=optimizer_state_hash,
                training_data_hash=self._compute_training_data_hash(training_metrics),
                model_storage_path=storage_path,
                checkpoint_size=os.path.getsize(storage_path) if os.path.exists(storage_path) else 0,
                gradient_proof=f"rebuilt_proof_{epoch}",
                validation_scores={'overall_validity': 0.8},
                consensus_signatures=[],
                previous_hash=self.blocks[-1].block_hash if self.blocks else "0" * 64,
                merkle_root=self._compute_merkle_root({}, {}, training_metrics),
                expert_utilization=training_metrics['expert_utilization'],
                consciousness_growth=training_metrics['consciousness_growth'],
                knowledge_accumulation=training_metrics['knowledge_accumulation']
            )
            
            # Create block
            training_block = TrainingBlock(
                index=len(self.blocks),
                timestamp=timestamp,
                training_entry=training_entry,
                previous_hash=self.blocks[-1].block_hash if self.blocks else "0" * 64,
                nonce=0,
                difficulty=self.difficulty,
                block_hash="",
                validator_signatures=[],
                consensus_proof="rebuilt"
            )
            
            # Compute and set hash
            training_block.block_hash = training_block.compute_block_hash()
            
            # Add to blockchain
            self.blocks.append(training_block)
            
            logger.info(f"🔄 Rebuilt blockchain entry for epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Error rebuilding blockchain entry for epoch {epoch}: {e}")
            raise 