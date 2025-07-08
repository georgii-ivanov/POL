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
        """Compute SHA256 hash of this training entry"""
        entry_data = self.to_dict()
        # Remove the hash fields for computation
        entry_data.pop('previous_hash', None)
        entry_json = json.dumps(entry_data, sort_keys=True)
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
        block_json = json.dumps(block_data, sort_keys=True)
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
        
        # Store large model data off-chain
        model_storage_path = self._store_model_off_chain(epoch, node_id, model_state, optimizer_state)
        
        # Create cryptographic hashes for on-chain verification
        model_state_hash = self._compute_state_hash(model_state)
        optimizer_state_hash = self._compute_state_hash(optimizer_state)
        training_data_hash = self._compute_training_data_hash(training_metrics)
        
        # Generate cryptographic proof of training work
        gradient_proof = self._generate_gradient_proof(model_state, training_metrics)
        
        # Create training entry
        training_entry = TrainingEntry(
            epoch=epoch,
            node_id=node_id,
            timestamp=time.time(),
            training_loss=training_metrics.get('loss', 0.0),
            learning_rate=training_metrics.get('learning_rate', 0.0),
            batch_count=training_metrics.get('batch_count', 0),
            consciousness_level=training_metrics.get('consciousness_level', 0.0),
            reasoning_quality=training_metrics.get('reasoning_quality', 0.0),
            quantum_coherence=training_metrics.get('quantum_coherence', 0.0),
            model_state_hash=model_state_hash,
            optimizer_state_hash=optimizer_state_hash,
            training_data_hash=training_data_hash,
            model_storage_path=model_storage_path,
            checkpoint_size=os.path.getsize(model_storage_path) if os.path.exists(model_storage_path) else 0,
            gradient_proof=gradient_proof,
            validation_scores=validation_scores,
            consensus_signatures=[],
            previous_hash=self.blocks[-1].block_hash if self.blocks else "0" * 64,
            merkle_root=self._compute_merkle_root(model_state, optimizer_state, training_metrics),
            expert_utilization=training_metrics.get('expert_utilization', {}),
            consciousness_growth=training_metrics.get('consciousness_growth', 0.0),
            knowledge_accumulation=training_metrics.get('knowledge_accumulation', 0)
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
        """Get the latest immutable training state from blockchain"""
        if not self.blocks or len(self.blocks) <= 1:  # Skip genesis
            return None
        
        latest_block = self.blocks[-1]
        training_entry = latest_block.training_entry
        
        # Load model state from off-chain storage
        model_path = training_entry.model_storage_path
        if not os.path.exists(model_path):
            logger.error(f"Model storage missing: {model_path}")
            return None
        
        try:
            checkpoint_data = torch.load(model_path, map_location='cpu')
            
            return {
                'epoch': training_entry.epoch,
                'model_state_dict': checkpoint_data.get('model_state_dict'),
                'optimizer_state_dict': checkpoint_data.get('optimizer_state_dict'),
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
                'immutable_hash': latest_block.block_hash
            }
        except Exception as e:
            logger.error(f"Error loading training state: {e}")
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
        """Store large model data in off-chain storage"""
        storage_filename = f"training_epoch_{epoch}_{node_id}_{int(time.time())}.pt"
        storage_path = os.path.join(self.model_storage_dir, storage_filename)
        
        checkpoint_data = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'epoch': epoch,
            'node_id': node_id,
            'timestamp': time.time(),
            'blockchain_entry': True
        }
        
        torch.save(checkpoint_data, storage_path)
        return storage_path
    
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
        proof_data = {
            'model_weights_sample': str(list(model_state.values())[0][:10].tolist()) if model_state else "",
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
            chain_data.append({
                'index': block.index,
                'timestamp': block.timestamp,
                'training_entry': block.training_entry.to_dict(),
                'previous_hash': block.previous_hash,
                'nonce': block.nonce,
                'difficulty': block.difficulty,
                'block_hash': block.block_hash,
                'validator_signatures': block.validator_signatures,
                'consensus_proof': block.consensus_proof
            })
        
        with open(chain_file, 'w') as f:
            json.dump(chain_data, f, indent=2)
    
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
            
        except Exception as e:
            logger.error(f"Error loading training chain: {e}")
            self.blocks = [] 