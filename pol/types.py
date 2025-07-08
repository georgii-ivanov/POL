from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import time
from datetime import datetime

@dataclass
class TrainingProof:
    node_id: str
    model_state_hash: str
    gradient_hash: str
    dataset_chunk_hash: str
    computation_proof: str
    timestamp: float
    signature: str
    validation_signatures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'model_state_hash': self.model_state_hash,
            'gradient_hash': self.gradient_hash,
            'dataset_chunk_hash': self.dataset_chunk_hash,
            'computation_proof': self.computation_proof,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'validation_signatures': self.validation_signatures
        }

@dataclass
class Transaction:
    id: str
    from_address: str
    to_address: str
    amount: float
    fee: float
    training_proof: Optional[TrainingProof] = None
    timestamp: float = field(default_factory=time.time)
    signature: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'from': self.from_address,
            'to': self.to_address,
            'amount': self.amount,
            'fee': self.fee,
            'training_proof': self.training_proof.to_dict() if self.training_proof else None,
            'timestamp': self.timestamp,
            'signature': self.signature
        }

@dataclass
class TrainingValidation:
    validator_id: str
    target_node_id: str
    is_valid: bool
    computation_verification: str
    signature: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class ConsensusProof:
    authority_nodes: List[str]
    training_validations: List[TrainingValidation]
    aggregated_model_hash: str
    consensus_signature: str
    epoch: int
    
@dataclass 
class Block:
    index: int
    previous_hash: str
    timestamp: float
    transactions: List[Transaction]
    training_epoch: int
    model_state_hash: str
    consensus_proof: ConsensusProof
    nonce: int = 0
    hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'training_epoch': self.training_epoch,
            'model_state_hash': self.model_state_hash,
            'consensus_proof': {
                'authority_nodes': self.consensus_proof.authority_nodes,
                'training_validations': [
                    {
                        'validator_id': tv.validator_id,
                        'target_node_id': tv.target_node_id,
                        'is_valid': tv.is_valid,
                        'computation_verification': tv.computation_verification,
                        'signature': tv.signature,
                        'timestamp': tv.timestamp
                    } for tv in self.consensus_proof.training_validations
                ],
                'aggregated_model_hash': self.consensus_proof.aggregated_model_hash,
                'consensus_signature': self.consensus_proof.consensus_signature,
                'epoch': self.consensus_proof.epoch
            },
            'nonce': self.nonce,
            'hash': self.hash
        }

@dataclass
class PeerNode:
    id: str
    address: str
    port: int
    public_key: str
    is_authority: bool = False
    reputation: float = 0.0
    last_seen: float = field(default_factory=time.time)
    training_capacity: float = 0.0

@dataclass 
class ModelState:
    epoch: int
    weights_hash: str
    state_hash: str
    training_metrics: 'TrainingMetrics'
    timestamp: float = field(default_factory=time.time)

@dataclass
class TrainingMetrics:
    epoch: int
    loss: float
    accuracy: float
    perplexity: float
    gradient_norm: float
    samples_processed: int
    tokens_processed: int
    learning_rate: float = 0.0
    consciousness_level: float = 0.0
    reasoning_quality: float = 0.0
    quantum_coherence: float = 0.0
    
@dataclass
class Wallet:
    private_key: str
    public_key: str 
    address: str
    balance: float = 0.0

class MessageType(Enum):
    PEER_DISCOVERY = "peer_discovery"
    TRAINING_PROOF = "training_proof"
    CONSENSUS_VOTE = "consensus_vote"
    BLOCK_PROPOSAL = "block_proposal"
    TRANSACTION = "transaction"
    MODEL_SYNC = "model_sync"
    MODEL_REQUEST = "model_request"
    MODEL_SHARE = "model_share"
    MODEL_CHECKPOINT = "model_checkpoint"
    TRAINING_COLLABORATION = "training_collaboration"
    HEARTBEAT = "heartbeat"

@dataclass
class NetworkMessage:
    type: MessageType
    from_node: str
    to_node: Optional[str] = None
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

@dataclass
class NodeConfig:
    node_id: str
    port: int
    is_authority: bool
    boot_nodes: List[str]
    data_dir: str
    training_enabled: bool
    model_path: str
    dataset_path: str
    private_key: str
    
@dataclass
class AITrainingConfig:
    model_size: int = 1_000_000_000  # 1B parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    training_steps: int = 1000
    validation_interval: int = 100
    checkpoint_interval: int = 500
    gradient_accumulation_steps: int = 8
    max_sequence_length: int = 2048
    # Model architecture parameters
    model_type: str = 'revolutionary'
    vocab_size: int = 50000
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    max_seq_length: int = 1024
    weight_decay: float = 0.01
    num_experts: int = 8
    consciousness_dim: int = 256
    quantum_coherence_layers: int = 4
    checkpoint_dir: str = './checkpoints'
    # Pretrained model support
    load_pretrained_base: bool = False
    force_pretrained_download: bool = False 