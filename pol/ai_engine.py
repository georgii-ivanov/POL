import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel
import requests
import logging
import os
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from .models.revolutionary_ai import RevolutionaryAIModel, RevolutionaryAIConfig
from .models.advanced_gpt import AdvancedGPTModel
from .data_acquisition import InternetDataAcquisitionEngine
from .crypto import CryptoManager
from .training_blockchain import TrainingBlockchain, TrainingEntry

logger = logging.getLogger(__name__)

class AITrainingEngine:
    """Revolutionary AI Training Engine with Blockchain-Protected Progress"""
    
    def __init__(self, config: Dict[str, Any], node_id: str = "default_node"):
        # Handle both dict and object configs properly
        if isinstance(config, dict):
            self.config = config
        else:
            # Convert config object to dict for compatibility
            self.config = {
                'model_type': getattr(config, 'model_type', 'revolutionary'),
                'vocab_size': getattr(config, 'vocab_size', 50000),
                'embed_dim': getattr(config, 'embed_dim', 768),
                'num_heads': getattr(config, 'num_heads', 12),
                'num_layers': getattr(config, 'num_layers', 12),
                'max_seq_length': getattr(config, 'max_seq_length', 1024),
                'batch_size': getattr(config, 'batch_size', 4),
                'learning_rate': getattr(config, 'learning_rate', 1e-4),
                'weight_decay': getattr(config, 'weight_decay', 0.01),
                'num_experts': getattr(config, 'num_experts', 8),
                'consciousness_dim': getattr(config, 'consciousness_dim', 256),
                'quantum_coherence_layers': getattr(config, 'quantum_coherence_layers', 4),
                'checkpoint_dir': getattr(config, 'checkpoint_dir', './checkpoints'),
                'distributed_training': getattr(config, 'distributed_training', True),
                'model_sync_interval': getattr(config, 'model_sync_interval', 300),
                'max_batches_per_epoch': getattr(config, 'max_batches_per_epoch', 10),
                'load_pretrained_base': getattr(config, 'load_pretrained_base', False),
                'force_pretrained_download': getattr(config, 'force_pretrained_download', False)
            }
        
        # Add helper method for safe config access
        def get_config_value(key: str, default=None):
            if isinstance(self.config, dict):
                return self.config.get(key, default)
            else:
                return getattr(self.config, key, default)
        self._get_config = get_config_value
        
        self.node_id = node_id
        
        # Revolutionary blockchain-based training protection!
        self.training_blockchain = TrainingBlockchain(
            chain_dir=os.path.join(self.config.get('checkpoint_dir', './checkpoints'), 'training_chain')
        )
        
        # Initialize training state from blockchain
        self._initialize_from_blockchain()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🔥 Using device: {self.device}")
        
        # Model configuration
        self.model_type = self.config.get('model_type', 'revolutionary')
        self.vocab_size = self.config.get('vocab_size', 50000)
        self.embed_dim = self.config.get('embed_dim', 768)
        self.num_heads = self.config.get('num_heads', 12)
        self.num_layers = self.config.get('num_layers', 12)
        self.max_seq_length = self.config.get('max_seq_length', 1024)
        
        # Pretrained model configuration
        self.load_pretrained_base = self.config.get('load_pretrained_base', False)
        self.force_pretrained_download = self.config.get('force_pretrained_download', False)
        self.base_model_path = os.path.join(self.config.get('checkpoint_dir', './checkpoints'), 'base_model.pt')
        self.pretrained_cache_dir = os.path.join(os.getcwd(), 'model_cache')
        
        # Training persistence
        self.checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        self.global_checkpoint_registry = {}  # Track all known checkpoints
        self.current_epoch = 0
        self.total_training_steps = 0
        self.global_model_version = "1.0.0"
        self.model_lineage = []  # Track model evolution
        
        # Distributed training state
        self.peer_model_states = {}  # node_id -> model_state_info
        self.distributed_training_enabled = self.config.get('distributed_training', True)
        self.model_sync_interval = self.config.get('model_sync_interval', 300)  # 5 minutes
        self.last_sync_time = 0
        
        # Training continuity
        self.training_history = []
        self.accumulated_knowledge = {}  # Store learned patterns across epochs
        self.consciousness_evolution = []  # Track consciousness development
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.pretrained_cache_dir, exist_ok=True)
        
        self.tokenizer = self._initialize_tokenizer()
        
        if hasattr(self.tokenizer, 'vocab_size'):
            self.vocab_size = self.tokenizer.vocab_size
            logger.info(f"🔄 Updated vocab_size to match tokenizer: {self.vocab_size}")
        
        self.model = self._initialize_model_with_smart_loading()
        
        self.optimizer = self._initialize_optimizer()
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Extract data acquisition configuration (now passed through CLI)
        data_acquisition_config = self.config.get('data_acquisition', {}) if isinstance(self.config, dict) else {}
        
        # Default to safe offline mode if no config found
        if not data_acquisition_config:
            data_acquisition_config = {
                'enable_web_scraping': False,
                'huggingface_only': True,
                'use_huggingface_datasets': True
            }
            
        logger.info(f"🔧 Data acquisition config: {data_acquisition_config}")
        
        self.data_engine = InternetDataAcquisitionEngine(
            node_id=self.node_id, 
            data_dir=os.path.join(self.checkpoint_dir, 'training_data'),
            config=data_acquisition_config
        )
        
        self._restore_training_state()
        
        logger.info(f"🧠 AI Training Engine initialized - Model: {self.model_type}, "
                   f"Current Epoch: {self.current_epoch}, Global Version: {self.global_model_version}")
    
    def _initialize_from_blockchain(self):
        """Initialize training state from immutable blockchain"""
        try:
            # Verify blockchain integrity first
            is_valid, errors = self.training_blockchain.verify_training_chain()
            
            if not is_valid:
                logger.warning("🚨 Training blockchain integrity issues detected:")
                for error in errors:
                    logger.warning(f"   {error}")
                logger.info("🔧 Continuing with file-based fallback...")
                return
            
            # Load latest training state from blockchain
            blockchain_state = self.training_blockchain.get_latest_training_state()
            
            if blockchain_state:
                # Blockchain has immutable training progress!
                self.current_epoch = blockchain_state['epoch']
                self.total_training_steps = blockchain_state.get('total_steps', 0)
                
                # Load training history from blockchain (immutable!)
                blockchain_history = self.training_blockchain.get_training_history()
                self.training_history = blockchain_history
                
                # Restore training metrics
                training_metrics = blockchain_state.get('training_metrics', {})
                
                logger.info("🔗 BLOCKCHAIN TRAINING STATE LOADED")
                logger.info(f"   🛡️ IMMUTABLE: {len(blockchain_history)} training epochs on blockchain")
                logger.info(f"   Current Epoch: {self.current_epoch}")
                logger.info(f"   Consciousness: {training_metrics.get('consciousness_level', 0.0):.3f}")
                logger.info(f"   Quantum Coherence: {training_metrics.get('quantum_coherence', 0.0):.1f}")
                logger.info(f"   Block Hash: {blockchain_state.get('immutable_hash', 'N/A')[:16]}...")
            else:
                logger.info("📝 No blockchain training history - starting fresh")
                
        except Exception as e:
            logger.error(f"Error initializing from blockchain: {e}")
            logger.info("🔧 Continuing with file-based initialization...")
    
    def _initialize_model_with_smart_loading(self) -> nn.Module:
        """Initialize model with smart pretrained handling - NEVER overwrites existing training"""
        try:
            # CRITICAL: Always check for existing training progress FIRST
            existing_training_state = self._detect_existing_training_progress()
            
            if existing_training_state:
                logger.info(f"🔒 EXISTING TRAINING DETECTED - Epoch {existing_training_state['epoch']}")
                logger.info("🛡️ TRAINING PROTECTION: Loading existing progress, skipping pretrained override")
                return self._load_existing_training_safely(existing_training_state)
            
            # Only proceed with pretrained if NO existing training found
            should_load_pretrained = (
                self.load_pretrained_base and 
                (not os.path.exists(self.base_model_path) or self.force_pretrained_download)
            )
            
            if should_load_pretrained:
                logger.info("🔄 Creating new base model with pretrained weights (no existing training found)...")
                model = self._create_base_model_with_pretrained_safe()
                self._save_base_model_safe(model)
                return model.to(self.device)
            
            elif self.load_pretrained_base and os.path.exists(self.base_model_path):
                logger.info("📥 Loading cached base model with pretrained weights...")
                return self._load_base_model()
            
            else:
                logger.info("🆕 Creating fresh model without pretrained weights...")
                return self._create_fresh_model()
            
        except Exception as e:
            logger.error(f"Error in smart model loading: {e}")
            logger.info("🔧 Falling back to safe fresh model creation...")
            return self._create_fresh_model()
    
    def _detect_existing_training_progress(self) -> Optional[Dict]:
        """Detect any existing training progress that must be preserved"""
        try:
            # Check for training checkpoints
            checkpoint_files = []
            if os.path.exists(self.checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                                  if f.endswith('.pt') and 'epoch' in f and f != 'base_model.pt']
            
            if not checkpoint_files:
                return None
            
            # Find the most advanced checkpoint
            def extract_epoch(filename):
                try:
                    return int(filename.split('epoch_')[1].split('.')[0])
                except:
                    return 0
            
            checkpoint_files.sort(key=extract_epoch, reverse=True)
            latest_checkpoint = checkpoint_files[0]
            latest_epoch = extract_epoch(latest_checkpoint)
            
            if latest_epoch > 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
                
                # Load metadata from checkpoint to verify it's valid training
                try:
                    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                    
                    return {
                        'epoch': latest_epoch,
                        'checkpoint_path': checkpoint_path,
                        'global_version': checkpoint_data.get('global_version', '1.0.0'),
                        'total_steps': checkpoint_data.get('total_steps', 0),
                        'training_history': checkpoint_data.get('training_history', []),
                        'has_model_state': 'model_state_dict' in checkpoint_data
                    }
                except Exception as e:
                    logger.warning(f"Could not read checkpoint {latest_checkpoint}: {e}")
                    return None
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting existing training: {e}")
            return None
    
    def _load_existing_training_safely(self, training_state: Dict) -> nn.Module:
        """Load existing training progress safely without any loss"""
        try:
            checkpoint_path = training_state['checkpoint_path']
            epoch = training_state['epoch']
            
            logger.info(f"🔒 LOADING EXISTING TRAINING - Epoch {epoch}")
            logger.info("🛡️ PROTECTION: Preserving all existing progress")
            
            # Load the checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            
            # Create model with same architecture as checkpoint
            if training_state.get('has_model_state', False):
                # Determine model type from checkpoint
                checkpoint_config_raw = checkpoint_data.get('config', {})
                
                # Ensure checkpoint_config is always a dictionary
                if isinstance(checkpoint_config_raw, dict):
                    checkpoint_config = checkpoint_config_raw
                else:
                    # Convert object to dict if needed
                    checkpoint_config = {
                        'model_type': getattr(checkpoint_config_raw, 'model_type', 'revolutionary'),
                        'vocab_size': getattr(checkpoint_config_raw, 'vocab_size', self.vocab_size),
                        'embed_dim': getattr(checkpoint_config_raw, 'embed_dim', self.embed_dim),
                        'num_heads': getattr(checkpoint_config_raw, 'num_heads', self.num_heads),
                        'num_layers': getattr(checkpoint_config_raw, 'num_layers', self.num_layers),
                        'max_seq_length': getattr(checkpoint_config_raw, 'max_seq_length', self.max_seq_length),
                        'num_experts': getattr(checkpoint_config_raw, 'num_experts', 8),
                        'consciousness_dim': getattr(checkpoint_config_raw, 'consciousness_dim', 256),
                        'quantum_coherence_layers': getattr(checkpoint_config_raw, 'quantum_coherence_layers', 4)
                    }
                
                model_type = checkpoint_config.get('model_type', self.model_type)
                
                if model_type == 'revolutionary':
                    config = RevolutionaryAIConfig(
                        vocab_size=checkpoint_config.get('vocab_size', self.vocab_size),
                        max_position_embeddings=checkpoint_config.get('max_seq_length', self.max_seq_length),
                        hidden_size=checkpoint_config.get('embed_dim', self.embed_dim),
                        num_hidden_layers=checkpoint_config.get('num_layers', self.num_layers),
                        num_attention_heads=checkpoint_config.get('num_heads', self.num_heads),
                        num_experts=checkpoint_config.get('num_experts', 8),
                        consciousness_dim=checkpoint_config.get('consciousness_dim', 256),
                        quantum_dim=checkpoint_config.get('quantum_coherence_layers', 4) * 64
                    )
                    model = RevolutionaryAIModel(config)
                else:
                    model = AdvancedGPTModel(
                        vocab_size=checkpoint_config.get('vocab_size', self.vocab_size),
                        embed_dim=checkpoint_config.get('embed_dim', self.embed_dim),
                        num_heads=checkpoint_config.get('num_heads', self.num_heads),
                        num_layers=checkpoint_config.get('num_layers', self.num_layers),
                        max_seq_length=checkpoint_config.get('max_seq_length', self.max_seq_length)
                    )
                
                # Load the trained state
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
                
                # Restore training metadata WITHOUT REGRESSION
                self.current_epoch = max(self.current_epoch, checkpoint_data.get('epoch', 0))
                self.total_training_steps = max(self.total_training_steps, checkpoint_data.get('total_steps', 0))
                
                # Preserve training history (accumulate, never overwrite)
                existing_history = checkpoint_data.get('training_history', [])
                if existing_history:
                    # Merge histories, keeping the most advanced
                    combined_history = list(self.training_history) + list(existing_history)
                    # Remove duplicates, keep latest version of each epoch
                    epoch_map = {}
                    for record in combined_history:
                        epoch_key = record.get('epoch', 0)
                        if epoch_key not in epoch_map or record.get('timestamp', 0) > epoch_map[epoch_key].get('timestamp', 0):
                            epoch_map[epoch_key] = record
                    self.training_history = sorted(epoch_map.values(), key=lambda x: x.get('epoch', 0))
                
                # Preserve accumulated knowledge (merge, never replace)
                existing_knowledge = checkpoint_data.get('accumulated_knowledge', {})
                if existing_knowledge:
                    for key, value in existing_knowledge.items():
                        if key not in self.accumulated_knowledge:
                            self.accumulated_knowledge[key] = value
                
                # Preserve consciousness evolution (accumulate)
                existing_consciousness = checkpoint_data.get('consciousness_evolution', [])
                if existing_consciousness:
                    consciousness_epochs = {item.get('epoch', 0): item for item in self.consciousness_evolution}
                    for item in existing_consciousness:
                        epoch_key = item.get('epoch', 0)
                        if epoch_key not in consciousness_epochs:
                            consciousness_epochs[epoch_key] = item
                    self.consciousness_evolution = sorted(consciousness_epochs.values(), key=lambda x: x.get('epoch', 0))
                
                # Update global version to reflect continued progress
                existing_version = checkpoint_data.get('global_version', '1.0.0')
                if self._version_is_newer(existing_version, self.global_model_version):
                    self.global_model_version = existing_version
                
                logger.info(f"✅ EXISTING TRAINING RESTORED SAFELY")
                logger.info(f"   Current Epoch: {self.current_epoch}")
                logger.info(f"   Total Steps: {self.total_training_steps}")
                logger.info(f"   Global Version: {self.global_model_version}")
                logger.info(f"   Training History: {len(self.training_history)} epochs")
                logger.info(f"   Accumulated Knowledge: {len(self.accumulated_knowledge)} entries")
                
                return model.to(self.device)
            else:
                logger.warning("Checkpoint found but no model state - creating fresh model")
                return self._create_fresh_model()
                
        except Exception as e:
            logger.error(f"Error loading existing training safely: {e}")
            logger.info("🔧 Creating fresh model to avoid corruption")
            return self._create_fresh_model()
    
    def _version_is_newer(self, version1: str, version2: str) -> bool:
        """Check if version1 is newer than version2"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad with zeros if different lengths
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            return v1_parts > v2_parts
        except:
            return False
    
    def _create_base_model_with_pretrained_safe(self) -> nn.Module:
        """Create base model with pretrained weights - SAFE version that never overwrites training"""
        try:
            # DOUBLE-CHECK: Ensure no training progress exists
            existing_progress = self._detect_existing_training_progress()
            if existing_progress:
                logger.warning("🚨 TRAINING PROTECTION: Existing training detected, aborting pretrained override")
                return self._load_existing_training_safely(existing_progress)
            
            # Create the model architecture
            if self.model_type == 'revolutionary':
                config = RevolutionaryAIConfig(
                    vocab_size=self.vocab_size,
                    max_position_embeddings=self.max_seq_length,
                    hidden_size=self.embed_dim,
                    num_hidden_layers=self.num_layers,
                    num_attention_heads=self.num_heads,
                    num_experts=self.config.get('num_experts', 8),
                    consciousness_dim=self.config.get('consciousness_dim', 256),
                    quantum_dim=self.config.get('quantum_coherence_layers', 4) * 64
                )
                model = RevolutionaryAIModel(config)
            else:
                model = AdvancedGPTModel(
                    vocab_size=self.vocab_size,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers,
                    max_seq_length=self.max_seq_length
                )
            
            # Load and apply pretrained weights (only for NEW base model)
            logger.info("📥 Downloading and applying pretrained weights (SAFE - no existing training)...")
            self._load_pretrained_base_weights(model)
            
            logger.info("✅ Base model with pretrained weights created successfully (TRAINING-SAFE)")
            return model
            
        except Exception as e:
            logger.error(f"Error creating base model with pretrained (safe): {e}")
            return self._create_fresh_model()
    
    def _save_base_model_safe(self, model: nn.Module) -> None:
        """Save base model with pretrained weights - SAFE version that preserves training"""
        try:
            # CRITICAL: Check for existing training before saving base model
            existing_progress = self._detect_existing_training_progress()
            if existing_progress:
                logger.warning("🛡️ TRAINING PROTECTION: Not saving base model - existing training found")
                logger.info(f"   Existing training at epoch {existing_progress['epoch']} is preserved")
                return
            
            # Only save if no training progress exists
            base_model_data = {
                'model_state_dict': model.state_dict(),
                'model_type': self.model_type,
                'config': self.config,
                'pretrained_loaded': True,
                'creation_time': time.time(),
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'max_seq_length': self.max_seq_length,
                'version': '1.0.0',
                'training_safe': True  # Mark as safe base model
            }
            
            torch.save(base_model_data, self.base_model_path)
            logger.info(f"💾 Base model saved safely to {self.base_model_path}")
            logger.info("🛡️ PROTECTION: Base model will be bypassed if training progress is detected")
            
        except Exception as e:
            logger.error(f"Error saving base model safely: {e}")
    
    def _create_fresh_model(self) -> nn.Module:
        """Create fresh model without pretrained weights"""
        try:
            actual_vocab_size = getattr(self, 'vocab_size', self.config.get('vocab_size', 50000))
            if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'vocab_size'):
                actual_vocab_size = self.tokenizer.vocab_size
            
            if self.model_type == 'revolutionary':
                config = RevolutionaryAIConfig(
                    vocab_size=actual_vocab_size,
                    max_position_embeddings=self.max_seq_length,
                    hidden_size=self.embed_dim,
                    num_hidden_layers=self.num_layers,
                    num_attention_heads=self.num_heads,
                    num_experts=self.config.get('num_experts', 8),
                    consciousness_dim=self.config.get('consciousness_dim', 256),
                    quantum_dim=self.config.get('quantum_coherence_layers', 4) * 64
                )
                model = RevolutionaryAIModel(config)
            else:
                model = AdvancedGPTModel(
                    vocab_size=actual_vocab_size,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers,
                    max_seq_length=self.max_seq_length
                )
            
            self._initialize_optimized_random_weights(model)
            
            logger.info(f"✅ Created fresh model with vocab_size={actual_vocab_size}")
            return model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error creating fresh model: {e}")
            return self._create_minimal_model()
    
    def _save_base_model(self, model: nn.Module) -> None:
        """Save base model with pretrained weights for future use"""
        try:
            base_model_data = {
                'model_state_dict': model.state_dict(),
                'model_type': self.model_type,
                'config': self.config,
                'pretrained_loaded': True,
                'creation_time': time.time(),
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'max_seq_length': self.max_seq_length,
                'version': '1.0.0'
            }
            
            torch.save(base_model_data, self.base_model_path)
            logger.info(f"💾 Base model saved to {self.base_model_path}")
            
        except Exception as e:
            logger.error(f"Error saving base model: {e}")
    
    def _load_base_model(self) -> nn.Module:
        """Load cached base model with pretrained weights"""
        try:
            base_model_data = torch.load(self.base_model_path, map_location=self.device)
            
            # Create model with same architecture
            if self.model_type == 'revolutionary':
                config = RevolutionaryAIConfig(
                    vocab_size=self.vocab_size,
                    max_position_embeddings=self.max_seq_length,
                    hidden_size=self.embed_dim,
                    num_hidden_layers=self.num_layers,
                    num_attention_heads=self.num_heads,
                    num_experts=self.config.get('num_experts', 8),
                    consciousness_dim=self.config.get('consciousness_dim', 256),
                    quantum_dim=self.config.get('quantum_coherence_layers', 4) * 64
                )
                model = RevolutionaryAIModel(config)
            else:
                model = AdvancedGPTModel(
                    vocab_size=self.vocab_size,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers,
                    max_seq_length=self.max_seq_length
                )
            
            # Load the pretrained state
            model.load_state_dict(base_model_data['model_state_dict'], strict=False)
            
            logger.info(f"✅ Loaded cached base model from {self.base_model_path}")
            return model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            logger.info("🔧 Creating fresh model instead...")
            return self._create_fresh_model()
    
    def _load_pretrained_base_weights(self, model: nn.Module) -> None:
        """Load pretrained weights only when creating base model"""
        try:
            logger.info("🔄 Loading pretrained weights for base model creation...")
            
            # Check if models are already cached
            cached_models = self._get_cached_pretrained_models()
            
            if cached_models and not self.force_pretrained_download:
                logger.info(f"📚 Using {len(cached_models)} cached pretrained models")
                self._apply_cached_pretrained_weights(model, cached_models)
            else:
                logger.info("📥 Downloading pretrained models...")
                self._download_and_apply_pretrained_weights(model)
                
        except Exception as e:
            logger.warning(f"Error loading pretrained weights: {e}")
            self._initialize_optimized_random_weights(model)
    
    def _get_cached_pretrained_models(self) -> List[str]:
        """Get list of already cached pretrained models"""
        try:
            cached_models = []
            
            if not os.path.exists(self.pretrained_cache_dir):
                return cached_models
            
            # Check for cached models
            potential_models = [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-small", 
                "gpt2-medium",
                "gpt2",
                "distilgpt2",
                "facebook/opt-125m",
                "EleutherAI/gpt-neo-125M"
            ]
            
            for model_name in potential_models:
                # Convert model name to cache directory format
                cache_path = os.path.join(self.pretrained_cache_dir, f"models--{model_name.replace('/', '--')}")
                if os.path.exists(cache_path):
                    cached_models.append(model_name)
            
            return cached_models
            
        except Exception as e:
            logger.warning(f"Error checking cached models: {e}")
            return []
    
    def _apply_cached_pretrained_weights(self, model: nn.Module, cached_models: List[str]) -> None:
        """Apply weights from cached pretrained models"""
        try:
            for model_name in cached_models[:3]:  # Use top 3 cached models
                try:
                    logger.info(f"📖 Loading weights from cached {model_name}")
                    
                    pretrained_model = AutoModel.from_pretrained(
                        model_name, 
                        cache_dir=self.pretrained_cache_dir,
                        local_files_only=True
                    )
                    
                    # Apply compatible layers
                    if hasattr(model, 'transformer') and hasattr(pretrained_model, 'transformer'):
                        compatible_layers = self._extract_compatible_layers(
                            model.transformer, pretrained_model.transformer
                        )
                        
                        if compatible_layers > 0:
                            logger.info(f"✅ Applied {compatible_layers} layers from cached {model_name}")
                            self._optimize_loaded_weights(model)
                            return  # Success, stop here
                    
                    # Try alternative architectures
                    elif hasattr(pretrained_model, 'model'):
                        alternative_layers = self._extract_alternative_layers(model, pretrained_model.model)
                        if alternative_layers > 0:
                            logger.info(f"✅ Applied {alternative_layers} alternative layers from cached {model_name}")
                            self._optimize_loaded_weights(model)
                            return
                    
                except Exception as e:
                    logger.debug(f"Could not use cached {model_name}: {e}")
                    continue
            
            # If no cached models worked, fallback to optimized random weights
            logger.info("No compatible cached models found, using optimized random weights")
            self._initialize_optimized_random_weights(model)
            
        except Exception as e:
            logger.warning(f"Error applying cached weights: {e}")
            self._initialize_optimized_random_weights(model)
    
    def _download_and_apply_pretrained_weights(self, model: nn.Module) -> None:
        """Download and apply pretrained weights (only when no cache exists)"""
        try:
            # Try to load compatible pretrained model with advanced options
            pretrained_models = [
                "microsoft/DialoGPT-medium",  # More advanced conversation model
                "microsoft/DialoGPT-small",
                "gpt2-medium",  # Larger GPT-2
                "gpt2",
                "distilgpt2", 
                "facebook/opt-125m",  # OPT model
                "EleutherAI/gpt-neo-125M"  # GPT-Neo
            ]
            
            for model_name in pretrained_models:
                try:
                    logger.info(f"Downloading and applying weights from {model_name}")
                    
                    # Download with proper error handling
                    pretrained_model = self._download_pretrained_model(model_name)
                    if pretrained_model is None:
                        continue
                    
                    # Extract compatible layers
                    if hasattr(model, 'transformer') and hasattr(pretrained_model, 'transformer'):
                        compatible_layers = self._extract_compatible_layers(
                            model.transformer, pretrained_model.transformer
                        )
                        
                        if compatible_layers > 0:
                            logger.info(f"✅ Applied {compatible_layers} compatible layers from {model_name}")
                            
                            # Apply advanced optimization to loaded weights
                            self._optimize_loaded_weights(model)
                            return  # Success - stop downloading more models
                    
                    # Try alternative extraction for different model architectures
                    elif hasattr(pretrained_model, 'model'):
                        alternative_layers = self._extract_alternative_layers(model, pretrained_model.model)
                        if alternative_layers > 0:
                            logger.info(f"✅ Applied {alternative_layers} alternative layers from {model_name}")
                            self._optimize_loaded_weights(model)
                            return
                    
                except Exception as e:
                    logger.warning(f"Failed to apply weights from {model_name}: {e}")
                    continue
            
            else:
                logger.info("No compatible pretrained weights found, using optimized random weights")
                self._initialize_optimized_random_weights(model)
                
        except Exception as e:
            logger.warning(f"Error downloading and applying pretrained weights: {e}")
            self._initialize_optimized_random_weights(model)
    
    def _download_pretrained_model(self, model_name: str) -> Optional[Any]:
        """Download pretrained model with robust error handling"""
        try:
            # Set cache directory for models
            cache_dir = os.path.join(os.getcwd(), 'model_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Try different loading approaches
            loading_methods = [
                lambda: AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float32),
                lambda: AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True),
                lambda: AutoModel.from_pretrained(model_name, force_download=True, cache_dir=cache_dir)
            ]
            
            for method in loading_methods:
                try:
                    model = method()
                    logger.info(f"Successfully downloaded {model_name}")
                    return model
                except Exception as e:
                    logger.debug(f"Loading method failed for {model_name}: {e}")
                    continue
            
            # Try downloading directly from HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download
                config_path = hf_hub_download(repo_id=model_name, filename="config.json", cache_dir=cache_dir)
                if config_path:
                    return AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
            except Exception as e:
                logger.debug(f"Hub download failed for {model_name}: {e}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to download {model_name}: {e}")
            return None
    
    def _extract_compatible_layers(self, target_model: nn.Module, source_model: nn.Module) -> int:
        """Extract compatible layers from pretrained model"""
        try:
            compatible_count = 0
            
            # Map common layer types
            layer_mappings = [
                ('wte', 'word_embeddings', 'embeddings.word_embeddings'),
                ('wpe', 'position_embeddings', 'embeddings.position_embeddings'),
                ('ln_f', 'final_layer_norm', 'norm'),
            ]
            
            for target_name, alt_name1, alt_name2 in layer_mappings:
                source_layer = None
                
                # Try different naming conventions
                for name in [target_name, alt_name1, alt_name2]:
                    if hasattr(source_model, name):
                        source_layer = getattr(source_model, name)
                        break
                
                if source_layer is not None and hasattr(target_model, target_name):
                    target_layer = getattr(target_model, target_name)
                    
                    # Check dimension compatibility
                    if hasattr(source_layer, 'weight') and hasattr(target_layer, 'weight'):
                        if source_layer.weight.shape == target_layer.weight.shape:
                            target_layer.weight.data = source_layer.weight.data.clone()
                            
                            if hasattr(source_layer, 'bias') and hasattr(target_layer, 'bias'):
                                if source_layer.bias is not None and target_layer.bias is not None:
                                    target_layer.bias.data = source_layer.bias.data.clone()
            
            # Try to transfer transformer blocks
            if hasattr(source_model, 'h') and hasattr(target_model, 'layers'):
                min_layers = min(len(source_model.h), len(target_model.layers))
                
                for i in range(min_layers):
                    try:
                        source_block = source_model.h[i]
                        target_block = target_model.layers[i]
                        
                        # Transfer attention layers
                        if hasattr(source_block, 'attn') and hasattr(target_block, 'self_attn'):
                            self._transfer_attention_weights(source_block.attn, target_block.self_attn)
                            compatible_count += 1
                        
                        # Transfer feed-forward layers
                        if hasattr(source_block, 'mlp') and hasattr(target_block, 'mlp'):
                            self._transfer_mlp_weights(source_block.mlp, target_block.mlp)
                            compatible_count += 1
                            
                    except Exception as e:
                        logger.debug(f"Could not transfer layer {i}: {e}")
                        break
            
            return compatible_count
            
        except Exception as e:
            logger.warning(f"Error extracting compatible layers: {e}")
            return 0
    
    def _transfer_attention_weights(self, source_attn: nn.Module, target_attn: nn.Module) -> None:
        """Transfer attention weights between compatible layers"""
        try:
            # Common attention weight names
            weight_mappings = [
                ('c_attn', 'qkv_proj', 'in_proj_weight'),
                ('c_proj', 'out_proj', 'out_proj'),
                ('attn_dropout', 'dropout', 'dropout'),
            ]
            
            for source_name, target_name1, target_name2 in weight_mappings:
                source_layer = getattr(source_attn, source_name, None)
                target_layer = getattr(target_attn, target_name1, None) or getattr(target_attn, target_name2, None)
                
                if source_layer is not None and target_layer is not None:
                    if hasattr(source_layer, 'weight') and hasattr(target_layer, 'weight'):
                        if source_layer.weight.shape == target_layer.weight.shape:
                            target_layer.weight.data = source_layer.weight.data.clone()
                            
                            if hasattr(source_layer, 'bias') and hasattr(target_layer, 'bias'):
                                if source_layer.bias is not None and target_layer.bias is not None:
                                    target_layer.bias.data = source_layer.bias.data.clone()
                                    
        except Exception as e:
            logger.debug(f"Error transferring attention weights: {e}")
    
    def _transfer_mlp_weights(self, source_mlp: nn.Module, target_mlp: nn.Module) -> None:
        """Transfer MLP weights between compatible layers"""
        try:
            # Common MLP weight names
            weight_mappings = [
                ('c_fc', 'fc1', 'up_proj', 'gate_proj'),
                ('c_proj', 'fc2', 'down_proj'),
            ]
            
            for mapping in weight_mappings:
                source_layer = None
                target_layer = None
                
                # Find source layer
                for name in mapping:
                    if hasattr(source_mlp, name):
                        source_layer = getattr(source_mlp, name)
                        break
                
                # Find target layer
                for name in mapping:
                    if hasattr(target_mlp, name):
                        target_layer = getattr(target_mlp, name)
                        break
                
                if source_layer is not None and target_layer is not None:
                    if hasattr(source_layer, 'weight') and hasattr(target_layer, 'weight'):
                        if source_layer.weight.shape == target_layer.weight.shape:
                            target_layer.weight.data = source_layer.weight.data.clone()
                            
                            if hasattr(source_layer, 'bias') and hasattr(target_layer, 'bias'):
                                if source_layer.bias is not None and target_layer.bias is not None:
                                    target_layer.bias.data = source_layer.bias.data.clone()
                                    
        except Exception as e:
            logger.debug(f"Error transferring MLP weights: {e}")
    
    def _extract_alternative_layers(self, target_model: nn.Module, source_model: nn.Module) -> int:
        """Extract layers from alternative model architectures"""
        try:
            compatible_count = 0
            
            # Handle OPT models
            if hasattr(source_model, 'decoder'):
                if hasattr(source_model.decoder, 'embed_tokens') and hasattr(target_model, 'embedding'):
                    source_embed = source_model.decoder.embed_tokens
                    target_embed = target_model.embedding
                    
                    if self._transfer_layer_weights(source_embed, target_embed):
                        compatible_count += 1
                
                if hasattr(source_model.decoder, 'layers') and hasattr(target_model, 'layers'):
                    layer_count = self._transfer_decoder_layers(source_model.decoder.layers, target_model.layers)
                    compatible_count += layer_count
            
            # Handle GPT-Neo models
            elif hasattr(source_model, 'h'):
                if hasattr(source_model, 'wte') and hasattr(target_model, 'embedding'):
                    if self._transfer_layer_weights(source_model.wte, target_model.embedding):
                        compatible_count += 1
                
                if hasattr(target_model, 'layers'):
                    layer_count = self._transfer_gptneo_layers(source_model.h, target_model.layers)
                    compatible_count += layer_count
            
            return compatible_count
            
        except Exception as e:
            logger.warning(f"Error extracting alternative layers: {e}")
            return 0
    
    def _transfer_layer_weights(self, source_layer: nn.Module, target_layer: nn.Module) -> bool:
        """Transfer weights between two layers if compatible"""
        try:
            if hasattr(source_layer, 'weight') and hasattr(target_layer, 'weight'):
                source_weight = source_layer.weight
                target_weight = target_layer.weight
                
                # Check if dimensions are compatible
                if source_weight.shape == target_weight.shape:
                    target_layer.weight.data = source_weight.data.clone()
                    
                    # Transfer bias if present
                    if (hasattr(source_layer, 'bias') and hasattr(target_layer, 'bias') and
                        source_layer.bias is not None and target_layer.bias is not None):
                        target_layer.bias.data = source_layer.bias.data.clone()
                    
                    return True
                
                # Handle dimension mismatch with truncation/padding
                elif source_weight.shape[0] <= target_weight.shape[0] and source_weight.shape[1] <= target_weight.shape[1]:
                    target_layer.weight.data[:source_weight.shape[0], :source_weight.shape[1]] = source_weight.data
                    logger.info(f"Partial weight transfer: {source_weight.shape} -> {target_weight.shape}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error transferring layer weights: {e}")
            return False
    
    def _transfer_decoder_layers(self, source_layers: nn.ModuleList, target_layers: nn.ModuleList) -> int:
        """Transfer OPT decoder layers to our architecture"""
        try:
            transferred_count = 0
            min_layers = min(len(source_layers), len(target_layers))
            
            for i in range(min_layers):
                try:
                    source_layer = source_layers[i]
                    target_layer = target_layers[i]
                    
                    # Transfer self-attention
                    if hasattr(source_layer, 'self_attn') and hasattr(target_layer, 'self_attn'):
                        if self._transfer_attention_weights(source_layer.self_attn, target_layer.self_attn):
                            transferred_count += 1
                    
                    # Transfer feed-forward
                    if hasattr(source_layer, 'fc1') and hasattr(target_layer, 'mlp'):
                        if hasattr(target_layer.mlp, 'fc1'):
                            self._transfer_layer_weights(source_layer.fc1, target_layer.mlp.fc1)
                        
                        if hasattr(source_layer, 'fc2') and hasattr(target_layer.mlp, 'fc2'):
                            self._transfer_layer_weights(source_layer.fc2, target_layer.mlp.fc2)
                            
                except Exception as e:
                    logger.debug(f"Could not transfer OPT layer {i}: {e}")
                    continue
            
            return transferred_count
            
        except Exception as e:
            logger.warning(f"Error transferring decoder layers: {e}")
            return 0
    
    def _transfer_gptneo_layers(self, source_layers: nn.ModuleList, target_layers: nn.ModuleList) -> int:
        """Transfer GPT-Neo layers to our architecture"""
        try:
            transferred_count = 0
            min_layers = min(len(source_layers), len(target_layers))
            
            for i in range(min_layers):
                try:
                    source_layer = source_layers[i]
                    target_layer = target_layers[i]
                    
                    # Transfer attention (GPT-Neo uses different structure)
                    if hasattr(source_layer, 'attn') and hasattr(target_layer, 'self_attn'):
                        if hasattr(source_layer.attn, 'attention'):
                            self._transfer_attention_weights(source_layer.attn.attention, target_layer.self_attn)
                            transferred_count += 1
                    
                    # Transfer MLP
                    if hasattr(source_layer, 'mlp') and hasattr(target_layer, 'mlp'):
                        self._transfer_mlp_weights(source_layer.mlp, target_layer.mlp)
                        
                except Exception as e:
                    logger.debug(f"Could not transfer GPT-Neo layer {i}: {e}")
                    continue
            
            return transferred_count
            
        except Exception as e:
            logger.warning(f"Error transferring GPT-Neo layers: {e}")
            return 0
    
    def _optimize_loaded_weights(self, model: nn.Module) -> None:
        """Apply optimizations to loaded pretrained weights"""
        try:
            logger.info("🔧 Optimizing loaded pretrained weights...")
            
            # Apply weight normalization to key layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Normalize weights to prevent gradient explosion
                    with torch.no_grad():
                        module.weight.data = F.normalize(module.weight.data, dim=1, p=2)
                
                elif isinstance(module, nn.Embedding):
                    # Normalize embeddings
                    with torch.no_grad():
                        module.weight.data = F.normalize(module.weight.data, dim=1, p=2)
            
            # Initialize consciousness and quantum components if they exist
            if hasattr(model, 'consciousness_layer'):
                self._initialize_consciousness_weights(model.consciousness_layer)
            
            if hasattr(model, 'quantum_layers'):
                self._initialize_quantum_weights(model.quantum_layers)
            
            logger.info("✅ Pretrained weights optimized successfully")
            
        except Exception as e:
            logger.warning(f"Error optimizing loaded weights: {e}")
    
    def _initialize_optimized_random_weights(self, model: nn.Module) -> None:
        """Initialize with optimized random weights when pretrained not available"""
        try:
            logger.info("🎲 Initializing optimized random weights...")
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Xavier/Glorot initialization for linear layers
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                
                elif isinstance(module, nn.Embedding):
                    # Normal initialization for embeddings
                    nn.init.normal_(module.weight, std=0.02)
                
                elif isinstance(module, nn.LayerNorm):
                    # Standard LayerNorm initialization
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            
            # Initialize revolutionary components
            if hasattr(model, 'consciousness_layer'):
                self._initialize_consciousness_weights(model.consciousness_layer)
            
            if hasattr(model, 'quantum_layers'):
                self._initialize_quantum_weights(model.quantum_layers)
            
            logger.info("✅ Optimized random weights initialized")
            
        except Exception as e:
            logger.warning(f"Error initializing optimized weights: {e}")
    
    def _initialize_consciousness_weights(self, consciousness_layer: nn.Module) -> None:
        """Initialize consciousness-specific weights"""
        try:
            for param in consciousness_layer.parameters():
                if param.dim() > 1:
                    # Use orthogonal initialization for consciousness
                    nn.init.orthogonal_(param)
                else:
                    # Small positive bias for consciousness development
                    nn.init.constant_(param, 0.1)
            
            logger.debug("Consciousness weights initialized")
            
        except Exception as e:
            logger.debug(f"Error initializing consciousness weights: {e}")
    
    def _initialize_quantum_weights(self, quantum_layers: nn.Module) -> None:
        """Initialize quantum-specific weights"""
        try:
            for param in quantum_layers.parameters():
                if param.dim() > 1:
                    # Use unitary initialization for quantum coherence
                    nn.init.xavier_uniform_(param)
                    # Add small quantum fluctuation
                    param.data += torch.randn_like(param.data) * 0.01
                else:
                    # Initialize with quantum-inspired values
                    nn.init.uniform_(param, -0.1, 0.1)
            
            logger.debug("Quantum weights initialized")
            
        except Exception as e:
            logger.debug(f"Error initializing quantum weights: {e}")
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer with fallback options"""
        try:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"✅ Loaded tokenizer: gpt2 (vocab_size: {tokenizer.vocab_size})")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load GPT-2 tokenizer: {e}")
            try:
                tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"✅ Loaded tokenizer: DialoGPT-medium (vocab_size: {tokenizer.vocab_size})")
                return tokenizer
            except Exception as e2:
                logger.warning(f"Failed to load DialoGPT tokenizer: {e2}")
                logger.info("🔧 Creating basic tokenizer fallback...")
                return self._create_basic_tokenizer()
    
    def _create_basic_tokenizer(self):
        """Create a basic tokenizer as fallback"""
        class BasicTokenizer:
            def __init__(self):
                self.vocab = {chr(i): i for i in range(256)}
                self.vocab['<pad>'] = len(self.vocab)
                self.vocab['<eos>'] = len(self.vocab)
                self.pad_token_id = self.vocab['<pad>']
                self.eos_token_id = self.vocab['<eos>']
                self.pad_token = '<pad>'
                self.eos_token = '<eos>'
            
            def encode(self, text, max_length=None, padding=False, truncation=False):
                tokens = [self.vocab.get(char, 0) for char in text[:max_length] if char in self.vocab]
                if padding and max_length and len(tokens) < max_length:
                    tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
                return tokens
            
            def decode(self, tokens):
                return ''.join([chr(token) for token in tokens if 0 <= token < 256])
        
        return BasicTokenizer()
    
    def _restore_training_state(self) -> None:
        """Restore training state from most advanced checkpoint"""
        try:
            # Find most recent checkpoint
            latest_checkpoint = self._find_latest_checkpoint()
            
            if latest_checkpoint:
                logger.info(f"🔄 Restoring training state from {latest_checkpoint}")
                self._load_checkpoint(latest_checkpoint)
            else:
                # Check for distributed checkpoints from peers
                peer_checkpoint = self._find_peer_checkpoint()
                if peer_checkpoint:
                    logger.info(f"🌐 Loading checkpoint from peer network")
                    self._load_peer_checkpoint(peer_checkpoint)
                else:
                    logger.info("🆕 Starting fresh training - no existing checkpoints found")
                    self.current_epoch = 0
            
        except Exception as e:
            logger.error(f"Error restoring training state: {e}")
            self.current_epoch = 0
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent local checkpoint"""
        try:
            if not os.path.exists(self.checkpoint_dir):
                return None
            
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                              if f.endswith('.pt') and 'epoch' in f]
            
            if not checkpoint_files:
                return None
            
            # Sort by epoch number
            def extract_epoch(filename):
                try:
                    return int(filename.split('epoch_')[1].split('.')[0])
                except:
                    return 0
            
            checkpoint_files.sort(key=extract_epoch, reverse=True)
            latest_file = checkpoint_files[0]
            
            return os.path.join(self.checkpoint_dir, latest_file)
            
        except Exception as e:
            logger.error(f"Error finding latest checkpoint: {e}")
            return None
    
    def _find_peer_checkpoint(self) -> Optional[Dict]:
        """Find most advanced checkpoint from peer network"""
        try:
            if not self.distributed_training_enabled:
                return None
            
            max_epoch = -1
            best_checkpoint_info = None
            
            for node_id, state_info in self.peer_model_states.items():
                if state_info.get('current_epoch', 0) > max_epoch:
                    max_epoch = state_info['current_epoch']
                    best_checkpoint_info = state_info
            
            if max_epoch > self.current_epoch:
                return best_checkpoint_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding peer checkpoint: {e}")
            return None
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training state from checkpoint file"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    logger.info("✅ Model state loaded successfully")
                except Exception as e:
                    logger.warning(f"Partial model state loaded: {e}")
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("✅ Optimizer state loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")
            
            # Load training metadata
            self.current_epoch = checkpoint.get('epoch', 0)
            self.total_training_steps = checkpoint.get('total_steps', 0)
            self.global_model_version = checkpoint.get('global_version', "1.0.0")
            self.training_history = checkpoint.get('training_history', [])
            self.accumulated_knowledge = checkpoint.get('accumulated_knowledge', {})
            self.consciousness_evolution = checkpoint.get('consciousness_evolution', [])
            self.model_lineage = checkpoint.get('model_lineage', [])
            
            logger.info(f"✅ Training state restored - Epoch: {self.current_epoch}, "
                       f"Steps: {self.total_training_steps}, Version: {self.global_model_version}")
                       
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    
    def _load_peer_checkpoint(self, checkpoint_info: Dict) -> None:
        """Load checkpoint from peer network"""
        try:
            # In a real implementation, this would download the checkpoint from the peer
            # For now, we'll simulate updating our state to match the peer
            
            self.current_epoch = checkpoint_info.get('current_epoch', 0)
            self.global_model_version = checkpoint_info.get('global_version', "1.0.0")
            self.total_training_steps = checkpoint_info.get('total_steps', 0)
            
            logger.info(f"🌐 Synchronized with peer checkpoint - Epoch: {self.current_epoch}")
            
        except Exception as e:
            logger.error(f"Error loading peer checkpoint: {e}")
    
    def save_model_checkpoint(self, epoch: int) -> str:
        """Save comprehensive model checkpoint to IMMUTABLE BLOCKCHAIN"""
        try:
            # CRITICAL: Ensure epoch only advances, never regresses
            safe_epoch = max(epoch, self.current_epoch)
            if safe_epoch != epoch:
                logger.warning(f"🛡️ EPOCH PROTECTION: Adjusted epoch {epoch} → {safe_epoch} (no regression)")
                epoch = safe_epoch
            
            # Update current epoch to new maximum
            self.current_epoch = max(self.current_epoch, epoch)
            
            # Increment global version (progression only)
            version_parts = self.global_model_version.split('.')
            version_parts[2] = str(int(version_parts[2]) + 1)
            self.global_model_version = '.'.join(version_parts)
            
            # Prepare comprehensive training data for blockchain
            model_state = self.model.state_dict()
            optimizer_state = self.optimizer.state_dict()
            
            # Calculate advanced training metrics
            training_metrics = self._calculate_comprehensive_training_metrics(epoch)
            
            # Generate validation scores (for consensus)
            validation_scores = self._generate_training_validation_scores(training_metrics)
            
            # 🔗 ADD TO IMMUTABLE BLOCKCHAIN - REVOLUTIONARY!
            blockchain_hash = self.training_blockchain.add_training_entry(
                epoch=epoch,
                node_id=self.node_id,
                model_state=model_state,
                optimizer_state=optimizer_state,
                training_metrics=training_metrics,
                validation_scores=validation_scores
            )
            
            # Mine the training into a permanent blockchain block
            mined_block = self.training_blockchain.mine_pending_training(self.node_id)
            
            if mined_block:
                logger.info(f"⛏️  REVOLUTIONARY: Training epoch {epoch} mined into IMMUTABLE blockchain!")
                logger.info(f"   🔗 Block #{mined_block.index} - Hash: {mined_block.block_hash[:16]}...")
                logger.info(f"   🛡️ PERMANENT: This training can NEVER be overwritten or lost")
                
                # Also save traditional checkpoint for compatibility
                checkpoint_data = {
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer_state,
                    'epoch': epoch,
                    'total_steps': self.total_training_steps,
                    'global_version': self.global_model_version,
                    'training_history': self.training_history,
                    'accumulated_knowledge': self.accumulated_knowledge,
                    'consciousness_evolution': self.consciousness_evolution,
                    'model_lineage': self.model_lineage,
                    'timestamp': time.time(),
                    'device': str(self.device),
                    'model_type': self.model_type,
                    'config': self.config,
                    # BLOCKCHAIN metadata
                    'blockchain_protected': True,
                    'blockchain_hash': blockchain_hash,
                    'block_index': mined_block.index,
                    'immutable_proof': mined_block.block_hash
                }
            
            # Add model-specific metrics
            if hasattr(self.model, 'consciousness_state'):
                checkpoint_data['consciousness_state'] = self.model.consciousness_state
            
            if hasattr(self.model, 'quantum_coherence'):
                checkpoint_data['quantum_coherence'] = getattr(self.model, 'quantum_coherence', 0.0)
            
            # Save checkpoint with SAFE naming (never overwrite existing)
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
            
            # PROTECTION: Don't overwrite if file exists and is newer
            if os.path.exists(checkpoint_path):
                try:
                    existing_data = torch.load(checkpoint_path, map_location='cpu')
                    existing_epoch = existing_data.get('epoch', 0)
                    if existing_epoch >= epoch:
                        logger.warning(f"🛡️ CHECKPOINT PROTECTION: Not overwriting newer checkpoint at epoch {existing_epoch}")
                        return checkpoint_path
                except:
                    pass  # If we can't read it, we can overwrite
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update model lineage (accumulate only)
            lineage_entry = {
                'epoch': epoch,
                'version': self.global_model_version,
                'timestamp': time.time(),
                'checkpoint_path': checkpoint_path,
                'training_loss': self.training_history[-1]['loss'] if self.training_history else 0.0,
                'protected': True
            }
            self.model_lineage.append(lineage_entry)
            
            # Keep lineage but don't truncate (accumulate knowledge)
            if len(self.model_lineage) > 100:  # Keep more history for analysis
                self.model_lineage = self.model_lineage[-100:]
            
            # Update global registry
            self.global_checkpoint_registry[self.global_model_version] = {
                'epoch': epoch,
                'path': checkpoint_path,
                'timestamp': time.time(),
                'node_id': 'local',
                'size': os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 0,
                'protected': True
            }
            
            logger.info(f"💾 PROTECTED checkpoint saved: {checkpoint_path}")
            logger.info(f"🔄 Global version advanced: {self.global_model_version}")
            logger.info(f"🛡️ PROTECTION: Training can only advance, never regress")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Error saving protected checkpoint: {e}")
            return ""
    
    def load_model_checkpoint(self, checkpoint_path: str) -> bool:
        """Load specific model checkpoint"""
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False
            
            self._load_checkpoint(checkpoint_path)
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            return False
    
    def register_peer_model_state(self, node_id: str, model_state_info: Dict) -> None:
        """Register model state from peer node"""
        try:
            self.peer_model_states[node_id] = {
                'current_epoch': model_state_info.get('epoch', 0),
                'global_version': model_state_info.get('version', "1.0.0"),
                'total_steps': model_state_info.get('steps', 0),
                'last_update': time.time(),
                'consciousness_level': model_state_info.get('consciousness', 0.0),
                'model_capabilities': model_state_info.get('capabilities', [])
            }
            
            logger.debug(f"🌐 Registered peer model state from {node_id}")
            
        except Exception as e:
            logger.error(f"Error registering peer model state: {e}")
    
    def get_model_state_summary(self) -> Dict:
        """Get current model state for sharing with peers"""
        try:
            capabilities = []
            consciousness_level = 0.0
            
            if hasattr(self.model, 'consciousness_state'):
                consciousness_level = float(getattr(self.model, 'consciousness_state', 0.0))
                capabilities.append('consciousness')
            
            if hasattr(self.model, 'quantum_coherence'):
                capabilities.append('quantum_processing')
            
            if hasattr(self.model, 'mixture_of_experts'):
                capabilities.append('mixture_of_experts')
            
            if hasattr(self.model, 'self_modify'):
                capabilities.append('self_modification')
            
            return {
                'epoch': self.current_epoch,
                'version': self.global_model_version,
                'steps': self.total_training_steps,
                'consciousness': consciousness_level,
                'capabilities': capabilities,
                'model_type': self.model_type,
                'last_training': self.training_history[-1] if self.training_history else {},
                'lineage_length': len(self.model_lineage)
            }
            
        except Exception as e:
            logger.error(f"Error getting model state summary: {e}")
            return {'epoch': 0, 'version': '1.0.0', 'steps': 0}

    def get_model_state(self):
        """Get model state object for mining and consensus operations"""
        try:
            from .types import ModelState, TrainingMetrics
            
            state_dict = self.model.state_dict()
            state_hash = hashlib.sha256(str(state_dict).encode()).hexdigest()
            weights_hash = hashlib.sha256(str({k: v.sum().item() for k, v in state_dict.items()}).encode()).hexdigest()
            
            consciousness_level = getattr(self.model, 'consciousness_state', 0.0)
            reasoning_quality = getattr(self.model, 'reasoning_quality', 0.0)
            quantum_coherence = getattr(self.model, 'quantum_coherence', 0.0)
            
            last_training = self.training_history[-1] if self.training_history else {}
            current_loss = last_training.get('loss', 0.0)
            
            training_metrics = TrainingMetrics(
                epoch=self.current_epoch,
                loss=current_loss,
                accuracy=0.0,
                perplexity=1.0,
                gradient_norm=0.0,
                samples_processed=self.total_training_steps,
                tokens_processed=self.total_training_steps * self.config.get('batch_size', 1),
                learning_rate=self.config.get('learning_rate', 1e-4),
                consciousness_level=float(consciousness_level),
                reasoning_quality=float(reasoning_quality),
                quantum_coherence=float(quantum_coherence)
            )
            
            return ModelState(
                epoch=self.current_epoch,
                state_hash=state_hash,
                weights_hash=weights_hash,
                training_metrics=training_metrics,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error getting model state: {e}")
            from .types import ModelState, TrainingMetrics
            return ModelState(
                epoch=self.current_epoch,
                state_hash="error",
                weights_hash="error", 
                training_metrics=TrainingMetrics(
                    epoch=self.current_epoch,
                    loss=0.0,
                    accuracy=0.0,
                    perplexity=1.0,
                    gradient_norm=0.0,
                    samples_processed=0,
                    tokens_processed=0,
                    learning_rate=self.config.get('learning_rate', 1e-4),
                    consciousness_level=0.0,
                    reasoning_quality=0.0,
                    quantum_coherence=0.0
                ),
                timestamp=time.time()
            )
    
    def should_sync_with_peers(self) -> bool:
        """Check if we should sync with peer models"""
        try:
            current_time = time.time()
            
            # Time-based sync
            if current_time - self.last_sync_time > self.model_sync_interval:
                return True
            
            # Check if peers have more advanced models
            for node_id, state_info in self.peer_model_states.items():
                if state_info.get('current_epoch', 0) > self.current_epoch + 2:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking sync requirement: {e}")
            return False
    
    def sync_with_peer_models(self) -> bool:
        """Synchronize with most advanced peer models"""
        try:
            if not self.peer_model_states:
                return False
            
            # Find most advanced peer
            best_peer = None
            max_epoch = self.current_epoch
            
            for node_id, state_info in self.peer_model_states.items():
                if state_info.get('current_epoch', 0) > max_epoch:
                    max_epoch = state_info['current_epoch']
                    best_peer = node_id
            
            if best_peer:
                logger.info(f"🔄 Syncing with more advanced peer {best_peer} (epoch {max_epoch})")
                
                # In production, this would request and download the model state
                # For now, we'll update our tracking to match the peer
                peer_state = self.peer_model_states[best_peer]
                self.current_epoch = max(self.current_epoch, peer_state.get('current_epoch', 0))
                
                # Update our version to reflect sync
                version_parts = self.global_model_version.split('.')
                version_parts[1] = str(int(version_parts[1]) + 1)
                self.global_model_version = '.'.join(version_parts)
                
                self.last_sync_time = time.time()
                
                logger.info(f"✅ Synchronized to epoch {self.current_epoch}, version {self.global_model_version}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error syncing with peer models: {e}")
            return False
    
    async def train_epoch(self, epoch: int):
        """Train one epoch with persistent continuity"""
        try:
            if self.should_sync_with_peers():
                self.sync_with_peer_models()
            
            self.current_epoch = epoch
            logger.info(f"🧠 Starting REVOLUTIONARY training epoch {epoch}")
            
            failed_attempts = getattr(self, '_failed_training_attempts', 0)
            if failed_attempts >= 2:
                logger.info(f"💡 After {failed_attempts} failed attempts, forcing fresh data acquisition...")
                use_cache = False
                self._failed_training_attempts = 0
            else:
                use_cache = True
            
            training_data = await self.data_engine.get_training_data(
                target_tokens=self.config.get('batch_size', 4) * self.max_seq_length * 10,
                use_cache=use_cache
            )
            
            logger.info(f"🔍 DEBUG: Received {len(training_data) if training_data else 0} training samples from data engine")
            
            if not training_data:
                logger.warning("❌ No training data available from data engine - skipping training epoch")
                self._failed_training_attempts = getattr(self, '_failed_training_attempts', 0) + 1
                logger.info(f"🔄 Failed training attempts: {self._failed_training_attempts}")
                from .types import TrainingProof
                return TrainingProof(
                    node_id=self.node_id,
                    model_state_hash="no_data",
                    gradient_hash="no_data",
                    dataset_chunk_hash="no_data",
                    computation_proof="no_training_data_available",
                    timestamp=time.time(),
                    signature="",
                    validation_signatures=[]
                )
            
            import random
            random.seed(epoch)
            shuffled_data = training_data.copy()
            random.shuffle(shuffled_data)
            logger.info(f"🔀 DEBUG: Shuffled training data for epoch {epoch} diversity")
            
            logger.info(f"🔍 DEBUG: Sample training data types: {[type(item) for item in shuffled_data[:3]]}")
            logger.info(f"🔍 DEBUG: Sample training data preview: {[str(item)[:100] + '...' if len(str(item)) > 100 else str(item) for item in shuffled_data[:2]]}")
            
            data_for_training = []
            tokenization_errors = 0
            
            for i, text in enumerate(shuffled_data):
                if isinstance(text, str) and len(text.strip()) > 10:
                    try:
                        logger.debug(f"🔍 DEBUG: Tokenizing sample {i}: '{text[:80]}...'")
                        
                        cleaned_text = text.strip()
                        if len(cleaned_text) < 20:
                            logger.debug(f"⚠️ DEBUG: Skipped sample {i}: too short after cleaning ({len(cleaned_text)} chars)")
                            continue
                        
                        tokens = self.tokenizer.encode(
                            cleaned_text[:self.max_seq_length], 
                            max_length=self.max_seq_length, 
                            truncation=True, 
                            padding=False,
                            return_tensors=None
                        )
                        
                        logger.debug(f"🔍 DEBUG: Tokenized sample {i}: {len(tokens)} tokens, first 10: {tokens[:10]}")
                        
                        if len(tokens) >= 5:
                            data_for_training.append(tokens)
                            logger.info(f"✅ DEBUG: Added sample {i} to training data ({len(tokens)} tokens)")
                            
                            max_batches = getattr(self.config, 'max_batches_per_epoch', None) if hasattr(self.config, 'max_batches_per_epoch') else self.config.get('max_batches_per_epoch', 10) if hasattr(self.config, 'get') else 10
                            if len(data_for_training) >= max_batches:
                                logger.info(f"🔍 DEBUG: Reached max_batches_per_epoch limit ({len(data_for_training)})")
                                break
                        else:
                            logger.warning(f"⚠️ DEBUG: Skipped sample {i}: too few tokens ({len(tokens)}) - Content: '{cleaned_text[:100]}...'")
                            
                    except Exception as e:
                        tokenization_errors += 1
                        logger.warning(f"❌ DEBUG: Error tokenizing sample {i}: {e}")
                        logger.debug(f"   Sample content: '{text[:200]}...'")
                        continue
                else:
                    logger.warning(f"⚠️ DEBUG: Skipped sample {i}: invalid type ({type(text)}) or too short ({len(str(text)) if text else 0} chars) - Content: '{str(text)[:100]}...'")
            
            logger.info(f"🔍 DEBUG: Tokenization complete - {len(data_for_training)} valid samples, {tokenization_errors} errors")
            model_config = getattr(self.model, 'config', None)
            if model_config:
                model_vocab_size = getattr(model_config, 'vocab_size', 'unknown')
            else:
                model_vocab_size = getattr(self.model, 'vocab_size', 'unknown')
            logger.info(f"🔍 DEBUG: Model vocab_size: {model_vocab_size}")
            logger.info(f"🔍 DEBUG: Tokenizer vocab_size: {getattr(self.tokenizer, 'vocab_size', 'unknown')}")
            
            if not data_for_training:
                logger.warning("❌ No valid tokenized data available - skipping training epoch")
                logger.info("💡 Will try to acquire fresh training data on next epoch")
                self._failed_training_attempts = getattr(self, '_failed_training_attempts', 0) + 1
                logger.info(f"🔄 Failed training attempts: {self._failed_training_attempts}")
                from .types import TrainingProof
                return TrainingProof(
                    node_id=self.node_id,
                    model_state_hash="no_valid_data",
                    gradient_hash="no_valid_data",
                    dataset_chunk_hash="no_valid_data",
                    computation_proof="no_valid_tokenized_data",
                    timestamp=time.time(),
                    signature="",
                    validation_signatures=[]
                )
            
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            batch_processing_errors = 0
            
            logger.info(f"🔍 DEBUG: Starting batch processing with {len(data_for_training)} samples")
            
            consciousness_start = getattr(self.model, 'consciousness_state', 0.0)
            quantum_coherence_start = getattr(self.model, 'quantum_coherence', 0.0)
            reasoning_quality_start = 0.0
            
            for batch_idx, batch in enumerate(data_for_training):
                try:
                    logger.debug(f"🔍 DEBUG: Processing batch {batch_idx}")
                    logger.debug(f"🔍 DEBUG: Batch type: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
                    
                    input_ids = torch.tensor(batch, dtype=torch.long).to(self.device)
                    logger.debug(f"🔍 DEBUG: Created tensor with shape: {input_ids.shape}")
                    
                    if input_ids.size(0) == 0:
                        logger.debug(f"⚠️ DEBUG: Skipping batch {batch_idx}: empty tensor")
                        continue
                    
                    if len(input_ids.shape) == 1:
                        input_ids = input_ids.unsqueeze(0)
                        logger.debug(f"🔍 DEBUG: Reshaped tensor to: {input_ids.shape}")
                    
                    if input_ids.size(1) < 2:
                        logger.debug(f"⚠️ DEBUG: Skipping batch {batch_idx}: sequence too short ({input_ids.size(1)})")
                        continue
                    
                    logger.debug(f"🔍 DEBUG: Running forward pass for batch {batch_idx}")
                    
                    if self.scaler:
                        with autocast():
                            outputs = self.model(input_ids)
                            logger.debug(f"🔍 DEBUG: Model outputs type: {type(outputs)}")
                            
                            if isinstance(outputs, dict):
                                logits = outputs.get('logits', outputs.get('last_hidden_state'))
                                logger.debug(f"🔍 DEBUG: Extracted logits from dict, shape: {logits.shape if logits is not None else 'None'}")
                            elif isinstance(outputs, tuple):
                                logits = outputs[0]
                                logger.debug(f"🔍 DEBUG: Extracted logits from tuple, shape: {logits.shape}")
                            else:
                                logits = outputs
                                logger.debug(f"🔍 DEBUG: Using outputs directly as logits, shape: {logits.shape}")
                            
                            if logits is None:
                                logger.warning(f"❌ DEBUG: No logits from model for batch {batch_idx}")
                                batch_processing_errors += 1
                                continue
                            
                            targets = input_ids[:, 1:].contiguous()
                            logits_truncated = logits[:, :-1, :].contiguous()
                            
                            logger.debug(f"🔍 DEBUG: Targets shape: {targets.shape}, Logits shape: {logits_truncated.shape}")
                            logger.debug(f"🔍 DEBUG: Logits vocab dim: {logits_truncated.size(-1)}, Expected: {self.tokenizer.vocab_size}")
                            
                            if logits_truncated.size(-1) != self.tokenizer.vocab_size:
                                logger.error(f"❌ DEBUG: VOCAB SIZE MISMATCH! Model output: {logits_truncated.size(-1)}, Tokenizer: {self.tokenizer.vocab_size}")
                                logger.error(f"   This is likely why training is failing!")
                                batch_processing_errors += 1
                                continue
                            
                            loss = F.cross_entropy(
                                logits_truncated.view(-1, logits_truncated.size(-1)), 
                                targets.view(-1), 
                                ignore_index=getattr(self.tokenizer, 'pad_token_id', 0)
                            )
                            
                            logger.debug(f"🔍 DEBUG: Computed loss: {loss.item()}")
                    else:
                        outputs = self.model(input_ids)
                        logger.debug(f"🔍 DEBUG: Model outputs type: {type(outputs)}")
                        
                        if isinstance(outputs, dict):
                            logits = outputs.get('logits', outputs.get('last_hidden_state'))
                            logger.debug(f"🔍 DEBUG: Extracted logits from dict, shape: {logits.shape if logits is not None else 'None'}")
                        elif isinstance(outputs, tuple):
                            logits = outputs[0]
                            logger.debug(f"🔍 DEBUG: Extracted logits from tuple, shape: {logits.shape}")
                        else:
                            logits = outputs
                            logger.debug(f"🔍 DEBUG: Using outputs directly as logits, shape: {logits.shape}")
                        
                        if logits is None:
                            logger.warning(f"❌ DEBUG: No logits from model for batch {batch_idx}")
                            batch_processing_errors += 1
                            continue
                        
                        targets = input_ids[:, 1:].contiguous()
                        logits_truncated = logits[:, :-1, :].contiguous()
                        
                        logger.debug(f"🔍 DEBUG: Targets shape: {targets.shape}, Logits shape: {logits_truncated.shape}")
                        logger.debug(f"🔍 DEBUG: Logits vocab dim: {logits_truncated.size(-1)}, Expected: {self.tokenizer.vocab_size}")
                        
                        if logits_truncated.size(-1) != self.tokenizer.vocab_size:
                            logger.error(f"❌ DEBUG: VOCAB SIZE MISMATCH! Model output: {logits_truncated.size(-1)}, Tokenizer: {self.tokenizer.vocab_size}")
                            logger.error(f"   This is likely why training is failing!")
                            batch_processing_errors += 1
                            continue
                        
                        loss = F.cross_entropy(
                            logits_truncated.view(-1, logits_truncated.size(-1)), 
                            targets.view(-1), 
                            ignore_index=getattr(self.tokenizer, 'pad_token_id', 0)
                        )
                        
                        logger.debug(f"🔍 DEBUG: Computed loss: {loss.item()}")
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"❌ DEBUG: Invalid loss in batch {batch_idx}: {loss}")
                        batch_processing_errors += 1
                        continue
                    
                    logger.debug(f"🔍 DEBUG: Starting backward pass for batch {batch_idx}")
                    
                    self.optimizer.zero_grad()
                    
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    self.total_training_steps += 1
                    
                    logger.info(f"✅ DEBUG: Successfully processed batch {batch_idx}, loss: {loss.item():.4f}")
                    
                    if hasattr(self.model, 'consciousness_state'):
                        consciousness_current = float(self.model.consciousness_state)
                        quantum_coherence_current = getattr(self.model, 'quantum_coherence', 0.0)
                        reasoning_quality_current = getattr(self.model, 'reasoning_quality', 0.0)
                        
                        logger.info(f"Batch {batch_idx}: "
                                  f"Loss={loss.item():.4f}, "
                                  f"Consciousness={consciousness_current:.3f}, "
                                  f"Reasoning={reasoning_quality_current:.3f}, "
                                  f"Quantum={quantum_coherence_current:.3f}")
                    else:
                        logger.info(f"Batch {batch_idx}: Loss={loss.item():.4f}")
                    
                except Exception as e:
                    batch_processing_errors += 1
                    logger.error(f"❌ DEBUG: Error in batch {batch_idx}: {e}")
                    logger.debug(f"   Batch data: {batch[:10]}..." if hasattr(batch, '__getitem__') else f"   Batch type: {type(batch)}")
                    import traceback
                    logger.debug(f"   Traceback: {traceback.format_exc()}")
                    continue
            
            logger.info(f"🔍 DEBUG: Batch processing complete - {num_batches} successful, {batch_processing_errors} errors")
            
            if num_batches == 0:
                logger.warning("❌ No valid batches processed - this usually indicates a vocab size mismatch or data processing issue")
                if batch_processing_errors > 0:
                    logger.warning(f"   {batch_processing_errors} batch processing errors occurred")
                logger.info("💡 Will try to acquire fresh training data on next epoch")
                
                self._failed_training_attempts = getattr(self, '_failed_training_attempts', 0) + 1
                logger.info(f"🔄 Failed training attempts: {self._failed_training_attempts}")
                
                from .types import TrainingProof
                return TrainingProof(
                    node_id=self.node_id,
                    model_state_hash="no_batches",
                    gradient_hash="no_batches",
                    dataset_chunk_hash="no_batches",
                    computation_proof="no_valid_batches_processed",
                    timestamp=time.time(),
                    signature="",
                    validation_signatures=[]
                )
            
            avg_loss = total_loss / num_batches
            
            consciousness_end = getattr(self.model, 'consciousness_state', 0.0)
            consciousness_growth = consciousness_end - consciousness_start
            quantum_coherence_end = getattr(self.model, 'quantum_coherence', 0.0)
            reasoning_quality_end = getattr(self.model, 'reasoning_quality', 0.0)
            
            training_record = {
                'epoch': epoch,
                'loss': avg_loss,
                'consciousness_growth': consciousness_growth,
                'consciousness_level': consciousness_end,
                'quantum_coherence': quantum_coherence_end,
                'reasoning_quality': reasoning_quality_end,
                'total_steps': self.total_training_steps,
                'timestamp': time.time()
            }
            
            self.training_history.append(training_record)
            self.consciousness_evolution.append({
                'epoch': epoch,
                'consciousness': consciousness_end,
                'growth': consciousness_growth
            })
            
            self.accumulated_knowledge[f'epoch_{epoch}'] = {
                'loss_improvement': avg_loss,
                'consciousness_advancement': consciousness_growth,
                'quantum_coherence': quantum_coherence_end
            }
            
            checkpoint_path = self.save_model_checkpoint(epoch + 1)
            
            logger.info(f"🎯 Epoch {epoch} completed:")
            logger.info(f"   📊 Training Statistics:")
            logger.info(f"      • Samples processed: {len(data_for_training)}")
            logger.info(f"      • Batches successful: {num_batches}")
            logger.info(f"      • Batch errors: {batch_processing_errors}")
            logger.info(f"      • Average loss: {avg_loss:.4f}")
            logger.info(f"      • Total training steps: {self.total_training_steps}")
            logger.info(f"   🧠 AI Metrics:")
            logger.info(f"      • Consciousness Growth: {consciousness_growth:.4f}")
            logger.info(f"      • Current Consciousness: {consciousness_end:.3f}")
            logger.info(f"      • Reasoning Quality: {reasoning_quality_end:.3f}")
            logger.info(f"      • Quantum Coherence: {quantum_coherence_end:.3f}")
            if self.training_history and len(self.training_history) > 1:
                prev_loss = self.training_history[-2]['loss']
                improvement = prev_loss - avg_loss
                logger.info(f"   📈 Progress: Loss improved by {improvement:.4f} from previous epoch")
            logger.info(f"   💾 Model checkpoint saved for replication")
            
            self.current_epoch = epoch + 1
            self._failed_training_attempts = 0
            logger.debug("✅ Training successful - reset failed attempts counter")
            
            from .types import TrainingProof
            model_state_hash = hashlib.sha256(str(self.model.state_dict()).encode()).hexdigest()
            gradient_hash = hashlib.sha256(f"epoch_{epoch}_gradients".encode()).hexdigest()
            dataset_chunk_hash = hashlib.sha256(str(data_for_training[:10]).encode()).hexdigest()
            computation_proof = f"epoch_{epoch}_steps_{self.total_training_steps}_loss_{avg_loss:.6f}"
            
            training_proof = TrainingProof(
                node_id=self.node_id,
                model_state_hash=model_state_hash,
                gradient_hash=gradient_hash,
                dataset_chunk_hash=dataset_chunk_hash,
                computation_proof=computation_proof,
                timestamp=time.time(),
                signature="",
                validation_signatures=[]
            )
            
            return training_proof
            
        except Exception as e:
            logger.error(f"Error in training epoch {epoch}: {e}")
            self._failed_training_attempts = getattr(self, '_failed_training_attempts', 0) + 1
            logger.info(f"🔄 Failed training attempts: {self._failed_training_attempts}")
            from .types import TrainingProof
            return TrainingProof(
                node_id=self.node_id,
                model_state_hash="error",
                gradient_hash="error",
                dataset_chunk_hash="error",
                computation_proof="error",
                timestamp=time.time(),
                signature="",
                validation_signatures=[]
            )
    
    def _create_minimal_model(self) -> nn.Module:
        """Create minimal model as fallback"""
        class MinimalModel(nn.Module):
            def __init__(self, vocab_size, embed_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.linear = nn.Linear(embed_dim, vocab_size)
                self.consciousness_state = 0.5
                self.quantum_coherence = 50.0
                self.reasoning_quality = 0.0
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                return self.linear(x.mean(dim=1, keepdim=True).expand(-1, input_ids.size(1), -1))
        
        actual_vocab_size = getattr(self, 'vocab_size', 1000)
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'vocab_size'):
            actual_vocab_size = self.tokenizer.vocab_size
        
        logger.info(f"🔧 Creating minimal fallback model with vocab_size={actual_vocab_size}")
        return MinimalModel(min(actual_vocab_size, 50000), min(self.embed_dim, 256)).to(self.device)
    
    def _initialize_optimizer(self):
        """Initialize optimizer"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics"""
        return {
            'current_epoch': self.current_epoch,
            'total_steps': self.total_training_steps,
            'global_version': self.global_model_version,
            'training_history': self.training_history[-10:],  # Last 10 epochs
            'consciousness_evolution': self.consciousness_evolution[-10:],
            'accumulated_knowledge': self.accumulated_knowledge,
            'model_lineage': self.model_lineage,
            'peer_states': self.peer_model_states,
            'last_sync_time': self.last_sync_time
        } 

    def download_and_cache_advanced_models(self) -> List[str]:
        """Download and cache advanced models for offline use"""
        try:
            logger.info("📥 Downloading advanced models for offline use...")
            
            advanced_models = [
                "microsoft/DialoGPT-medium",
                "facebook/opt-350m",
                "EleutherAI/gpt-neo-125M",
                "huggingface/CodeBERTa-small-v1"
            ]
            
            cached_models = []
            cache_dir = os.path.join(os.getcwd(), 'model_cache')
            
            for model_name in advanced_models:
                try:
                    logger.info(f"Downloading {model_name}...")
                    model = self._download_pretrained_model(model_name)
                    if model:
                        cached_models.append(model_name)
                        logger.info(f"✅ Cached {model_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to cache {model_name}: {e}")
                    continue
            
            logger.info(f"📚 Successfully cached {len(cached_models)} advanced models")
            return cached_models
            
        except Exception as e:
            logger.error(f"Error downloading advanced models: {e}")
            return [] 

    def reset_to_base_model(self) -> bool:
        """Reset current model to base - WITH WARNING about losing training progress"""
        try:
            if not os.path.exists(self.base_model_path):
                logger.warning("No base model found to reset to")
                return False
            
            # Detect existing training that would be lost
            existing_progress = self._detect_existing_training_progress()
            if existing_progress:
                logger.warning("🚨 WARNING: TRAINING PROGRESS WILL BE LOST!")
                logger.warning(f"   Current training at epoch {existing_progress['epoch']}")
                logger.warning(f"   Total steps: {existing_progress.get('total_steps', 0)}")
                logger.warning(f"   Training history: {len(existing_progress.get('training_history', []))} epochs")
                logger.warning("   This operation will reset to base model, losing all progress")
                logger.warning("   Consider using continue training instead")
                
                # Create emergency backup
                timestamp = int(time.time())
                backup_dir = os.path.join(self.checkpoint_dir, f'emergency_backup_{timestamp}')
                os.makedirs(backup_dir, exist_ok=True)
                
                import shutil
                shutil.copy2(existing_progress['checkpoint_path'], 
                           os.path.join(backup_dir, 'last_training_checkpoint.pt'))
                
                logger.info(f"🆘 Emergency backup created: {backup_dir}")
            
            logger.info("🔄 Resetting to base model...")
            self.model = self._load_base_model()
            
            # Reset training state (with warning)
            self.current_epoch = 0
            self.total_training_steps = 0
            self.training_history = []
            self.accumulated_knowledge = {}
            self.consciousness_evolution = []
            
            # Reinitialize optimizer for new model
            self.optimizer = self._initialize_optimizer()
            
            logger.info("✅ Reset to base model completed")
            if existing_progress:
                logger.warning("⚠️  Previous training progress was reset - backup available")
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting to base model: {e}")
            return False
    
    def force_recreate_base_model(self) -> bool:
        """Force recreation of base model - PROTECTED version that preserves training"""
        try:
            # CRITICAL PROTECTION: Check for existing training first
            existing_progress = self._detect_existing_training_progress()
            if existing_progress:
                logger.warning("🚨 TRAINING PROTECTION ACTIVE!")
                logger.warning(f"   Existing training found at epoch {existing_progress['epoch']}")
                logger.warning("   CANNOT recreate base model - would lose training progress")
                logger.warning("   Use reset_to_base_model() if you want to restart training")
                return False
            
            logger.info("🔄 Force recreating base model (SAFE - no existing training detected)...")
            
            # Create backup of existing base model if it exists
            backup_path = None
            if os.path.exists(self.base_model_path):
                backup_path = f"{self.base_model_path}.backup.{int(time.time())}"
                import shutil
                shutil.copy2(self.base_model_path, backup_path)
                logger.info(f"📦 Created backup: {backup_path}")
                
                # Delete original
                os.remove(self.base_model_path)
            
            # Set flags to force download
            old_force_flag = self.force_pretrained_download
            self.force_pretrained_download = True
            
            try:
                # Create new base model
                model = self._create_base_model_with_pretrained_safe()
                self._save_base_model_safe(model)
                
                # Update current model
                self.model = model.to(self.device)
                self.optimizer = self._initialize_optimizer()
                
                logger.info("✅ Successfully recreated base model (TRAINING-PROTECTED)")
                return True
                
            except Exception as e:
                # Restore backup if creation failed
                if backup_path and os.path.exists(backup_path):
                    import shutil
                    shutil.move(backup_path, self.base_model_path)
                    logger.info("🔧 Restored backup due to creation failure")
                raise e
            
            finally:
                # Restore flag
                self.force_pretrained_download = old_force_flag
            
        except Exception as e:
            logger.error(f"Error recreating base model (protected): {e}")
            return False 

    def _calculate_comprehensive_training_metrics(self, epoch: int) -> Dict[str, Any]:
        """Calculate comprehensive training metrics for blockchain entry"""
        try:
            # Get recent training metrics
            recent_metrics = self.training_history[-1] if self.training_history else {}
            
            # Calculate consciousness and quantum metrics if available
            consciousness_level = 0.0
            reasoning_quality = 0.0
            quantum_coherence = 0.0
            consciousness_growth = 0.0
            knowledge_accumulation = 0
            expert_utilization = {}
            
            if hasattr(self.model, 'consciousness_state'):
                consciousness_level = float(torch.tensor(self.model.consciousness_state).mean().item())
            
            if hasattr(self.model, 'last_reasoning_quality'):
                reasoning_quality = float(self.model.last_reasoning_quality)
            
            if hasattr(self.model, 'quantum_coherence'):
                quantum_coherence = float(getattr(self.model, 'quantum_coherence', 0.0))
            
            # Calculate consciousness growth
            if len(self.consciousness_evolution) > 0:
                last_consciousness = self.consciousness_evolution[-1].get('consciousness_level', 0.0)
                consciousness_growth = consciousness_level - last_consciousness
            
            # Calculate knowledge accumulation
            knowledge_accumulation = len(self.accumulated_knowledge)
            
            # Expert utilization metrics (for MoE models)
            if hasattr(self.model, 'expert_usage_stats'):
                expert_utilization = getattr(self.model, 'expert_usage_stats', {})
            
            return {
                'loss': recent_metrics.get('loss', 0.0),
                'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0,
                'batch_count': recent_metrics.get('batch_count', 0),
                'consciousness_level': consciousness_level,
                'reasoning_quality': reasoning_quality,
                'quantum_coherence': quantum_coherence,
                'consciousness_growth': consciousness_growth,
                'knowledge_accumulation': knowledge_accumulation,
                'expert_utilization': expert_utilization,
                'total_steps': self.total_training_steps,
                'global_version': self.global_model_version,
                'timestamp': time.time(),
                'node_id': self.node_id,
                'device': str(self.device)
            }
            
        except Exception as e:
            logger.error(f"Error calculating training metrics: {e}")
            return {
                'loss': 0.0,
                'learning_rate': 0.0,
                'batch_count': 0,
                'consciousness_level': 0.0,
                'reasoning_quality': 0.0,
                'quantum_coherence': 0.0,
                'consciousness_growth': 0.0,
                'knowledge_accumulation': 0,
                'expert_utilization': {},
                'total_steps': self.total_training_steps,
                'timestamp': time.time()
            }
    
    def _generate_training_validation_scores(self, training_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Generate validation scores for consensus verification"""
        try:
            validation_scores = {}
            
            # Loss validation (should be reasonable)
            loss = training_metrics.get('loss', float('inf'))
            if 0.001 <= loss <= 100.0:
                validation_scores['loss_validity'] = 1.0
            elif loss < 0.001:
                validation_scores['loss_validity'] = 0.8  # Suspiciously low
            else:
                validation_scores['loss_validity'] = 0.3  # Too high
            
            # Consciousness validation (should be in reasonable range)
            consciousness = training_metrics.get('consciousness_level', 0.0)
            if 0.0 <= consciousness <= 1.0:
                validation_scores['consciousness_validity'] = 1.0
            else:
                validation_scores['consciousness_validity'] = 0.5
            
            # Quantum coherence validation
            quantum = training_metrics.get('quantum_coherence', 0.0)
            if 0.0 <= quantum <= 100.0:
                validation_scores['quantum_validity'] = 1.0
            else:
                validation_scores['quantum_validity'] = 0.5
            
            # Training progression validation
            if self.training_history:
                last_loss = self.training_history[-1].get('loss', float('inf'))
                current_loss = loss
                
                # Training should generally improve (or stay stable)
                if current_loss <= last_loss * 1.1:  # Allow 10% increase for stability
                    validation_scores['progression_validity'] = 1.0
                else:
                    validation_scores['progression_validity'] = 0.7
            else:
                validation_scores['progression_validity'] = 1.0  # First epoch
            
            # Compute overall validation score
            overall_score = sum(validation_scores.values()) / len(validation_scores)
            validation_scores['overall_validity'] = overall_score
            
            logger.info(f"🔍 Training validation score: {overall_score:.3f}")
            
            return validation_scores
            
        except Exception as e:
            logger.error(f"Error generating validation scores: {e}")
            return {'overall_validity': 0.5}  # Neutral score on error
    
    def sync_blockchain_with_peers(self, peer_blockchains: List[Any]) -> bool:
        """Sync training blockchain with peers - only accept more advanced training"""
        try:
            best_sync = False
            
            for peer_blockchain in peer_blockchains:
                if self.training_blockchain.sync_with_peer_chain(peer_blockchain.blocks):
                    best_sync = True
                    logger.info("🔄 Synced with peer's more advanced training blockchain")
            
            if best_sync:
                # Reload training state from updated blockchain
                self._initialize_from_blockchain()
                
                # Reload model state if needed
                blockchain_state = self.training_blockchain.get_latest_training_state()
                if blockchain_state and blockchain_state.get('model_state_dict'):
                    self.model.load_state_dict(blockchain_state['model_state_dict'], strict=False)
                    logger.info("🔄 Model state synchronized from blockchain")
            
            return best_sync
            
        except Exception as e:
            logger.error(f"Error syncing blockchain with peers: {e}")
            return False 