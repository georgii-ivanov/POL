import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from .node import ProofOfLearningNode
from .types import NodeConfig, AITrainingConfig, TrainingProof
from .training_validator import AdvancedTrainingValidator
from .data_acquisition import InternetDataAcquisitionEngine
from .reward_system import ProportionalRewardSystem
from .crypto import CryptoManager

logger = logging.getLogger(__name__)

class EnhancedProofOfLearningNode(ProofOfLearningNode):
    """Enhanced POL node with advanced security, internet-scale data, and proportional rewards"""
    
    def __init__(self, config: NodeConfig, ai_config: AITrainingConfig):
        super().__init__(config, ai_config)
        
        # Initialize enhanced components
        self.training_validator = AdvancedTrainingValidator(config.node_id, config.private_key)
        # Pass AI config to data engine for proper configuration
        data_config = {
            'enable_web_scraping': getattr(ai_config, 'enable_web_scraping', False),
            'huggingface_only': getattr(ai_config, 'huggingface_only', True),
            'use_huggingface_datasets': getattr(ai_config, 'use_huggingface_datasets', True)
        }
        self.data_engine = InternetDataAcquisitionEngine(config.node_id, f"{config.data_dir}/training_corpus", data_config)
        self.reward_system = ProportionalRewardSystem()
        
        # Enhanced training tracking
        self.validated_proofs: List[TrainingProof] = []
        self.consciousness_tracking: Dict[str, Any] = {}
        
        logger.info("🚀 Enhanced POL Node initialized with:")
        logger.info("   ✅ Advanced Training Validation")
        logger.info("   ✅ Internet-Scale Data Acquisition")
        logger.info("   ✅ Proportional Reward System")
        logger.info("   ✅ Revolutionary AI Capabilities")
    
    async def start(self) -> None:
        """Start enhanced node with all advanced features"""
        await super().start()
        
        # Start enhanced data acquisition
        if self.config.training_enabled:
            await self._initialize_enhanced_training_data()
        
        logger.info("🎯 Enhanced POL Node fully operational")
    
    async def _initialize_enhanced_training_data(self) -> None:
        """Initialize with internet-scale training data"""
        try:
            logger.info("🌐 Acquiring internet-scale training data...")
            
            # Target 10M tokens for comprehensive training
            training_data = await self.data_engine.get_training_data(
                target_tokens=10_000_000,
                use_cache=True
            )
            
            logger.info(f"📚 Loaded {len(training_data):,} training samples")
            self.ai_engine.load_training_data(training_data)
            
        except Exception as e:
            logger.error(f"Error loading enhanced training data: {e}")
            # Fallback to basic data
            await super().load_training_data()
    
    async def training_loop(self) -> None:
        """Enhanced training loop with advanced validation and rewards"""
        if not self.config.training_enabled:
            return
        
        while self.is_running:
            try:
                logger.info(f"🧠 Starting ENHANCED training epoch {self.current_training_epoch}")
                
                training_proof = await self.ai_engine.train_epoch(self.current_training_epoch)
                
                is_valid = await self._validate_training_proof_enhanced(training_proof)
                
                if is_valid:
                    await self.consensus.submit_training_proof(training_proof)
                    
                    await self._broadcast_enhanced_training_proof(training_proof)
                    
                    self.validated_proofs.append(training_proof)
                    
                    logger.info(f"✅ Enhanced training proof validated and submitted")
                else:
                    logger.warning("❌ Training proof failed enhanced validation")
                
                self.current_training_epoch += 1
                
                if self.current_training_epoch % 10 == 0:
                    await self._calculate_enhanced_rewards()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in enhanced training loop: {e}")
                await asyncio.sleep(30)
    
    async def _validate_training_proof_enhanced(self, proof: TrainingProof) -> bool:
        """Enhanced training proof validation with anti-gaming measures"""
        try:
            # Get peer model weights for comparison if available
            peer_weights = None
            if hasattr(self.ai_engine, 'serialize_model_weights'):
                peer_weights = self.ai_engine.serialize_model_weights()
            
            # Perform comprehensive validation
            validation_result = await self.training_validator.validate_training_proof(proof, peer_weights)
            
            if validation_result.is_valid:
                logger.info(f"🔐 Training proof validation PASSED for {proof.node_id}")
                logger.info(f"   Validation details: {validation_result.computation_verification}")
                return True
            else:
                logger.warning(f"🚨 Training proof validation FAILED for {proof.node_id}")
                logger.warning(f"   Failure details: {validation_result.computation_verification}")
                return False
                
        except Exception as e:
            logger.error(f"Error in enhanced validation: {e}")
            return False
    
    async def _broadcast_enhanced_training_proof(self, proof: TrainingProof) -> None:
        """Broadcast training proof with enhanced metadata"""
        from .types import NetworkMessage, MessageType
        
        # Add consciousness tracking data if available
        enhanced_data = proof.to_dict()
        
        if hasattr(self.ai_engine, 'get_revolutionary_capabilities'):
            capabilities = self.ai_engine.get_revolutionary_capabilities()
            enhanced_data['revolutionary_capabilities'] = capabilities
            
            # Update consciousness tracking
            self.consciousness_tracking[proof.node_id] = capabilities
        
        message = NetworkMessage(
            type=MessageType.TRAINING_PROOF,
            from_node=self.node_id,
            data=enhanced_data
        )
        
        await self.network.broadcast_message(message)
    
    async def _calculate_enhanced_rewards(self) -> None:
        """Calculate and distribute proportional rewards"""
        try:
            if not self.validated_proofs:
                logger.info("No validated proofs for reward calculation")
                return
            
            logger.info(f"💰 Calculating proportional rewards for {len(self.validated_proofs)} validated proofs")
            
            # Calculate contributions
            contributions = self.reward_system.calculate_training_contributions(
                self.validated_proofs,
                self.consciousness_tracking
            )
            
            # Calculate proportional rewards
            rewards = self.reward_system.calculate_proportional_rewards(contributions)
            
            # Create reward transactions
            reward_transactions = self.reward_system.create_reward_transactions(
                rewards,
                self.current_training_epoch
            )
            
            # Add reward transactions to blockchain
            for transaction in reward_transactions:
                if self.blockchain.add_transaction(transaction):
                    logger.info(f"💎 Reward transaction created: {transaction.amount:.2f} POL to {transaction.to_address}")
            
            # Log reward statistics
            total_rewards = sum(r.total_reward for r in rewards)
            logger.info(f"📊 Total rewards distributed: {total_rewards:.2f} POL")
            
            # Log top contributors
            top_rewards = sorted(rewards, key=lambda r: r.total_reward, reverse=True)[:5]
            for i, reward in enumerate(top_rewards, 1):
                logger.info(f"   #{i}: {reward.node_id} - {reward.total_reward:.2f} POL ({reward.contribution_percentage*100:.1f}%)")
            
            # Clear processed proofs
            self.validated_proofs.clear()
            
        except Exception as e:
            logger.error(f"Error calculating enhanced rewards: {e}")
    
    async def handle_training_proof(self, message) -> None:
        """Enhanced training proof handling with advanced validation"""
        try:
            proof_data = message.data
            training_proof = TrainingProof(**proof_data)
            
            # Enhanced validation for peer proofs
            if self.config.is_authority:
                is_valid = await self._validate_training_proof_enhanced(training_proof)
                
                if is_valid:
                    # Submit to consensus for validation
                    await self.consensus.submit_training_proof(training_proof)
                    logger.info(f"✅ Peer training proof validated: {training_proof.node_id}")
                else:
                    logger.warning(f"❌ Peer training proof rejected: {training_proof.node_id}")
            else:
                # Non-authority nodes still submit to consensus
                await self.consensus.submit_training_proof(training_proof)
            
            logger.info(f"📥 Received training proof from {training_proof.node_id}")
            
        except Exception as e:
            logger.error(f"Error handling enhanced training proof: {e}")
    
    def get_enhanced_node_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced node status"""
        base_status = self.get_node_status()
        
        # Add enhanced status information
        enhanced_status = {
            **base_status,
            'enhanced_features': {
                'advanced_validation': True,
                'internet_scale_data': True,
                'proportional_rewards': True,
                'revolutionary_ai': True,
                'consciousness_tracking': len(self.consciousness_tracking) > 0,
                'anti_gaming_protection': True
            },
            'training_validation': {
                'validated_proofs': len(self.validated_proofs),
                'validation_enabled': True,
                'security_level': 'Maximum'
            },
            'data_acquisition': {
                'internet_sources': len(self.data_engine.data_sources),
                'huggingface_datasets': len(self.data_engine.huggingface_datasets),
                'data_cache_available': len(self.data_engine.load_cached_training_data()) > 0
            },
            'reward_system': self.reward_system.get_reward_statistics(),
            'consciousness_tracking': {
                'tracked_nodes': len(self.consciousness_tracking),
                'tracking_enabled': True
            }
        }
        
        # Add revolutionary AI capabilities if available
        if hasattr(self.ai_engine, 'get_revolutionary_capabilities'):
            enhanced_status['revolutionary_ai_status'] = self.ai_engine.get_revolutionary_capabilities()
        
        return enhanced_status
    
    async def acquire_fresh_training_data(self, target_tokens: int = 10_000_000) -> None:
        """Manually trigger fresh training data acquisition"""
        try:
            logger.info(f"🌐 Acquiring fresh training data: {target_tokens:,} tokens")
            
            training_data = await self.data_engine.acquire_internet_scale_data(target_tokens)
            self.ai_engine.load_training_data(training_data)
            
            logger.info(f"✅ Fresh training data acquired: {len(training_data):,} samples")
            
        except Exception as e:
            logger.error(f"Error acquiring fresh training data: {e}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report"""
        return {
            'training_validation': {
                'anti_gaming_enabled': True,
                'gradient_authenticity_check': True,
                'computation_proof_validation': True,
                'algorithm_consistency_check': True,
                'timing_manipulation_detection': True,
                'pattern_spoofing_detection': True,
                'cross_node_collusion_detection': True
            },
            'consensus_security': {
                'authority_nodes': len(self.consensus.authority_nodes),
                'consensus_threshold': self.consensus.consensus_threshold,
                'reputation_system': True,
                'signature_verification': True
            },
            'network_security': {
                'encrypted_communications': True,
                'peer_verification': True,
                'ddos_protection': True,
                'rate_limiting': True
            },
            'blockchain_security': {
                'cryptographic_hashing': True,
                'ecdsa_signatures': True,
                'proof_of_learning_validation': True,
                'chain_integrity_checks': True
            }
        }
    
    def get_decentralization_metrics(self) -> Dict[str, Any]:
        """Get decentralization health metrics"""
        total_nodes = len(self.network.peers) + 1  # Include self
        authority_nodes = len(self.consensus.authority_nodes)
        
        return {
            'network_size': total_nodes,
            'authority_nodes': authority_nodes,
            'authority_percentage': (authority_nodes / max(total_nodes, 1)) * 100,
            'max_node_reward_percentage': self.reward_system.max_node_reward_percentage * 100,
            'reward_distribution_gini': self.reward_system.get_reward_statistics().get('network_decentralization_gini', 0),
            'consensus_threshold': self.consensus.consensus_threshold * 100,
            'network_health': 'Healthy' if total_nodes >= 3 and authority_nodes >= 3 else 'Needs more nodes'
        } 