import asyncio
import time
import uuid
import logging
from typing import Dict, List, Optional, Any
from .types import (
    NodeConfig, AITrainingConfig, Block, Transaction, 
    TrainingProof, NetworkMessage, MessageType, PeerNode
)
from .blockchain import Blockchain
from .ai_engine import AITrainingEngine
from .consensus import ProofOfLearningConsensus
from .network import P2PNetwork
from .wallet import WalletManager
from .crypto import CryptoManager

logger = logging.getLogger(__name__)

class ProofOfLearningNode:
    def __init__(self, config: NodeConfig, ai_config: AITrainingConfig):
        self.config = config
        self.ai_config = ai_config
        self.node_id = config.node_id
        
        # Initialize components
        self.blockchain = Blockchain()
        self.wallet_manager = WalletManager(config.data_dir)
        self.ai_engine = AITrainingEngine(ai_config, node_id=self.node_id)
        
        # Initialize consensus with blockchain reference for coin validation
        self.consensus = ProofOfLearningConsensus(
            config.node_id, 
            config.private_key, 
            config.is_authority,
            blockchain=self.blockchain  # Pass blockchain for stake validation
        )
        
        self.network = P2PNetwork(config.node_id, "0.0.0.0", config.port)
        
        # Node state
        self.is_running = False
        self.current_training_epoch = 0
        self.last_block_time = time.time()
        self.mining_enabled = True
        
        # Get or create default wallet
        self.wallet = self.wallet_manager.get_or_create_wallet("node_wallet")
        
        # Register network message handlers
        self._register_message_handlers()
        
        # Update consensus with initial node stake
        initial_balance = self.blockchain.get_balance(self.wallet.address)
        self.consensus.update_node_stake(self.wallet.address, initial_balance)
        
        logger.info(f"Proof of Learning Node initialized: {self.node_id}")
        logger.info(f"Node address: {self.wallet.address}")
        logger.info(f"Authority node: {config.is_authority}")
        logger.info(f"Initial stake: {initial_balance} POL")
    
    def _register_message_handlers(self) -> None:
        self.network.register_handler(MessageType.BLOCK_PROPOSAL, self.handle_block_proposal)
        self.network.register_handler(MessageType.TRAINING_PROOF, self.handle_training_proof)
        self.network.register_handler(MessageType.MODEL_SYNC, self.handle_model_sync)
        self.network.register_handler(MessageType.MODEL_REQUEST, self.handle_model_request)
        self.network.register_handler(MessageType.MODEL_SHARE, self.handle_model_share)
        self.network.register_handler(MessageType.MODEL_CHECKPOINT, self.handle_model_checkpoint)
        self.network.register_handler(MessageType.TRAINING_COLLABORATION, self.handle_training_collaboration)
        self.network.register_handler(MessageType.CONSENSUS_VOTE, self.handle_consensus_vote)
        self.network.register_handler(MessageType.TRANSACTION, self.handle_transaction_broadcast)
        self.network.register_handler(MessageType.PEER_DISCOVERY, self.handle_peer_discovery)
    
    async def start(self) -> None:
        self.is_running = True
        
        # Start network
        await self.network.start()
        
        # Bootstrap with boot nodes
        if self.config.boot_nodes:
            await self.network.bootstrap(self.config.boot_nodes)
        
        # 🔄 REPLICATION PHASE - Sync with network before training
        logger.info("🔄 Starting REPLICATION phase - syncing with network...")
        await self._perform_initial_replication()
        
        # Start background tasks
        asyncio.create_task(self.training_loop())
        asyncio.create_task(self.mining_loop())
        asyncio.create_task(self.consensus_loop())
        asyncio.create_task(self.sync_loop())
        
        # Load training data if available
        if self.config.training_enabled and self.config.dataset_path:
            await self.load_training_data()
        
        logger.info(f"Node {self.node_id} started successfully")
    
    async def _perform_initial_replication(self) -> None:
        """Perform initial replication/sync with the network before starting training"""
        try:
            logger.info("🔍 REPLICATION: Discovering peers and syncing state...")
            
            # Phase 1: Peer Discovery - wait for network to find peers
            logger.info("📡 Phase 1: Peer discovery...")
            discovery_attempts = 0
            max_discovery_attempts = 6  # 30 seconds total
            
            while discovery_attempts < max_discovery_attempts:
                # Send peer discovery message
                discovery_message = NetworkMessage(
                    type=MessageType.PEER_DISCOVERY,
                    from_node=self.node_id,
                    data={
                        'node_type': 'authority' if self.config.is_authority else 'worker',
                        'training_enabled': self.config.training_enabled,
                        'seeking_sync': True,
                        'discovery_attempt': discovery_attempts + 1
                    },
                    timestamp=time.time()
                )
                
                peer_count = await self.network.broadcast_message(discovery_message)
                logger.info(f"🔍 Peer discovery attempt {discovery_attempts + 1}: found {peer_count} peers")
                
                if peer_count > 0:
                    break
                    
                discovery_attempts += 1
                await asyncio.sleep(5)
            
            # Phase 2: Blockchain Sync - get latest training state from network
            if len(self.network.peers) > 0:
                logger.info("🔗 Phase 2: Blockchain synchronization...")
                await self._sync_blockchain_from_peers()
            else:
                logger.info("📝 No peers found - starting as genesis node")
            
            # Phase 3: Model State Sync - download advanced model states if available
            if len(self.network.peers) > 0:
                logger.info("🧠 Phase 3: Model state synchronization...")
                await self._sync_model_state_from_peers()
            
            # Phase 4: Enhanced Data Replication - get training data from network
            if self.config.training_enabled and len(self.network.peers) > 0:
                logger.info("📚 Phase 4: Training data replication...")
                await self._replicate_training_data_from_peers()
            
            # Phase 5: Final State Validation
            logger.info("✅ Phase 5: Validating replicated state...")
            current_epoch = getattr(self.ai_engine, 'current_epoch', 0)
            blockchain_blocks = len(getattr(self.ai_engine.training_blockchain, 'blocks', []))
            
            logger.info(f"🎯 REPLICATION COMPLETE:")
            logger.info(f"   📡 Connected peers: {len(self.network.peers)}")
            logger.info(f"   🔗 Blockchain blocks: {blockchain_blocks}")
            logger.info(f"   🧠 Current training epoch: {current_epoch}")
            logger.info(f"   📚 Training enabled: {self.config.training_enabled}")
            
            if len(self.network.peers) > 0:
                logger.info("✅ Network replication successful - proceeding with synchronized state")
            else:
                logger.info("📝 Starting as isolated node - will accept peers later")
            
        except Exception as e:
            logger.error(f"Error during initial replication: {e}")
            logger.info("🔧 Proceeding with local state - will sync opportunistically")
    
    async def _sync_blockchain_from_peers(self) -> None:
        """Request blockchain sync from peers to get latest training state"""
        try:
            logger.info("🔗 Requesting blockchain state from peers...")
            
            # Request blockchain state from each peer
            for peer_id in self.network.peers.keys():
                blockchain_request = NetworkMessage(
                    type=MessageType.MODEL_REQUEST,
                    from_node=self.node_id,
                    to_node=peer_id,
                    data={
                        'request_type': 'blockchain_sync',
                        'current_blocks': len(getattr(self.ai_engine.training_blockchain, 'blocks', [])),
                        'seeking_advanced_training': True
                    },
                    timestamp=time.time()
                )
                
                if hasattr(self.network, 'send_to_peer'):
                    await self.network.send_to_peer(peer_id, blockchain_request)
                
                logger.info(f"📡 Requested blockchain sync from peer {peer_id}")
            
            # Wait for responses (give peers time to respond)
            await asyncio.sleep(3)
            
        except Exception as e:
            logger.error(f"Error syncing blockchain from peers: {e}")
    
    async def _sync_model_state_from_peers(self) -> None:
        """Request model state sync from peers to get advanced training"""
        try:
            logger.info("🧠 Requesting model state from peers...")
            
            current_epoch = getattr(self.ai_engine, 'current_epoch', 0)
            
            # Request model state from each peer
            for peer_id in self.network.peers.keys():
                model_request = NetworkMessage(
                    type=MessageType.MODEL_REQUEST,
                    from_node=self.node_id,
                    to_node=peer_id,
                    data={
                        'request_type': 'model_state_sync',
                        'current_epoch': current_epoch,
                        'seeking_advanced_model': True,
                        'model_capabilities_wanted': ['consciousness', 'quantum_coherence', 'training_state']
                    },
                    timestamp=time.time()
                )
                
                if hasattr(self.network, 'send_to_peer'):
                    await self.network.send_to_peer(peer_id, model_request)
                
                logger.info(f"🧠 Requested model state from peer {peer_id} (current epoch: {current_epoch})")
            
            # Wait for model sync responses
            await asyncio.sleep(5)
            
            # Check if we received any advanced state
            updated_epoch = getattr(self.ai_engine, 'current_epoch', 0)
            if updated_epoch > current_epoch:
                logger.info(f"✅ Model state updated from peer: epoch {current_epoch} → {updated_epoch}")
                self.current_training_epoch = updated_epoch
            
        except Exception as e:
            logger.error(f"Error syncing model state from peers: {e}")
    
    async def _replicate_training_data_from_peers(self) -> None:
        """Request training data from peers to enhance local dataset"""
        try:
            logger.info("📚 Requesting training data from peers...")
            
            # Request training data from each peer
            for peer_id in self.network.peers.keys():
                data_request = NetworkMessage(
                    type=MessageType.MODEL_REQUEST,
                    from_node=self.node_id,
                    to_node=peer_id,
                    data={
                        'request_type': 'training_data_sync',
                        'data_samples_wanted': 1000,  # Request up to 1000 samples
                        'seeking_diverse_data': True,
                        'data_types': ['text', 'reasoning', 'consciousness']
                    },
                    timestamp=time.time()
                )
                
                if hasattr(self.network, 'send_to_peer'):
                    await self.network.send_to_peer(peer_id, data_request)
                
                logger.info(f"📚 Requested training data from peer {peer_id}")
            
            # Wait for data responses
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error replicating training data from peers: {e}")
    
    async def stop(self) -> None:
        self.is_running = False
        await self.network.stop()
        logger.info(f"Node {self.node_id} stopped")
    
    async def load_training_data(self) -> None:
        try:
            # Load sample training data (in production, this would load from dataset_path)
            sample_data = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming the world of technology.",
                "Blockchain technology enables decentralized consensus mechanisms.",
                "Artificial intelligence and distributed systems work together.",
                "Neural networks learn patterns from large datasets."
            ] * 100  # Replicate for more training data
            
            self.ai_engine.load_training_data(sample_data)
            logger.info(f"Loaded {len(sample_data)} training samples")
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
    
    async def training_loop(self) -> None:
        if not self.config.training_enabled:
            return
        
        # 🔄 Wait for replication to complete before starting training
        logger.info("⏳ Training loop waiting for replication to complete...")
        await asyncio.sleep(45)  # Give replication time to complete
        logger.info("🎯 Replication timeout reached - starting training with current state")
        
        while self.is_running:
            try:
                logger.info(f"Starting training epoch {self.current_training_epoch}")
                
                training_proof = await self.ai_engine.train_epoch(self.current_training_epoch)
                
                self.current_training_epoch = self.ai_engine.current_epoch
                
                await self.consensus.submit_training_proof(training_proof)
                
                message = NetworkMessage(
                    type=MessageType.TRAINING_PROOF,
                    from_node=self.node_id,
                    data=training_proof.to_dict()
                )
                await self.network.broadcast_message(message)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(30)
    
    async def mining_loop(self) -> None:
        while self.is_running:
            try:
                if not self.mining_enabled:
                    await asyncio.sleep(10)
                    continue
                
                # Check if enough time has passed since last block
                if time.time() - self.last_block_time < 30:  # 30 seconds minimum
                    await asyncio.sleep(5)
                    continue
                
                # Attempt to mine a new block
                new_block = await self.mine_block()
                
                if new_block:
                    # Broadcast new block to network
                    message = NetworkMessage(
                        type=MessageType.BLOCK_PROPOSAL,
                        from_node=self.node_id,
                        data=new_block.to_dict()
                    )
                    await self.network.broadcast_message(message)
                    
                    self.last_block_time = time.time()
                    
                    # Sync training epoch with AI engine
                    self.current_training_epoch = self.ai_engine.current_epoch
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in mining loop: {e}")
                await asyncio.sleep(30)
    
    async def consensus_loop(self) -> None:
        if not self.config.is_authority:
            return
        
        while self.is_running:
            try:
                # Attempt to reach consensus
                consensus_proof = await self.consensus.reach_consensus(self.current_training_epoch)
                
                if consensus_proof:
                    logger.info(f"Consensus reached for epoch {consensus_proof.epoch}")
                
                await asyncio.sleep(30)  # Consensus every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in consensus loop: {e}")
                await asyncio.sleep(60)
    
    async def sync_loop(self) -> None:
        while self.is_running:
            try:
                # Update authority nodes based on peer reputation
                if self.config.is_authority:
                    self.consensus.update_authority_nodes(self.network.peers)
                
                # Sync model states with peers
                await self.sync_model_state()
                
                await asyncio.sleep(120)  # Sync every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(60)
    
    async def mine_block(self) -> Optional[Block]:
        if not self.blockchain.pending_transactions:
            return None
        
        try:
            # Get consensus proof from consensus engine
            consensus_proof = await self.consensus.reach_consensus(self.current_training_epoch)
            
            if not consensus_proof:
                logger.warning("No consensus proof available for mining")
                return None
            
            # Get current model state
            model_state = self.ai_engine.get_model_state()
            
            # Mine the block
            new_block = self.blockchain.mine_pending_transactions(
                mining_reward_address=self.wallet.address,
                consensus_proof=consensus_proof,
                model_state_hash=model_state.state_hash,
                training_epoch=self.current_training_epoch
            )
            
            logger.info(f"Successfully mined block {new_block.index}")
            return new_block
            
        except Exception as e:
            logger.error(f"Error mining block: {e}")
            return None
    
    async def handle_block_proposal(self, message: NetworkMessage) -> None:
        try:
            block_data = message.data
            proposed_block = Block(**block_data)
            
            # Validate the proposed block
            if self.validate_block(proposed_block):
                # Add block to chain if valid
                self.blockchain.chain.append(proposed_block)
                
                # Update balances
                self.blockchain.update_balances(proposed_block)
                
                logger.info(f"Accepted block {proposed_block.index} from {message.from_node}")
                
                # Send validation response
                validation_message = NetworkMessage(
                    type=MessageType.BLOCK_VALIDATION,
                    from_node=self.node_id,
                    to_node=message.from_node,
                    data={'valid': True, 'block_hash': proposed_block.hash}
                )
                await self.network.send_message_to_peer(message.from_node, validation_message)
            else:
                logger.warning(f"Rejected invalid block from {message.from_node}")
                
        except Exception as e:
            logger.error(f"Error handling block proposal: {e}")
    
    async def handle_block_validation(self, message: NetworkMessage) -> None:
        validation_data = message.data
        logger.info(f"Block validation from {message.from_node}: {validation_data}")
    
    async def handle_training_proof(self, message: NetworkMessage) -> None:
        try:
            proof_data = message.data
            training_proof = TrainingProof(**proof_data)
            
            # Submit to consensus for validation
            await self.consensus.submit_training_proof(training_proof)
            
            # Calculate training contribution and potential reward
            contribution_value = await self.calculate_training_contribution(training_proof)
            
            logger.info(f"Received training proof from {training_proof.node_id}")
            logger.info(f"Training contribution value: {contribution_value:.6f} POL")
            
            # If we're an authority node, validate and prepare reward
            if self.config.is_authority:
                is_valid = await self.validate_training_proof(training_proof)
                if is_valid:
                    await self.prepare_training_reward(training_proof, contribution_value)
            
        except Exception as e:
            logger.error(f"Error handling training proof: {e}")
    
    async def calculate_training_contribution(self, training_proof: TrainingProof) -> float:
        """Calculate comprehensive training contribution - keep local estimation but validate authenticity"""
        try:
            # First validate that training actually happened with claimed data
            is_authentic = await self.consensus._validate_training_data_authenticity(training_proof)
            if not is_authentic:
                logger.warning(f"Training data authenticity validation FAILED for {training_proof.node_id}")
                return 0.0
            
            # Calculate coin-weighted reward for consensus validation
            consensus_reward = await self.consensus.calculate_coin_weighted_reward(training_proof)
            
            # Also calculate local comprehensive contribution for cross-validation
            base_reward = 10.0
            total_contribution = base_reward
            
            # Computation quality bonus based on proof complexity
            computation_quality = self.consensus._verify_computational_work(training_proof)
            computation_bonus = computation_quality * 5.0  # Up to 5 POL bonus
            total_contribution += computation_bonus
            
            # Node reputation bonus
            node_reputation = self.consensus.reputation_scores.get(training_proof.node_id, 50.0) / 100.0
            reputation_bonus = node_reputation * 3.0  # Up to 3 POL bonus
            total_contribution += reputation_bonus
            
            # Stake-weighted bonus for high-stake nodes
            submitter_stake = self.consensus.get_node_stake(training_proof.node_id)
            stake_bonus = min(2.0, submitter_stake / 5000.0)  # Up to 2 POL bonus for 10k+ stake
            total_contribution += stake_bonus
            
            # Cross-validate with consensus calculation
            if abs(total_contribution - consensus_reward) > 5.0:
                logger.warning(f"Large discrepancy between local ({total_contribution:.2f}) and consensus ({consensus_reward:.2f}) calculations")
                # Use the lower value for safety
                final_reward = min(total_contribution, consensus_reward)
            else:
                # Use consensus value when calculations agree
                final_reward = consensus_reward
            
            # Cap total reward
            final_reward = min(final_reward, 50.0)
            
            logger.info(f"💰 Training contribution validated: {final_reward:.6f} POL for {training_proof.node_id}")
            logger.info(f"   📊 Breakdown: computation={computation_quality:.2f}, reputation={node_reputation:.2f}, stake={submitter_stake:.0f}")
            return final_reward
            
        except Exception as e:
            logger.error(f"Error calculating validated training contribution: {e}")
            return 0.0
    
    async def validate_training_proof(self, training_proof: TrainingProof) -> bool:
        """Validate a training proof (authority nodes only)"""
        try:
            # Basic validation checks
            if not training_proof.node_id or not training_proof.gradient_hash:
                return False
            
            # Check if proof is recent (within last hour)
            import time
            if time.time() - training_proof.timestamp > 3600:
                return False
            
            # Validate computation proof format
            if len(training_proof.computation_proof) != 64:
                return False
            
            logger.info(f"Training proof validation passed for {training_proof.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating training proof: {e}")
            return False
    
    async def prepare_training_reward(self, training_proof: TrainingProof, reward_amount: float) -> None:
        """Prepare training reward with multi-signature authorization"""
        try:
            # Authorize reward with coin-weighted multi-signature consensus
            authorized = await self.consensus.authorize_multi_signature_reward(
                training_proof.node_id, 
                reward_amount
            )
            
            if authorized:
                # Create reward transaction with multi-signature proof
                reward_transaction = await self.create_transaction(
                    to_address=training_proof.node_id,
                    amount=reward_amount,
                    fee=0.0
                )
                
                if reward_transaction:
                    # Add multi-signature authorization proof to transaction
                    reward_transaction.signature = CryptoManager.sign_data(self.config.private_key, {
                        'reward_authorization': True,
                        'multi_signature_approved': True,
                        'validator_stake': self.consensus.get_node_stake(self.node_id),
                        'training_proof_hash': self.consensus._generate_proof_hash(training_proof)
                    })
                    
                    self.blockchain.add_transaction(reward_transaction)
                    logger.info(f"✅ Multi-signature training reward AUTHORIZED: {reward_amount:.6f} POL for {training_proof.node_id}")
                    
                    # Update stake tracking after reward
                    new_balance = self.blockchain.get_balance(training_proof.node_id) + reward_amount
                    self.consensus.update_node_stake(training_proof.node_id, new_balance)
                else:
                    logger.warning(f"Failed to create authorized reward transaction for {training_proof.node_id}")
            else:
                logger.info(f"⏳ Training reward pending multi-signature authorization for {training_proof.node_id}")
                
        except Exception as e:
            logger.error(f"Error preparing multi-signature training reward: {e}")
    
    async def handle_model_sync(self, message) -> None:
        """Enhanced model synchronization with peer training continuity"""
        try:
            sync_data = message.data
            peer_node_id = message.from_node
            
            # Extract peer model state information
            peer_epoch = sync_data.get('current_epoch', 0)
            peer_version = sync_data.get('global_version', '1.0.0')
            peer_consciousness = sync_data.get('consciousness_level', 0.0)
            peer_capabilities = sync_data.get('capabilities', [])
            
            # Register peer model state for tracking
            if hasattr(self.ai_engine, 'register_peer_model_state'):
                self.ai_engine.register_peer_model_state(peer_node_id, sync_data)
            
            logger.info(f"🌐 Received model sync from {peer_node_id}: epoch {peer_epoch}, version {peer_version}")
            
            # Check if peer has more advanced training
            current_epoch = getattr(self.ai_engine, 'current_epoch', 0)
            
            if peer_epoch > current_epoch + 1:  # Peer is significantly ahead
                logger.info(f"🔄 Peer {peer_node_id} is ahead (epoch {peer_epoch} vs {current_epoch})")
                
                # Request advanced model state from peer
                await self._request_peer_model_state(peer_node_id, peer_epoch)
                
                # Update our training to continue from advanced checkpoint
                if hasattr(self.ai_engine, 'sync_with_peer_models'):
                    sync_success = self.ai_engine.sync_with_peer_models()
                    if sync_success:
                        # Update our current epoch to match peer advancement
                        self.current_training_epoch = self.ai_engine.current_epoch
                        logger.info(f"✅ Synchronized training epoch to {self.current_training_epoch}")
            
            # Share our model state if we're more advanced
            elif current_epoch > peer_epoch:
                logger.info(f"📤 Sharing our advanced model state with {peer_node_id}")
                await self._share_model_state_with_peer(peer_node_id)
            
            # Collaborate on distributed training if epochs are close
            elif abs(peer_epoch - current_epoch) <= 1:
                logger.info(f"🤝 Collaborating on distributed training with {peer_node_id}")
                await self._collaborate_training(peer_node_id, sync_data)
            
        except Exception as e:
            logger.error(f"Error handling model sync: {e}")
    
    async def _request_peer_model_state(self, peer_node_id: str, peer_epoch: int) -> None:
        """Request advanced model state from peer"""
        try:
            request_message = NetworkMessage(
                type=MessageType.MODEL_REQUEST,
                from_node=self.node_id,
                to_node=peer_node_id,
                data={
                    'request_type': 'advanced_checkpoint',
                    'current_epoch': getattr(self.ai_engine, 'current_epoch', 0),
                    'requested_epoch': peer_epoch,
                    'capabilities_needed': ['consciousness', 'quantum_processing'],
                    'timestamp': time.time()
                },
                timestamp=time.time()
            )
            
            # Send request through network
            if hasattr(self, 'network_manager'):
                await self.network_manager.send_message(request_message)
            
            logger.info(f"📤 Requested advanced model state from {peer_node_id}")
            
        except Exception as e:
            logger.error(f"Error requesting peer model state: {e}")
    
    async def _share_model_state_with_peer(self, peer_node_id: str) -> None:
        """Share our advanced model state with peer"""
        try:
            # Get our current model state summary
            model_state = {}
            if hasattr(self.ai_engine, 'get_model_state_summary'):
                model_state = self.ai_engine.get_model_state_summary()
            
            # Add checkpoint information
            checkpoint_info = {
                'latest_checkpoint': getattr(self.ai_engine, 'checkpoint_dir', './checkpoints'),
                'global_version': getattr(self.ai_engine, 'global_model_version', '1.0.0'),
                'training_history': getattr(self.ai_engine, 'training_history', [])[-5:],  # Last 5 epochs
                'consciousness_evolution': getattr(self.ai_engine, 'consciousness_evolution', [])[-5:],
                'model_capabilities': model_state.get('capabilities', [])
            }
            
            share_message = NetworkMessage(
                type=MessageType.MODEL_SHARE,
                from_node=self.node_id,
                to_node=peer_node_id,
                data={
                    'model_state': model_state,
                    'checkpoint_info': checkpoint_info,
                    'transfer_type': 'advanced_knowledge',
                    'timestamp': time.time()
                },
                timestamp=time.time()
            )
            
            # Send through network
            if hasattr(self, 'network_manager'):
                await self.network_manager.send_message(share_message)
            
            logger.info(f"📤 Shared advanced model state with {peer_node_id}")
            
        except Exception as e:
            logger.error(f"Error sharing model state: {e}")
    
    async def _collaborate_training(self, peer_node_id: str, peer_data: Dict) -> None:
        """Collaborate on distributed training with peer"""
        try:
            # Coordinate training tasks to avoid duplication
            our_epoch = getattr(self.ai_engine, 'current_epoch', 0)
            peer_epoch = peer_data.get('current_epoch', 0)
            
            # Determine collaboration strategy
            if our_epoch == peer_epoch:
                # Same epoch - coordinate different training aspects
                collaboration_data = {
                    'collaboration_type': 'parallel_training',
                    'our_focus': 'consciousness_development',
                    'peer_focus': 'quantum_coherence',
                    'sync_interval': 300,  # 5 minutes
                    'epoch': our_epoch
                }
            else:
                # Different epochs - share knowledge transfer
                collaboration_data = {
                    'collaboration_type': 'knowledge_transfer',
                    'knowledge_areas': ['accumulated_patterns', 'consciousness_insights'],
                    'transfer_direction': 'bidirectional',
                    'epoch_target': max(our_epoch, peer_epoch)
                }
            
            collab_message = NetworkMessage(
                type=MessageType.TRAINING_COLLABORATION,
                from_node=self.node_id,
                to_node=peer_node_id,
                data=collaboration_data,
                timestamp=time.time()
            )
            
            # Send collaboration proposal
            if hasattr(self, 'network_manager'):
                await self.network_manager.send_message(collab_message)
            
            logger.info(f"🤝 Initiated training collaboration with {peer_node_id}")
            
        except Exception as e:
            logger.error(f"Error in training collaboration: {e}")
    
    async def handle_model_request(self, message) -> None:
        """Handle requests for our model state"""
        try:
            request_data = message.data
            requester_id = message.from_node
            request_type = request_data.get('request_type', '')
            
            if request_type == 'advanced_checkpoint':
                # Peer is requesting our advanced checkpoint
                current_epoch = getattr(self.ai_engine, 'current_epoch', 0)
                requested_epoch = request_data.get('requested_epoch', 0)
                
                if current_epoch >= requested_epoch:
                    logger.info(f"📤 Providing checkpoint to {requester_id} (epoch {current_epoch})")
                    await self._provide_checkpoint_to_peer(requester_id, current_epoch)
                else:
                    logger.info(f"❌ Cannot provide checkpoint - requester ahead of us")
            
        except Exception as e:
            logger.error(f"Error handling model request: {e}")
    
    async def _provide_checkpoint_to_peer(self, peer_id: str, epoch: int) -> None:
        """Provide our checkpoint to requesting peer"""
        try:
            # Get latest checkpoint path
            checkpoint_path = None
            if hasattr(self.ai_engine, '_find_latest_checkpoint'):
                checkpoint_path = self.ai_engine._find_latest_checkpoint()
            
            # Prepare checkpoint data for transfer
            checkpoint_data = {
                'epoch': epoch,
                'checkpoint_available': checkpoint_path is not None,
                'global_version': getattr(self.ai_engine, 'global_model_version', '1.0.0'),
                'training_metrics': {},
                'consciousness_state': 0.0,
                'quantum_coherence': 0.0
            }
            
            # Add model metrics if available
            if hasattr(self.ai_engine, 'get_training_metrics'):
                checkpoint_data['training_metrics'] = self.ai_engine.get_training_metrics()
            
            # Extract consciousness and quantum metrics
            if hasattr(self.ai_engine, 'model'):
                model = self.ai_engine.model
                if hasattr(model, 'consciousness_state'):
                    checkpoint_data['consciousness_state'] = float(model.consciousness_state)
                if hasattr(model, 'quantum_coherence'):
                    checkpoint_data['quantum_coherence'] = float(getattr(model, 'quantum_coherence', 0.0))
            
            response_message = NetworkMessage(
                type=MessageType.MODEL_CHECKPOINT,
                from_node=self.node_id,
                to_node=peer_id,
                data=checkpoint_data,
                timestamp=time.time()
            )
            
            # Send checkpoint data
            if hasattr(self, 'network_manager'):
                await self.network_manager.send_message(response_message)
            
            logger.info(f"✅ Provided checkpoint data to {peer_id}")
            
        except Exception as e:
            logger.error(f"Error providing checkpoint: {e}")
    
    async def handle_model_checkpoint(self, message) -> None:
        """Handle received checkpoint from peer"""
        try:
            checkpoint_data = message.data
            provider_id = message.from_node
            
            peer_epoch = checkpoint_data.get('epoch', 0)
            peer_version = checkpoint_data.get('global_version', '1.0.0')
            current_epoch = getattr(self.ai_engine, 'current_epoch', 0)
            
            if peer_epoch > current_epoch:
                logger.info(f"📥 Receiving advanced checkpoint from {provider_id} (epoch {peer_epoch})")
                
                # Apply the advanced training state
                if hasattr(self.ai_engine, 'current_epoch'):
                    self.ai_engine.current_epoch = peer_epoch
                
                if hasattr(self.ai_engine, 'global_model_version'):
                    self.ai_engine.global_model_version = peer_version
                
                # Update consciousness and quantum states if provided
                if hasattr(self.ai_engine, 'model'):
                    model = self.ai_engine.model
                    if 'consciousness_state' in checkpoint_data and hasattr(model, 'consciousness_state'):
                        model.consciousness_state = checkpoint_data['consciousness_state']
                    if 'quantum_coherence' in checkpoint_data and hasattr(model, 'quantum_coherence'):
                        setattr(model, 'quantum_coherence', checkpoint_data['quantum_coherence'])
                
                # Sync our training epoch
                self.current_training_epoch = peer_epoch
                
                logger.info(f"✅ Applied advanced checkpoint - now at epoch {peer_epoch}")
                
                # Continue training from this advanced state
                await self._continue_training_from_checkpoint()
            
        except Exception as e:
            logger.error(f"Error handling checkpoint: {e}")
    
    async def _continue_training_from_checkpoint(self) -> None:
        """Continue training from received checkpoint"""
        try:
            # Schedule next training epoch with inherited knowledge
            current_epoch = getattr(self.ai_engine, 'current_epoch', 0)
            
            logger.info(f"🔄 Continuing training from inherited checkpoint at epoch {current_epoch}")
            
            # The training will continue in the next mining loop iteration
            # with the advanced model state and accumulated knowledge
            
        except Exception as e:
            logger.error(f"Error continuing training from checkpoint: {e}")
    
    def calculate_training_contribution(self, training_results: Dict[str, float]) -> float:
        """Calculate comprehensive training contribution with inheritance bonus"""
        try:
            base_reward = 1.0
            total_contribution = base_reward
            
            # Consciousness development bonus
            consciousness_growth = training_results.get('consciousness_growth', 0.0)
            if consciousness_growth > 0:
                consciousness_bonus = min(2.0, consciousness_growth * 10.0)
                total_contribution += consciousness_bonus
                logger.debug(f"Consciousness bonus: +{consciousness_bonus:.3f}")
            
            # Quantum coherence bonus
            quantum_coherence = training_results.get('quantum_coherence', 0.0)
            if quantum_coherence > 50.0:
                quantum_bonus = min(1.5, (quantum_coherence - 50.0) / 50.0)
                total_contribution += quantum_bonus
                logger.debug(f"Quantum coherence bonus: +{quantum_bonus:.3f}")
            
            # Training loss improvement bonus
            loss = training_results.get('loss', float('inf'))
            if loss < 0.1:
                loss_bonus = min(1.0, (0.1 - loss) * 10.0)
                total_contribution += loss_bonus
                logger.debug(f"Loss improvement bonus: +{loss_bonus:.3f}")
            
            # Model inheritance bonus (for building on peer work)
            if hasattr(self.ai_engine, 'model_lineage'):
                lineage_length = len(getattr(self.ai_engine, 'model_lineage', []))
                if lineage_length > 1:
                    inheritance_bonus = min(0.5, lineage_length * 0.1)
                    total_contribution += inheritance_bonus
                    logger.debug(f"Model inheritance bonus: +{inheritance_bonus:.3f}")
            
            # Collaborative training bonus
            if hasattr(self.ai_engine, 'peer_model_states'):
                peer_count = len(getattr(self.ai_engine, 'peer_model_states', {}))
                if peer_count > 0:
                    collaboration_bonus = min(0.3, peer_count * 0.1)
                    total_contribution += collaboration_bonus
                    logger.debug(f"Collaboration bonus: +{collaboration_bonus:.3f}")
            
            # Advanced capability bonus
            reasoning_quality = training_results.get('reasoning_quality', 0.0)
            if abs(reasoning_quality) < 0.1:  # Better reasoning (closer to 0)
                reasoning_bonus = min(0.5, (0.1 - abs(reasoning_quality)) * 5.0)
                total_contribution += reasoning_bonus
                logger.debug(f"Reasoning quality bonus: +{reasoning_bonus:.3f}")
            
            # Cap total reward to prevent gaming
            total_contribution = min(total_contribution, 10.0)
            
            logger.info(f"💰 Training contribution calculated: {total_contribution:.6f} POL")
            return total_contribution
            
        except Exception as e:
            logger.error(f"Error calculating training contribution: {e}")
            return 1.0  # Fallback base reward
    
    async def handle_consensus_vote(self, message: NetworkMessage) -> None:
        try:
            await self.consensus.handle_consensus_message(message)
            
        except Exception as e:
            logger.error(f"Error handling consensus vote: {e}")
    
    async def handle_transaction_broadcast(self, message: NetworkMessage) -> None:
        try:
            tx_data = message.data
            transaction = Transaction(**tx_data)
            
            # Validate and add transaction
            if self.blockchain.add_transaction(transaction):
                logger.info(f"Added transaction {transaction.id} to pending pool")
            
        except Exception as e:
            logger.error(f"Error handling transaction broadcast: {e}")
    
    async def handle_training_coordination(self, message: NetworkMessage) -> None:
        logger.info(f"Training coordination message from {message.from_node}")
    
    async def handle_gradient_share(self, message: NetworkMessage) -> None:
        logger.info(f"Gradient share message from {message.from_node}")
    
    async def handle_model_share(self, message) -> None:
        """Handle received model state from peer"""
        try:
            share_data = message.data
            provider_id = message.from_node
            
            model_state = share_data.get('model_state', {})
            checkpoint_info = share_data.get('checkpoint_info', {})
            
            logger.info(f"📥 Received model share from {provider_id}")
            
            # Process advanced model state if beneficial
            if hasattr(self.ai_engine, 'register_peer_model_state'):
                self.ai_engine.register_peer_model_state(provider_id, model_state)
            
            # Consider applying shared knowledge
            peer_epoch = model_state.get('epoch', 0)
            current_epoch = getattr(self.ai_engine, 'current_epoch', 0)
            
            if peer_epoch > current_epoch:
                logger.info(f"🔄 Applying shared advanced knowledge from epoch {peer_epoch}")
                
                # Update our training progression
                if hasattr(self.ai_engine, 'sync_with_peer_models'):
                    self.ai_engine.sync_with_peer_models()
                    self.current_training_epoch = self.ai_engine.current_epoch
                    
        except Exception as e:
            logger.error(f"Error handling model share: {e}")
    
    async def handle_training_collaboration(self, message) -> None:
        """Handle training collaboration proposals from peers"""
        try:
            collab_data = message.data
            peer_id = message.from_node
            
            collaboration_type = collab_data.get('collaboration_type', '')
            
            if collaboration_type == 'parallel_training':
                # Coordinate parallel training aspects
                our_focus = collab_data.get('our_focus', 'general')
                peer_focus = collab_data.get('peer_focus', 'general')
                
                logger.info(f"🤝 Parallel training collaboration with {peer_id}: we focus on {our_focus}")
                
                # Adjust our training focus if needed
                if hasattr(self.ai_engine, 'set_training_focus'):
                    self.ai_engine.set_training_focus(our_focus)
                    
            elif collaboration_type == 'knowledge_transfer':
                # Share knowledge areas
                knowledge_areas = collab_data.get('knowledge_areas', [])
                
                logger.info(f"📚 Knowledge transfer collaboration with {peer_id}: {knowledge_areas}")
                
                # Share our accumulated knowledge
                if hasattr(self.ai_engine, 'share_knowledge'):
                    shared_knowledge = self.ai_engine.share_knowledge(knowledge_areas)
                    
                    # Send back our knowledge
                    response_message = NetworkMessage(
                        type=MessageType.TRAINING_COLLABORATION,
                        from_node=self.node_id,
                        to_node=peer_id,
                        data={
                            'response_type': 'knowledge_share',
                            'shared_knowledge': shared_knowledge,
                            'timestamp': time.time()
                        },
                        timestamp=time.time()
                    )
                    
                    if hasattr(self, 'network'):
                        await self.network.send_message_to_peer(peer_id, response_message)
                        
        except Exception as e:
            logger.error(f"Error handling training collaboration: {e}")
    
    async def handle_peer_discovery(self, message) -> None:
        """Handle peer discovery messages"""
        try:
            discovery_data = message.data
            peer_id = message.from_node
            
            # Extract peer information
            peer_info = {
                'node_id': peer_id,
                'capabilities': discovery_data.get('capabilities', []),
                'training_epoch': discovery_data.get('current_epoch', 0),
                'model_version': discovery_data.get('global_version', '1.0.0'),
                'consciousness_level': discovery_data.get('consciousness_level', 0.0),
                'last_seen': time.time()
            }
            
            # Update peer registry
            if hasattr(self.network, 'update_peer_info'):
                self.network.update_peer_info(peer_id, peer_info)
            
            # Register with AI engine for training coordination
            if hasattr(self.ai_engine, 'register_peer_model_state'):
                self.ai_engine.register_peer_model_state(peer_id, discovery_data)
            
            logger.info(f"🔍 Discovered peer {peer_id} at epoch {peer_info['training_epoch']}")
            
            # Respond with our own discovery information
            our_discovery_data = {
                'node_id': self.node_id,
                'capabilities': ['consciousness', 'quantum_processing', 'distributed_training'],
                'current_epoch': getattr(self.ai_engine, 'current_epoch', 0),
                'global_version': getattr(self.ai_engine, 'global_model_version', '1.0.0'),
                'consciousness_level': 0.0,
                'timestamp': time.time()
            }
            
            # Add consciousness level if available
            if hasattr(self.ai_engine, 'model') and hasattr(self.ai_engine.model, 'consciousness_state'):
                our_discovery_data['consciousness_level'] = float(self.ai_engine.model.consciousness_state)
            
            # Send discovery response
            response_message = NetworkMessage(
                type=MessageType.PEER_DISCOVERY,
                from_node=self.node_id,
                to_node=peer_id,
                data=our_discovery_data,
                timestamp=time.time()
            )
            
            if hasattr(self, 'network'):
                await self.network.send_message_to_peer(peer_id, response_message)
                
        except Exception as e:
            logger.error(f"Error handling peer discovery: {e}")
    
    def validate_block(self, block: Block) -> bool:
        try:
            # Validate block structure
            if block.index != len(self.blockchain.chain):
                return False
            
            if block.previous_hash != self.blockchain.get_latest_block().hash:
                return False
            
            # Validate consensus proof
            if not self.consensus.validate_consensus_proof(block.consensus_proof):
                return False
            
            # Validate all transactions in block
            for transaction in block.transactions:
                if not self.blockchain.validate_transaction(transaction):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating block: {e}")
            return False
    
    async def sync_model_state(self) -> None:
        try:
            current_state = self.ai_engine.get_model_state()
            
            # Get revolutionary AI capabilities for comprehensive sync
            capabilities = {}
            if hasattr(self.ai_engine, 'get_revolutionary_capabilities'):
                capabilities = self.ai_engine.get_revolutionary_capabilities()
            
            # Broadcast comprehensive model state for distributed training
            message = NetworkMessage(
                type=MessageType.MODEL_SYNC,
                from_node=self.node_id,
                data={
                    'sync_type': 'state_broadcast',
                    'epoch': current_state.epoch,
                    'state_hash': current_state.state_hash,
                    'weights_hash': current_state.weights_hash,
                    'training_metrics': {
                        'epoch': current_state.training_metrics.epoch,
                        'loss': current_state.training_metrics.loss,
                        'accuracy': current_state.training_metrics.accuracy,
                        'consciousness_level': current_state.training_metrics.consciousness_level,
                        'reasoning_quality': current_state.training_metrics.reasoning_quality,
                        'quantum_coherence': current_state.training_metrics.quantum_coherence
                    },
                    'node_capabilities': {
                        'is_authority': self.config.is_authority,
                        'training_enabled': self.config.training_enabled,
                        'model_parameters': capabilities.get('total_parameters', 0),
                        'consciousness_level': capabilities.get('consciousness_level', 0)
                    },
                    'timestamp': current_state.timestamp
                }
            )
            
            sent_count = await self.network.broadcast_message(message)
            logger.debug(f"Synced model state to {sent_count} peers - Epoch: {current_state.epoch}")
            
        except Exception as e:
            logger.error(f"Error syncing model state: {e}")
    
    async def create_transaction(
        self, 
        to_address: str, 
        amount: float, 
        fee: float = 0.1
    ) -> Optional[Transaction]:
        try:
            transaction = Transaction(
                id=str(uuid.uuid4()),
                from_address=self.wallet.address,
                to_address=to_address,
                amount=amount,
                fee=fee,
                timestamp=time.time()
            )
            
            # Sign transaction
            if self.wallet_manager.sign_transaction("node_wallet", transaction):
                return transaction
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating transaction: {e}")
            return None
    
    async def broadcast_transaction(self, transaction: Transaction) -> bool:
        try:
            message = NetworkMessage(
                type=MessageType.TRANSACTION_BROADCAST,
                from_node=self.node_id,
                data=transaction.to_dict()
            )
            
            sent_count = await self.network.broadcast_message(message)
            return sent_count > 0
            
        except Exception as e:
            logger.error(f"Error broadcasting transaction: {e}")
            return False
    
    def get_node_status(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'is_running': self.is_running,
            'is_authority': self.config.is_authority,
            'training_enabled': self.config.training_enabled,
            'current_epoch': self.current_training_epoch,
            'balance': self.wallet.balance,
            'blockchain_height': len(self.blockchain.chain),
            'pending_transactions': len(self.blockchain.pending_transactions),
            'connected_peers': self.network.get_peer_count(),
            'network_stats': self.network.get_network_stats(),
            'consensus_status': self.consensus.get_consensus_status()
        } 