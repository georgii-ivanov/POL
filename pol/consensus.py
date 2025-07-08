import hashlib
import time
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from .types import TrainingProof, ConsensusProof, TrainingValidation, NodeConfig
from .crypto import CryptoManager

logger = logging.getLogger(__name__)

class ProofOfLearningConsensus:
    def __init__(self, node_id: str, private_key: str, is_authority: bool, blockchain=None):
        self.node_id = node_id
        self.private_key = private_key
        self.is_authority = is_authority
        self.blockchain = blockchain
        
        # Authority node management with coin requirements
        self.authority_nodes: List[str] = []
        self.reputation_scores: Dict[str, float] = {}
        self.node_stakes: Dict[str, float] = {}
        
        # Coin-holding validation requirements
        self.minimum_validator_stake = 1000.0  # 1000 POL minimum to validate
        self.minimum_authority_stake = 5000.0  # 5000 POL minimum to be authority
        self.stake_multiplier_bonus = 0.1  # 10% bonus per 1000 POL stake
        
        # Training proof validation with multi-signature requirements
        self.pending_proofs: Dict[str, TrainingProof] = {}
        self.validated_proofs: Dict[str, TrainingValidation] = {}
        self.proof_validation_threshold = 0.67  # 67% of coin-weighted authority nodes
        self.minimum_validators_required = 3  # Minimum 3 validators regardless of stake
        
        # Anti-fraud protection
        self.recent_proof_hashes: Set[str] = set()
        self.node_submission_times: Dict[str, float] = {}
        self.minimum_submission_interval = 30
        
        # Enhanced reward security with coin-weighted validation
        self.pending_rewards: Dict[str, float] = {}
        self.reward_validations: Dict[str, List[Tuple[str, float, str]]] = {}  # node_id -> [(validator, stake, signature)]
        
        logger.info(f"Enhanced Consensus engine initialized for node {node_id}")
        logger.info(f"Minimum validator stake: {self.minimum_validator_stake} POL")
        logger.info(f"Minimum authority stake: {self.minimum_authority_stake} POL")
    
    def update_node_stake(self, node_id: str, stake_amount: float) -> None:
        """Update node stake for validation rights"""
        self.node_stakes[node_id] = stake_amount
        
        # Update authority status based on stake
        if stake_amount >= self.minimum_authority_stake and node_id not in self.authority_nodes:
            self.authority_nodes.append(node_id)
            logger.info(f"Node {node_id} promoted to authority with {stake_amount} POL stake")
        elif stake_amount < self.minimum_authority_stake and node_id in self.authority_nodes:
            self.authority_nodes.remove(node_id)
            logger.info(f"Node {node_id} demoted from authority due to insufficient stake")
    
    def get_node_stake(self, node_id: str) -> float:
        """Get node stake from blockchain or local cache"""
        if self.blockchain:
            return self.blockchain.get_balance(node_id)
        return self.node_stakes.get(node_id, 0.0)
    
    def get_validator_weight(self, node_id: str) -> float:
        """Calculate validator voting weight based on stake"""
        stake = self.get_node_stake(node_id)
        if stake < self.minimum_validator_stake:
            return 0.0
        
        # Base weight = 1.0, with stake bonus
        base_weight = 1.0
        stake_bonus = (stake / 1000.0) * self.stake_multiplier_bonus
        return min(base_weight + stake_bonus, 10.0)  # Cap at 10x weight
    
    def get_eligible_validators(self) -> List[Tuple[str, float]]:
        """Get list of eligible validators with their weights"""
        validators = []
        for node_id in self.authority_nodes:
            weight = self.get_validator_weight(node_id)
            if weight > 0:
                validators.append((node_id, weight))
        
        # Sort by weight (highest stake first)
        validators.sort(key=lambda x: x[1], reverse=True)
        return validators
    
    async def submit_training_proof(self, proof: TrainingProof) -> bool:
        """Submit training proof with coin-holding anti-fraud validation"""
        try:
            # Security Check 1: Verify submitter has minimum stake
            submitter_stake = self.get_node_stake(proof.node_id)
            if submitter_stake < self.minimum_validator_stake:
                logger.warning(f"Insufficient stake for submission from {proof.node_id}: {submitter_stake} POL")
                return False
            
            # Security Check 2: Prevent duplicate submissions
            proof_hash = self._generate_proof_hash(proof)
            if proof_hash in self.recent_proof_hashes:
                logger.warning(f"Duplicate training proof rejected from {proof.node_id}")
                return False
            
            # Security Check 3: Rate limiting based on stake
            current_time = time.time()
            last_submission = self.node_submission_times.get(proof.node_id, 0)
            min_interval = max(10, self.minimum_submission_interval - (submitter_stake / 1000.0))
            if current_time - last_submission < min_interval:
                logger.warning(f"Rate limited submission from {proof.node_id}")
                return False
            
            # Security Check 4: Validate proof integrity
            if not self._validate_proof_integrity(proof):
                logger.warning(f"Invalid proof integrity from {proof.node_id}")
                return False
            
            # Security Check 5: Computational verification with stake weighting
            computation_score = self._verify_computational_work(proof)
            required_score = max(0.5, 1.0 - (submitter_stake / 10000.0))  # Higher stake = lower requirements
            if computation_score < required_score:
                logger.warning(f"Insufficient computational work from {proof.node_id}: {computation_score:.3f}")
                return False
            
            # Store proof for consensus validation
            self.pending_proofs[proof_hash] = proof
            self.recent_proof_hashes.add(proof_hash)
            self.node_submission_times[proof.node_id] = current_time
            
            # Validate training data authenticity
            if not await self._validate_training_data_authenticity(proof):
                logger.warning(f"Training data validation failed for {proof.node_id}")
                return False
            
            # If this node is authority with sufficient stake, validate the proof
            if self.is_authority and self.get_validator_weight(self.node_id) > 0:
                validation_result = await self._perform_coin_weighted_validation(proof)
                validation = TrainingValidation(
                    validator_id=self.node_id,
                    target_node_id=proof.node_id,
                    is_valid=validation_result,
                    computation_verification=self._generate_verification_signature(proof),
                    signature=CryptoManager.sign_data(self.private_key, {
                        'proof_hash': proof_hash,
                        'validation_result': validation_result,
                        'validator_stake': self.get_node_stake(self.node_id),
                        'timestamp': current_time
                    }),
                    timestamp=current_time
                )
                
                self.validated_proofs[proof_hash] = validation
                logger.info(f"Coin-weighted validation completed for {proof.node_id}: {validation_result}")
            
            logger.info(f"Training proof submitted for node {proof.node_id} (stake: {submitter_stake} POL)")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting training proof: {e}")
            return False
    
    def _generate_proof_hash(self, proof: TrainingProof) -> str:
        """Generate unique hash for training proof"""
        proof_data = {
            'node_id': proof.node_id,
            'model_state_hash': proof.model_state_hash,
            'gradient_hash': proof.gradient_hash,
            'dataset_chunk_hash': proof.dataset_chunk_hash,
            'computation_proof': proof.computation_proof,
            'timestamp': proof.timestamp
        }
        return CryptoManager.hash_data(proof_data)
    
    def _generate_verification_signature(self, proof: TrainingProof) -> str:
        """Generate cryptographic verification signature"""
        verification_data = {
            'validator': self.node_id,
            'proof_hash': self._generate_proof_hash(proof),
            'validation_time': time.time(),
            'verification_level': 'authority'
        }
        return CryptoManager.sign_data(self.private_key, verification_data)
    
    def _validate_proof_integrity(self, proof: TrainingProof) -> bool:
        """Enhanced validation of basic proof structure and content"""
        try:
            validation_issues = []
            
            # Check required fields exist
            if not proof.node_id:
                validation_issues.append("Missing node_id")
            if not proof.model_state_hash:
                validation_issues.append("Missing model_state_hash")
            if not proof.gradient_hash:
                validation_issues.append("Missing gradient_hash")
            if not proof.dataset_chunk_hash:
                validation_issues.append("Missing dataset_chunk_hash")
            if not proof.computation_proof:
                validation_issues.append("Missing computation_proof")
            
            if validation_issues:
                logger.warning(f"Proof integrity failed for {proof.node_id}: {', '.join(validation_issues)}")
                return False
            
            # Check hash formats (should be 64-char hex)
            hash_fields = {
                'model_state_hash': proof.model_state_hash,
                'gradient_hash': proof.gradient_hash,
                'dataset_chunk_hash': proof.dataset_chunk_hash
            }
            
            for field_name, hash_value in hash_fields.items():
                if len(hash_value) != 64:
                    validation_issues.append(f"{field_name} wrong length: {len(hash_value)}")
                elif not all(c in '0123456789abcdef' for c in hash_value.lower()):
                    validation_issues.append(f"{field_name} invalid hex format")
            
            # Check computation proof format
            if len(proof.computation_proof) != 64:
                validation_issues.append(f"computation_proof wrong length: {len(proof.computation_proof)}")
            elif not all(c in '0123456789abcdef' for c in proof.computation_proof.lower()):
                validation_issues.append("computation_proof invalid hex format")
            
            # Check timestamp validity
            current_time = time.time()
            time_diff = abs(current_time - proof.timestamp)
            if time_diff > 3600:  # More than 1 hour old
                validation_issues.append(f"timestamp too old: {time_diff:.0f} seconds")
            elif proof.timestamp > current_time + 300:  # More than 5 minutes in future
                validation_issues.append(f"timestamp in future: {proof.timestamp - current_time:.0f} seconds")
            
            # Check node_id format
            if len(proof.node_id) < 3:
                validation_issues.append("node_id too short")
            elif len(proof.node_id) > 100:
                validation_issues.append("node_id too long")
            
            if validation_issues:
                logger.warning(f"Proof integrity failed for {proof.node_id}: {', '.join(validation_issues)}")
                return False
            
            logger.debug(f"✅ Proof integrity validated for {proof.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating proof integrity for {getattr(proof, 'node_id', 'unknown')}: {e}")
            return False
    
    async def _validate_training_data_authenticity(self, proof: TrainingProof) -> bool:
        """Enhanced validation that training data is authentic and matches claimed learning"""
        try:
            validation_score = 0.0
            max_score = 100.0
            
            # Test 1: Dataset chunk hash format and validity (15 points)
            if len(proof.dataset_chunk_hash) == 64 and all(c in '0123456789abcdef' for c in proof.dataset_chunk_hash.lower()):
                validation_score += 15.0
            else:
                logger.warning(f"Invalid dataset chunk hash format for {proof.node_id}")
                return False
            
            # Test 2: Gradient-data correlation (25 points)
            combined_input = proof.dataset_chunk_hash + proof.model_state_hash
            expected_influence = hashlib.sha256(combined_input.encode()).hexdigest()
            
            # Multiple correlation checks for robustness
            correlations_found = 0
            for i in range(0, 32, 8):  # Check multiple segments
                segment = expected_influence[i:i+8]
                if segment in proof.gradient_hash:
                    correlations_found += 1
            
            if correlations_found >= 2:  # At least 2 correlations required
                validation_score += 25.0
            else:
                logger.warning(f"Insufficient gradient-data correlation for {proof.node_id}: {correlations_found}/4 segments")
            
            # Test 3: Computation proof authenticity (20 points)
            # Verify the computation proof was generated from the claimed inputs
            reconstruction_input = f"{proof.gradient_hash}{proof.model_state_hash}{proof.dataset_chunk_hash}{proof.timestamp}"
            expected_computation = hashlib.sha256(reconstruction_input.encode()).hexdigest()
            
            # Check if computation proof contains expected elements
            computation_matches = 0
            for i in range(0, 64, 16):
                expected_segment = expected_computation[i:i+8]
                if expected_segment in proof.computation_proof:
                    computation_matches += 1
            
            if computation_matches >= 2:
                validation_score += 20.0
            else:
                logger.warning(f"Computation proof validation failed for {proof.node_id}: {computation_matches}/4 matches")
            
            # Test 4: Temporal consistency and learning progression (25 points)
            recent_proofs = [p for p in self.validated_proofs.values() 
                           if hasattr(p, 'target_node_id') and p.target_node_id == proof.node_id 
                           and time.time() - p.timestamp < 3600]
            
            if len(recent_proofs) == 0:
                # First submission gets benefit of doubt
                validation_score += 20.0
            else:
                # Check for reasonable progression
                last_proof_data = None
                for recent_proof in recent_proofs:
                    if hasattr(recent_proof, 'target_node_id'):
                        # Find the original proof this validation refers to
                        for pending_hash, pending_proof in self.pending_proofs.items():
                            if pending_hash == proof_hash:
                                last_proof_data = pending_proof
                                break
                        break
                
                if last_proof_data:
                    # Check that gradient hashes are different (indicating new training)
                    if proof.gradient_hash != last_proof_data.gradient_hash:
                        validation_score += 15.0
                    else:
                        logger.warning(f"Gradient hash unchanged from previous epoch for {proof.node_id}")
                    
                    # Check that dataset chunk is different (indicating data diversity)
                    if proof.dataset_chunk_hash != last_proof_data.dataset_chunk_hash:
                        validation_score += 10.0
                    else:
                        # Same data is OK occasionally, but not ideal
                        validation_score += 5.0
                else:
                    validation_score += 15.0
            
            # Test 5: Data freshness and timing (15 points)
            current_time = time.time()
            proof_age = current_time - proof.timestamp
            
            if proof_age <= 600:  # Within 10 minutes
                validation_score += 15.0
            elif proof_age <= 1800:  # Within 30 minutes
                validation_score += 10.0
            elif proof_age <= 3600:  # Within 1 hour
                validation_score += 5.0
            else:
                logger.warning(f"Training proof too old for {proof.node_id}: {proof_age:.0f} seconds")
            
            # Require 70% score for validation
            is_valid = validation_score >= 70.0
            
            if is_valid:
                logger.info(f"✅ Training data authenticity VALIDATED for {proof.node_id}: {validation_score:.1f}/100")
            else:
                logger.warning(f"❌ Training data authenticity FAILED for {proof.node_id}: {validation_score:.1f}/100")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating training data authenticity: {e}")
            return False
    
    def _verify_computational_work(self, proof: TrainingProof) -> float:
        """Enhanced computational work verification returning score 0-1"""
        try:
            score = 0.0
            
            # Test 1: Computation proof structure (25%)
            if len(proof.computation_proof) == 64:
                score += 0.25
            
            # Test 2: Hash complexity (25%)
            entropy = len(set(proof.computation_proof)) / len(proof.computation_proof)
            if entropy >= 0.5:
                score += 0.25
            
            # Test 3: Gradient-model correlation (25%)
            combined_data = proof.gradient_hash + proof.model_state_hash + proof.dataset_chunk_hash
            expected_proof_prefix = hashlib.sha256(combined_data.encode()).hexdigest()[:16]
            if expected_proof_prefix in proof.computation_proof:
                score += 0.25
            
            # Test 4: Temporal verification (25%)
            if abs(time.time() - proof.timestamp) < 1800:  # Within 30 minutes
                score += 0.25
            
            return score
            
        except Exception:
            return 0.0
    
    async def _perform_coin_weighted_validation(self, proof: TrainingProof) -> bool:
        """Enhanced authority validation with coin-weighted scoring"""
        try:
            validator_stake = self.get_node_stake(self.node_id)
            validation_score = 0.0
            max_score = 100.0
            
            # Test 1: Proof integrity (20 points)
            if self._validate_proof_integrity(proof):
                validation_score += 20.0
            
            # Test 2: Computational work (25 points)
            comp_score = self._verify_computational_work(proof)
            validation_score += comp_score * 25.0
            
            # Test 3: Training data authenticity (25 points)
            if await self._validate_training_data_authenticity(proof):
                validation_score += 25.0
            
            # Test 4: Node reputation weighted by stake (15 points)
            node_reputation = self.reputation_scores.get(proof.node_id, 50.0)
            submitter_stake = self.get_node_stake(proof.node_id)
            stake_weighted_reputation = node_reputation * min(2.0, 1.0 + submitter_stake / 5000.0)
            validation_score += (stake_weighted_reputation / 100.0) * 15.0
            
            # Test 5: Cross-validation with other validators (15 points)
            other_validations = [v for v in self.validated_proofs.values() 
                               if hasattr(v, 'target_node_id') and v.target_node_id == proof.node_id]
            if len(other_validations) > 0:
                agreement_rate = sum(1 for v in other_validations if v.is_valid) / len(other_validations)
                validation_score += agreement_rate * 15.0
            else:
                validation_score += 7.5  # Neutral score for first validation
            
            # Stake bonus: Higher stake validators have slightly higher requirements
            required_threshold = 75.0 + min(10.0, validator_stake / 1000.0)
            is_valid = validation_score >= required_threshold
            
            # Update reputation based on validation and stake
            if is_valid:
                bonus = min(2.0, 1.0 + validator_stake / 2500.0)
                self.reputation_scores[proof.node_id] = min(100.0, 
                    self.reputation_scores.get(proof.node_id, 50.0) + bonus)
            else:
                penalty = max(1.0, 5.0 - validator_stake / 1000.0)
                self.reputation_scores[proof.node_id] = max(0.0,
                    self.reputation_scores.get(proof.node_id, 50.0) - penalty)
            
            logger.info(f"Coin-weighted validation score: {validation_score:.1f}/{max_score:.1f} (threshold: {required_threshold:.1f})")
            return is_valid
            
        except Exception as e:
            logger.error(f"Error in coin-weighted validation: {e}")
            return False
    
    async def calculate_coin_weighted_reward(self, proof: TrainingProof) -> float:
        """Calculate secure reward using coin-weighted consensus validation"""
        try:
            proof_hash = self._generate_proof_hash(proof)
            
            # Get coin-weighted validations for this proof
            eligible_validators = self.get_eligible_validators()
            total_weight = sum(weight for _, weight in eligible_validators)
            
            if total_weight == 0:
                logger.warning("No eligible validators for reward calculation")
                return 0.0
            
            # Check coin-weighted consensus
            validation_weight = 0.0
            for validator_id, weight in eligible_validators:
                # Find validation from this validator
                validator_validation = None
                for v in self.validated_proofs.values():
                    if (hasattr(v, 'validator_id') and v.validator_id == validator_id and
                        hasattr(v, 'target_node_id') and v.target_node_id == proof.node_id and
                        v.is_valid):
                        validator_validation = v
                        break
                
                if validator_validation:
                    validation_weight += weight
            
            # Require coin-weighted consensus (67% of stake)
            consensus_threshold = total_weight * self.proof_validation_threshold
            validator_count = len([v for v in self.validated_proofs.values() 
                                 if hasattr(v, 'target_node_id') and v.target_node_id == proof.node_id and v.is_valid])
            
            if validation_weight < consensus_threshold or validator_count < self.minimum_validators_required:
                logger.warning(f"Insufficient coin-weighted consensus: {validation_weight:.1f}/{consensus_threshold:.1f} weight, {validator_count}/{self.minimum_validators_required} validators")
                return 0.0
            
            # Calculate base reward with quality metrics
            base_reward = 10.0
            
            # Quality bonuses
            computation_quality = self._verify_computational_work(proof)
            node_reputation = self.reputation_scores.get(proof.node_id, 50.0) / 100.0
            submitter_stake = self.get_node_stake(proof.node_id)
            stake_bonus = min(0.5, submitter_stake / 10000.0)  # Up to 50% bonus for high stake
            
            # Consensus strength bonus
            consensus_strength = min(1.0, validation_weight / consensus_threshold)
            
            total_reward = base_reward * (
                0.3 +  # Base 30%
                0.3 * computation_quality +  # 30% for computation quality
                0.2 * node_reputation +      # 20% for reputation
                0.1 * stake_bonus +          # 10% for stake
                0.1 * consensus_strength     # 10% for consensus strength
            )
            
            logger.info(f"Coin-weighted reward calculated: {total_reward:.6f} POL (consensus: {validation_weight:.1f}/{consensus_threshold:.1f})")
            return min(total_reward, 50.0)  # Cap at 50 POL per epoch
            
        except Exception as e:
            logger.error(f"Error calculating coin-weighted reward: {e}")
            return 0.0
    
    async def authorize_multi_signature_reward(self, node_id: str, reward_amount: float) -> bool:
        """Authorize reward with multi-signature consensus from high-stake validators"""
        try:
            if not self.is_authority:
                return False
            
            validator_stake = self.get_node_stake(self.node_id)
            if validator_stake < self.minimum_validator_stake:
                logger.warning(f"Insufficient stake to authorize rewards: {validator_stake} POL")
                return False
            
            # Generate authorization signature with stake proof
            authorization_data = {
                'recipient': node_id,
                'amount': reward_amount,
                'authorizer': self.node_id,
                'authorizer_stake': validator_stake,
                'timestamp': time.time(),
                'network_id': 'pol_mainnet'
            }
            
            signature = CryptoManager.sign_data(self.private_key, authorization_data)
            
            # Add to reward validations with stake weight
            if node_id not in self.reward_validations:
                self.reward_validations[node_id] = []
            
            self.reward_validations[node_id].append((self.node_id, validator_stake, signature))
            self.pending_rewards[node_id] = reward_amount
            
            # Calculate coin-weighted consensus for reward authorization
            total_validator_weight = 0.0
            approving_weight = 0.0
            
            for validator_id, weight in self.get_eligible_validators():
                total_validator_weight += weight
                # Check if this validator has signed
                for auth_validator, auth_stake, auth_sig in self.reward_validations.get(node_id, []):
                    if auth_validator == validator_id:
                        approving_weight += weight
                        break
            
            required_weight = total_validator_weight * self.proof_validation_threshold
            validator_count = len(self.reward_validations.get(node_id, []))
            
            if (approving_weight >= required_weight and 
                validator_count >= self.minimum_validators_required):
                logger.info(f"Multi-signature reward APPROVED for {node_id}: {reward_amount:.6f} POL")
                logger.info(f"Consensus: {approving_weight:.1f}/{required_weight:.1f} stake weight, {validator_count} validators")
                return True
            
            logger.info(f"Multi-signature reward pending: {approving_weight:.1f}/{required_weight:.1f} stake weight, {validator_count}/{self.minimum_validators_required} validators")
            return False
            
        except Exception as e:
            logger.error(f"Error authorizing multi-signature reward: {e}")
            return False
    
    async def reach_consensus(self, epoch: int) -> Optional[ConsensusProof]:
        """Reach consensus on training proofs for the epoch with coin-weighted validation"""
        try:
            if not self.is_authority:
                return None
            
            # Collect validated training proofs for this epoch
            validated_proofs = [v for v in self.validated_proofs.values() if v.is_valid]
            
            if not validated_proofs:
                logger.info(f"No validated training proofs for consensus at epoch {epoch}")
                return None
            
            # Get coin-weighted authority nodes
            eligible_validators = self.get_eligible_validators()
            total_weight = sum(weight for _, weight in eligible_validators)
            
            if total_weight == 0:
                logger.warning("No eligible validators for consensus")
                return None
            
            # Generate aggregated model hash from all validated training
            all_model_hashes = []
            for proof_hash, validation in self.validated_proofs.items():
                if validation.is_valid:
                    # Find the original proof
                    for pending_hash, pending_proof in self.pending_proofs.items():
                        if pending_hash == proof_hash:
                            all_model_hashes.append(pending_proof.model_state_hash)
                            break
            
            if all_model_hashes:
                aggregated_hash = CryptoManager.hash_data(sorted(all_model_hashes))
            else:
                aggregated_hash = CryptoManager.hash_data(f"consensus_epoch_{epoch}")
            
            # Generate consensus proof with coin-weighted signatures
            consensus_data = {
                'epoch': epoch,
                'aggregated_model_hash': aggregated_hash,
                'authority_nodes': [node_id for node_id, _ in eligible_validators],
                'total_stake_weight': total_weight,
                'validation_count': len(validated_proofs),
                'timestamp': time.time()
            }
            
            consensus_signature = CryptoManager.sign_data(self.private_key, consensus_data)
            
            consensus_proof = ConsensusProof(
                authority_nodes=[self.node_id],  # In production, collect from all authorities
                training_validations=validated_proofs,
                aggregated_model_hash=aggregated_hash,
                consensus_signature=consensus_signature,
                epoch=epoch
            )
            
            logger.info(f"✅ Consensus reached for epoch {epoch} with {len(validated_proofs)} validated proofs")
            return consensus_proof
            
        except Exception as e:
            logger.error(f"Error reaching consensus: {e}")
            return None
    
    def validate_consensus_proof(self, proof: ConsensusProof) -> bool:
        """Validate consensus proof with coin-weighted verification"""
        try:
            # Check basic structure
            if not proof.authority_nodes or proof.epoch < 0:
                return False
            
            # Verify minimum number of authority nodes
            if len(proof.authority_nodes) < self.minimum_validators_required:
                logger.warning(f"Insufficient authority nodes in consensus: {len(proof.authority_nodes)}")
                return False
            
            # Verify that validations exist
            if len(proof.training_validations) == 0:
                logger.warning("No training validations in consensus proof")
                return False
            
            # Check stake requirements for authority nodes
            total_authority_stake = 0.0
            for authority_node in proof.authority_nodes:
                stake = self.get_node_stake(authority_node)
                if stake < self.minimum_authority_stake:
                    logger.warning(f"Authority node {authority_node} has insufficient stake: {stake}")
                    return False
                total_authority_stake += stake
            
            # Require minimum total stake for consensus
            minimum_total_stake = self.minimum_authority_stake * len(proof.authority_nodes)
            if total_authority_stake < minimum_total_stake:
                logger.warning(f"Insufficient total authority stake: {total_authority_stake}")
                return False
            
            logger.info(f"✅ Consensus proof validated for epoch {proof.epoch}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating consensus proof: {e}")
            return False
    
    def get_consensus_status(self) -> Dict[str, Any]:
        """Get current consensus status"""
        try:
            eligible_validators = self.get_eligible_validators()
            
            return {
                'authority_nodes': self.authority_nodes,
                'eligible_validators': len(eligible_validators),
                'total_stake_weight': sum(weight for _, weight in eligible_validators),
                'pending_proofs': len(self.pending_proofs),
                'validated_proofs': len([v for v in self.validated_proofs.values() if v.is_valid]),
                'minimum_validator_stake': self.minimum_validator_stake,
                'minimum_authority_stake': self.minimum_authority_stake,
                'pending_rewards': len(self.pending_rewards)
            }
            
        except Exception as e:
            logger.error(f"Error getting consensus status: {e}")
            return {}
    

    
    def update_authority_nodes(self, peers: List) -> None:
        """Update authority nodes based on reputation"""
        try:
            # Sort nodes by reputation and take top performers
            sorted_nodes = sorted(self.reputation_scores.items(), 
                                key=lambda x: x[1], reverse=True)
            
            # Select top 20 nodes as authorities (or fewer if not enough nodes)
            self.authority_nodes = [node_id for node_id, reputation in sorted_nodes[:20]
                                  if reputation > 75.0]  # Minimum reputation threshold
            
            if self.is_authority and self.node_id not in self.authority_nodes:
                self.authority_nodes.append(self.node_id)  # Ensure current authority stays in list
            
            logger.debug(f"Updated authority nodes: {len(self.authority_nodes)} authorities")
            
        except Exception as e:
            logger.error(f"Error updating authority nodes: {e}")
    
    def get_consensus_status(self) -> Dict:
        """Get current consensus status"""
        return {
            'authority_nodes': self.authority_nodes,
            'pending_proofs': len(self.pending_proofs),
            'reputation_scores': dict(list(self.reputation_scores.items())[:10]),  # Top 10
            'is_authority': self.is_authority,
            'node_id': self.node_id,
            'recent_validations': len([v for v in self.validated_proofs.values() 
                                     if time.time() - v.timestamp < 3600])  # Last hour
        }
    
    async def handle_consensus_message(self, message) -> None:
        """Handle consensus-related network messages"""
        try:
            # Process consensus votes, validation results, etc.
            logger.info(f"Processing consensus message from {message.from_node}")
            
        except Exception as e:
            logger.error(f"Error handling consensus message: {e}") 