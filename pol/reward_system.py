import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from .types import TrainingProof, Transaction, Block
from .crypto import CryptoManager

logger = logging.getLogger(__name__)

@dataclass
class ContributionMetrics:
    """Detailed metrics for measuring training contributions"""
    node_id: str
    training_quality: float      # 0-1 score based on model improvement
    computational_work: float    # Amount of actual computation performed
    consciousness_development: float  # Revolutionary AI consciousness growth
    reasoning_improvement: float # Reasoning capability enhancement
    quantum_coherence: float    # Quantum processing efficiency
    validation_accuracy: float  # Accuracy as validator
    network_stability: float    # Contribution to network health
    data_contribution: float    # Quality of training data provided
    timestamp: float

@dataclass
class RewardCalculation:
    """Detailed reward calculation breakdown"""
    node_id: str
    base_reward: float
    quality_bonus: float
    consciousness_bonus: float
    reasoning_bonus: float
    quantum_bonus: float
    validation_bonus: float
    stability_bonus: float
    data_bonus: float
    total_reward: float
    contribution_percentage: float

class ProportionalRewardSystem:
    """Advanced reward system that ensures fair proportional distribution based on actual contributions"""
    
    def __init__(self):
        # Base reward configuration
        self.base_training_reward = 50.0
        self.base_validation_reward = 10.0
        self.total_reward_pool = 1000.0  # Per epoch
        
        # Bonus multipliers for revolutionary features
        self.consciousness_multiplier = 2.0
        self.reasoning_multiplier = 1.8
        self.quantum_multiplier = 1.5
        self.validation_multiplier = 1.3
        self.stability_multiplier = 1.2
        self.data_quality_multiplier = 1.4
        
        # Historical contribution tracking
        self.contribution_history: Dict[str, List[ContributionMetrics]] = {}
        self.reputation_scores: Dict[str, float] = {}
        
        # Decentralization metrics
        self.network_distribution: Dict[str, float] = {}
        self.max_node_reward_percentage = 0.15  # No single node can get more than 15%
        
        logger.info("Proportional Reward System initialized")
        logger.info(f"Base training reward: {self.base_training_reward}")
        logger.info(f"Total reward pool per epoch: {self.total_reward_pool}")
    
    def calculate_training_contributions(
        self, 
        training_proofs: List[TrainingProof],
        consciousness_data: Dict[str, Any] = None
    ) -> List[ContributionMetrics]:
        """Calculate detailed contribution metrics for all training nodes"""
        
        contributions = []
        
        for proof in training_proofs:
            metrics = self._analyze_training_proof(proof, consciousness_data)
            contributions.append(metrics)
            
            # Update historical tracking
            if proof.node_id not in self.contribution_history:
                self.contribution_history[proof.node_id] = []
            
            self.contribution_history[proof.node_id].append(metrics)
            
            # Keep only recent history (last 100 contributions)
            if len(self.contribution_history[proof.node_id]) > 100:
                self.contribution_history[proof.node_id] = self.contribution_history[proof.node_id][-100:]
        
        return contributions
    
    def _analyze_training_proof(
        self, 
        proof: TrainingProof, 
        consciousness_data: Dict[str, Any] = None
    ) -> ContributionMetrics:
        """Analyze a training proof to extract contribution metrics"""
        
        # Base computational work (normalized by hash complexity)
        computational_work = self._calculate_computational_work(proof)
        
        # Training quality based on gradient patterns and model improvement
        training_quality = self._assess_training_quality(proof)
        
        # Revolutionary AI specific metrics
        consciousness_score = 0.0
        reasoning_score = 0.0
        quantum_score = 0.0
        
        if consciousness_data and proof.node_id in consciousness_data:
            node_consciousness = consciousness_data[proof.node_id]
            consciousness_score = node_consciousness.get('consciousness_level', 0.0)
            reasoning_score = node_consciousness.get('reasoning_quality', 0.0)
            quantum_score = node_consciousness.get('quantum_coherence', 0.0)
        
        # Extract metrics from computation proof if available
        if hasattr(proof, 'computation_proof') and proof.computation_proof:
            consciousness_score, reasoning_score, quantum_score = self._extract_revolutionary_metrics(proof.computation_proof)
        
        # Validation accuracy (if node acts as validator)
        validation_accuracy = self._calculate_validation_accuracy(proof.node_id)
        
        # Network stability contribution
        network_stability = self._assess_network_stability_contribution(proof.node_id)
        
        # Data contribution quality
        data_contribution = self._assess_data_contribution_quality(proof)
        
        return ContributionMetrics(
            node_id=proof.node_id,
            training_quality=training_quality,
            computational_work=computational_work,
            consciousness_development=consciousness_score,
            reasoning_improvement=reasoning_score,
            quantum_coherence=quantum_score,
            validation_accuracy=validation_accuracy,
            network_stability=network_stability,
            data_contribution=data_contribution,
            timestamp=proof.timestamp
        )
    
    def calculate_proportional_rewards(
        self, 
        contributions: List[ContributionMetrics]
    ) -> List[RewardCalculation]:
        """Calculate proportional rewards based on contributions"""
        
        if not contributions:
            return []
        
        # Calculate total contributions across all metrics
        total_metrics = self._calculate_total_contributions(contributions)
        
        rewards = []
        total_distributed = 0.0
        
        for contribution in contributions:
            reward_calc = self._calculate_individual_reward(contribution, total_metrics)
            rewards.append(reward_calc)
            total_distributed += reward_calc.total_reward
        
        # Ensure decentralization - no single node gets too much
        rewards = self._enforce_decentralization_limits(rewards, total_distributed)
        
        # Normalize to fit within reward pool
        rewards = self._normalize_rewards_to_pool(rewards)
        
        # Update reputation scores
        self._update_reputation_scores(rewards)
        
        logger.info(f"Calculated proportional rewards for {len(rewards)} nodes")
        logger.info(f"Total rewards distributed: {sum(r.total_reward for r in rewards):.2f}")
        
        return rewards
    
    def _calculate_computational_work(self, proof: TrainingProof) -> float:
        """Calculate computational work performed based on proof complexity"""
        
        # Analyze gradient hash complexity
        gradient_complexity = self._calculate_hash_complexity(proof.gradient_hash)
        
        # Analyze computation proof complexity
        computation_complexity = self._calculate_hash_complexity(proof.computation_proof)
        
        # Analyze model state changes
        model_complexity = self._calculate_hash_complexity(proof.model_state_hash)
        
        # Combine complexities with weights
        total_work = (
            gradient_complexity * 0.5 +
            computation_complexity * 0.3 +
            model_complexity * 0.2
        )
        
        # Normalize to 0-1 range
        return min(1.0, total_work / 10.0)
    
    def _calculate_hash_complexity(self, hash_string: str) -> float:
        """Calculate complexity score of a hash string"""
        if not hash_string:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = 0.0
        char_counts = {}
        
        for char in hash_string:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        length = len(hash_string)
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Look for computational patterns
        pattern_score = 0.0
        computational_patterns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        
        for pattern in computational_patterns:
            if pattern in hash_string:
                pattern_score += 0.1
        
        # Combine entropy and pattern scores
        complexity = (entropy * 0.7) + (min(pattern_score, 1.0) * 0.3)
        
        return complexity
    
    def _assess_training_quality(self, proof: TrainingProof) -> float:
        """Assess the quality of training based on proof characteristics"""
        
        quality_score = 0.5  # Base quality
        
        # Check for validation signatures (higher quality if validated by others)
        if hasattr(proof, 'validation_signatures') and proof.validation_signatures:
            validation_bonus = min(0.3, len(proof.validation_signatures) * 0.05)
            quality_score += validation_bonus
        
        # Check timestamp recency (more recent training gets slight bonus)
        current_time = time.time()
        time_diff = current_time - proof.timestamp
        if time_diff < 300:  # Within 5 minutes
            quality_score += 0.1
        elif time_diff < 1800:  # Within 30 minutes
            quality_score += 0.05
        
        # Check for dataset diversity (different dataset hash = more diverse training)
        if hasattr(proof, 'dataset_chunk_hash') and proof.dataset_chunk_hash:
            diversity_score = self._calculate_hash_complexity(proof.dataset_chunk_hash)
            quality_score += diversity_score * 0.1
        
        return min(1.0, quality_score)
    
    def _extract_revolutionary_metrics(self, computation_proof: str) -> Tuple[float, float, float]:
        """Extract consciousness, reasoning, and quantum metrics from computation proof"""
        
        consciousness_score = 0.0
        reasoning_score = 0.0
        quantum_score = 0.0
        
        try:
            # Look for revolutionary AI indicators in computation proof
            if 'consciousness' in computation_proof.lower():
                consciousness_score = 0.8
            
            if 'reasoning' in computation_proof.lower():
                reasoning_score = 0.7
            
            if 'quantum' in computation_proof.lower():
                quantum_score = 0.6
            
            # Extract numerical indicators if present
            import re
            numbers = re.findall(r'\d+\.?\d*', computation_proof)
            if numbers:
                # Use hash complexity of numerical patterns as proxy for sophisticated processing
                num_str = ''.join(numbers)
                complexity = self._calculate_hash_complexity(num_str)
                
                consciousness_score = max(consciousness_score, complexity * 0.8)
                reasoning_score = max(reasoning_score, complexity * 0.7)
                quantum_score = max(quantum_score, complexity * 0.6)
        
        except Exception as e:
            logger.warning(f"Error extracting revolutionary metrics: {e}")
        
        return consciousness_score, reasoning_score, quantum_score
    
    def _calculate_validation_accuracy(self, node_id: str) -> float:
        """Calculate validation accuracy for nodes that act as validators"""
        
        # This would be enhanced with actual validation history
        # For now, use reputation as proxy
        return self.reputation_scores.get(node_id, 0.5)
    
    def _assess_network_stability_contribution(self, node_id: str) -> float:
        """Assess node's contribution to network stability"""
        
        # Check historical uptime and participation
        if node_id in self.contribution_history:
            recent_contributions = self.contribution_history[node_id][-10:]  # Last 10 contributions
            if recent_contributions:
                # Consistent participation indicates stability
                consistency = len(recent_contributions) / 10.0
                return consistency
        
        return 0.5  # Default stability score
    
    def _assess_data_contribution_quality(self, proof: TrainingProof) -> float:
        """Assess quality of training data contributed"""
        
        if hasattr(proof, 'dataset_chunk_hash') and proof.dataset_chunk_hash:
            # Use dataset hash complexity as proxy for data quality
            return self._calculate_hash_complexity(proof.dataset_chunk_hash) * 0.5
        
        return 0.3  # Default data contribution score
    
    def _calculate_total_contributions(self, contributions: List[ContributionMetrics]) -> Dict[str, float]:
        """Calculate total contributions across all metrics"""
        
        totals = {
            'training_quality': sum(c.training_quality for c in contributions),
            'computational_work': sum(c.computational_work for c in contributions),
            'consciousness_development': sum(c.consciousness_development for c in contributions),
            'reasoning_improvement': sum(c.reasoning_improvement for c in contributions),
            'quantum_coherence': sum(c.quantum_coherence for c in contributions),
            'validation_accuracy': sum(c.validation_accuracy for c in contributions),
            'network_stability': sum(c.network_stability for c in contributions),
            'data_contribution': sum(c.data_contribution for c in contributions)
        }
        
        # Avoid division by zero
        for key in totals:
            if totals[key] == 0:
                totals[key] = 1.0
        
        return totals
    
    def _calculate_individual_reward(
        self, 
        contribution: ContributionMetrics,
        total_metrics: Dict[str, float]
    ) -> RewardCalculation:
        """Calculate individual node reward based on proportional contribution"""
        
        # Base reward proportional to training quality
        base_reward = self.base_training_reward * (contribution.training_quality / total_metrics['training_quality'])
        
        # Quality bonus based on computational work
        quality_bonus = 20.0 * (contribution.computational_work / total_metrics['computational_work'])
        
        # Revolutionary AI bonuses
        consciousness_bonus = 30.0 * self.consciousness_multiplier * (
            contribution.consciousness_development / total_metrics['consciousness_development']
        )
        
        reasoning_bonus = 25.0 * self.reasoning_multiplier * (
            contribution.reasoning_improvement / total_metrics['reasoning_improvement']
        )
        
        quantum_bonus = 20.0 * self.quantum_multiplier * (
            contribution.quantum_coherence / total_metrics['quantum_coherence']
        )
        
        # Network contribution bonuses
        validation_bonus = 15.0 * self.validation_multiplier * (
            contribution.validation_accuracy / total_metrics['validation_accuracy']
        )
        
        stability_bonus = 10.0 * self.stability_multiplier * (
            contribution.network_stability / total_metrics['network_stability']
        )
        
        data_bonus = 15.0 * self.data_quality_multiplier * (
            contribution.data_contribution / total_metrics['data_contribution']
        )
        
        # Calculate total reward
        total_reward = (
            base_reward + quality_bonus + consciousness_bonus + 
            reasoning_bonus + quantum_bonus + validation_bonus + 
            stability_bonus + data_bonus
        )
        
        return RewardCalculation(
            node_id=contribution.node_id,
            base_reward=base_reward,
            quality_bonus=quality_bonus,
            consciousness_bonus=consciousness_bonus,
            reasoning_bonus=reasoning_bonus,
            quantum_bonus=quantum_bonus,
            validation_bonus=validation_bonus,
            stability_bonus=stability_bonus,
            data_bonus=data_bonus,
            total_reward=total_reward,
            contribution_percentage=0.0  # Will be calculated after normalization
        )
    
    def _enforce_decentralization_limits(
        self, 
        rewards: List[RewardCalculation],
        total_distributed: float
    ) -> List[RewardCalculation]:
        """Ensure no single node gets too large a percentage of rewards"""
        
        # Calculate contribution percentages
        for reward in rewards:
            reward.contribution_percentage = reward.total_reward / total_distributed if total_distributed > 0 else 0
        
        # Cap any node that exceeds maximum percentage
        max_absolute_reward = total_distributed * self.max_node_reward_percentage
        excess_rewards = 0.0
        capped_nodes = 0
        
        for reward in rewards:
            if reward.total_reward > max_absolute_reward:
                excess = reward.total_reward - max_absolute_reward
                excess_rewards += excess
                reward.total_reward = max_absolute_reward
                reward.contribution_percentage = self.max_node_reward_percentage
                capped_nodes += 1
        
        # Redistribute excess rewards among remaining nodes
        if excess_rewards > 0 and len(rewards) > capped_nodes:
            remaining_nodes = [r for r in rewards if r.total_reward < max_absolute_reward]
            additional_per_node = excess_rewards / len(remaining_nodes)
            
            for reward in remaining_nodes:
                new_total = reward.total_reward + additional_per_node
                if new_total <= max_absolute_reward:
                    reward.total_reward = new_total
                else:
                    # If this would exceed cap, just give what we can
                    reward.total_reward = max_absolute_reward
        
        logger.info(f"Decentralization enforced: {capped_nodes} nodes capped at {self.max_node_reward_percentage*100}%")
        
        return rewards
    
    def _normalize_rewards_to_pool(self, rewards: List[RewardCalculation]) -> List[RewardCalculation]:
        """Normalize total rewards to fit within the reward pool"""
        
        current_total = sum(r.total_reward for r in rewards)
        
        if current_total > self.total_reward_pool:
            normalization_factor = self.total_reward_pool / current_total
            
            for reward in rewards:
                reward.base_reward *= normalization_factor
                reward.quality_bonus *= normalization_factor
                reward.consciousness_bonus *= normalization_factor
                reward.reasoning_bonus *= normalization_factor
                reward.quantum_bonus *= normalization_factor
                reward.validation_bonus *= normalization_factor
                reward.stability_bonus *= normalization_factor
                reward.data_bonus *= normalization_factor
                reward.total_reward *= normalization_factor
            
            logger.info(f"Rewards normalized by factor {normalization_factor:.3f} to fit pool of {self.total_reward_pool}")
        
        return rewards
    
    def _update_reputation_scores(self, rewards: List[RewardCalculation]) -> None:
        """Update long-term reputation scores based on rewards"""
        
        for reward in rewards:
            current_reputation = self.reputation_scores.get(reward.node_id, 0.5)
            
            # Calculate reputation change based on relative performance
            if reward.contribution_percentage > 0.1:  # High contributor
                reputation_change = 0.05
            elif reward.contribution_percentage > 0.05:  # Medium contributor
                reputation_change = 0.02
            elif reward.contribution_percentage > 0.01:  # Low contributor
                reputation_change = 0.01
            else:  # Very low contributor
                reputation_change = -0.01
            
            # Update reputation with momentum
            new_reputation = current_reputation * 0.9 + (current_reputation + reputation_change) * 0.1
            self.reputation_scores[reward.node_id] = max(0.0, min(1.0, new_reputation))
    
    def create_reward_transactions(
        self, 
        rewards: List[RewardCalculation],
        epoch: int
    ) -> List[Transaction]:
        """Create transactions to distribute calculated rewards"""
        
        transactions = []
        
        for reward in rewards:
            if reward.total_reward > 0.01:  # Minimum reward threshold
                transaction = Transaction(
                    id=f"reward_{reward.node_id}_{epoch}_{int(time.time())}",
                    from_address="system",
                    to_address=reward.node_id,
                    amount=reward.total_reward,
                    fee=0.0,
                    timestamp=time.time(),
                    reward_details={
                        'epoch': epoch,
                        'base_reward': reward.base_reward,
                        'quality_bonus': reward.quality_bonus,
                        'consciousness_bonus': reward.consciousness_bonus,
                        'reasoning_bonus': reward.reasoning_bonus,
                        'quantum_bonus': reward.quantum_bonus,
                        'validation_bonus': reward.validation_bonus,
                        'stability_bonus': reward.stability_bonus,
                        'data_bonus': reward.data_bonus,
                        'contribution_percentage': reward.contribution_percentage
                    }
                )
                
                transactions.append(transaction)
        
        logger.info(f"Created {len(transactions)} reward transactions for epoch {epoch}")
        
        return transactions
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward system statistics"""
        
        total_nodes = len(self.reputation_scores)
        avg_reputation = sum(self.reputation_scores.values()) / max(total_nodes, 1)
        
        # Calculate network decentralization metrics
        if self.network_distribution:
            gini_coefficient = self._calculate_gini_coefficient(list(self.network_distribution.values()))
        else:
            gini_coefficient = 0.0
        
        return {
            'total_nodes': total_nodes,
            'average_reputation': avg_reputation,
            'network_decentralization_gini': gini_coefficient,
            'max_node_reward_percentage': self.max_node_reward_percentage,
            'total_reward_pool': self.total_reward_pool,
            'consciousness_multiplier': self.consciousness_multiplier,
            'reasoning_multiplier': self.reasoning_multiplier,
            'quantum_multiplier': self.quantum_multiplier
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for measuring inequality"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n == 0:
            return 0.0
        
        cumulative_sum = sum(sorted_values)
        if cumulative_sum == 0:
            return 0.0
        
        gini = 0.0
        for i, value in enumerate(sorted_values):
            gini += (2 * (i + 1) - n - 1) * value
        
        return gini / (n * cumulative_sum) 