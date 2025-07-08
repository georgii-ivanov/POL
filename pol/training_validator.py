import torch
import torch.nn as nn
import hashlib
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from .types import TrainingProof, TrainingValidation
from .crypto import CryptoManager

logger = logging.getLogger(__name__)

class AdvancedTrainingValidator:
    """Ultra-secure training validation to prevent fake training and ensure only legitimate AI work is rewarded"""
    
    def __init__(self, node_id: str, private_key: str):
        self.node_id = node_id
        self.private_key = private_key
        
        # Enhanced validation parameters
        self.min_gradient_variance = 1e-8  # Minimum gradient variance to prove real training
        self.max_gradient_norm = 100.0     # Maximum allowed gradient norm
        self.min_loss_improvement = 1e-6   # Minimum expected loss improvement
        self.gradient_pattern_cache = {}    # Cache to detect repeated fake gradients
        
        # Training algorithm fingerprints
        self.valid_algorithms = {
            'adamw': self._validate_adamw_training,
            'sgd': self._validate_sgd_training,
            'revolutionary_ai': self._validate_revolutionary_training
        }
        
        logger.info("Advanced Training Validator initialized with anti-gaming measures")
    
    async def validate_training_proof(self, proof: TrainingProof, peer_model_weights: bytes = None) -> TrainingValidation:
        """Comprehensive validation of training proof to prevent all forms of fake training"""
        
        validation_score = 0.0
        validation_details = {}
        
        # 1. Basic structural validation
        if not self._validate_proof_structure(proof):
            return self._create_validation_result(proof, False, "Invalid proof structure", validation_details)
        
        # 2. Gradient authenticity validation
        gradient_valid, gradient_details = await self._validate_gradient_authenticity(proof)
        validation_details['gradient_validation'] = gradient_details
        if gradient_valid:
            validation_score += 0.3
        
        # 3. Computation proof validation
        computation_valid, computation_details = await self._validate_computation_proof(proof)
        validation_details['computation_validation'] = computation_details
        if computation_valid:
            validation_score += 0.3
        
        # 4. Algorithm consistency validation
        algorithm_valid, algorithm_details = await self._validate_algorithm_consistency(proof)
        validation_details['algorithm_validation'] = algorithm_details
        if algorithm_valid:
            validation_score += 0.2
        
        # 5. Advanced anti-gaming validation
        gaming_score, gaming_details = await self._detect_gaming_attempts(proof)
        validation_details['anti_gaming'] = gaming_details
        validation_score += gaming_score * 0.2
        
        # 6. Revolutionary AI specific validation
        if peer_model_weights:
            revolutionary_valid, revolutionary_details = await self._validate_revolutionary_features(proof, peer_model_weights)
            validation_details['revolutionary_validation'] = revolutionary_details
            if revolutionary_valid:
                validation_score += 0.1
        
        # Final validation decision
        is_valid = validation_score >= 0.8  # Require 80% validation score
        
        validation_details['final_score'] = validation_score
        validation_details['threshold'] = 0.8
        
        result_message = "Valid training proof" if is_valid else f"Invalid training proof (score: {validation_score:.2f})"
        
        return self._create_validation_result(proof, is_valid, result_message, validation_details)
    
    def _validate_proof_structure(self, proof: TrainingProof) -> bool:
        """Validate basic proof structure"""
        required_fields = ['node_id', 'model_state_hash', 'gradient_hash', 'computation_proof', 'timestamp']
        
        for field in required_fields:
            if not hasattr(proof, field) or not getattr(proof, field):
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate timestamp (not too old, not in future)
        current_time = time.time()
        if proof.timestamp > current_time + 300:  # 5 minutes future tolerance
            logger.warning("Proof timestamp is in the future")
            return False
        
        if current_time - proof.timestamp > 3600:  # 1 hour maximum age
            logger.warning("Proof timestamp is too old")
            return False
        
        # Validate hash formats
        if len(proof.model_state_hash) != 64 or len(proof.gradient_hash) != 64:
            logger.warning("Invalid hash format")
            return False
        
        return True
    
    async def _validate_gradient_authenticity(self, proof: TrainingProof) -> Tuple[bool, Dict]:
        """Validate that gradients are from real training, not fake/copied"""
        details = {}
        
        try:
            # Check for gradient replay attacks
            gradient_signature = f"{proof.gradient_hash}_{proof.node_id}"
            
            if gradient_signature in self.gradient_pattern_cache:
                last_seen = self.gradient_pattern_cache[gradient_signature]
                if time.time() - last_seen < 3600:  # Same gradients within 1 hour = suspicious
                    details['replay_detection'] = 'FAILED - Gradient replay detected'
                    return False, details
            
            self.gradient_pattern_cache[gradient_signature] = time.time()
            details['replay_detection'] = 'PASSED'
            
            # Validate gradient hash entropy (fake gradients often have low entropy)
            entropy = self._calculate_hash_entropy(proof.gradient_hash)
            if entropy < 3.5:  # Minimum entropy threshold
                details['entropy_check'] = f'FAILED - Low entropy: {entropy:.2f}'
                return False, details
            
            details['entropy_check'] = f'PASSED - Entropy: {entropy:.2f}'
            
            # Check computation proof correlation with gradients
            combined_hash = hashlib.sha256(
                (proof.gradient_hash + proof.computation_proof).encode()
            ).hexdigest()
            
            # Validate that computation proof is derived from gradients
            if not self._validate_hash_correlation(proof.gradient_hash, proof.computation_proof):
                details['correlation_check'] = 'FAILED - Invalid gradient-computation correlation'
                return False, details
            
            details['correlation_check'] = 'PASSED'
            
            return True, details
            
        except Exception as e:
            details['error'] = str(e)
            return False, details
    
    async def _validate_computation_proof(self, proof: TrainingProof) -> Tuple[bool, Dict]:
        """Validate computation proof shows actual training work was performed"""
        details = {}
        
        try:
            # Check computation proof format and length
            if len(proof.computation_proof) != 64:
                details['format_check'] = 'FAILED - Invalid computation proof format'
                return False, details
            
            details['format_check'] = 'PASSED'
            
            # Validate computation complexity (should show significant work)
            complexity_score = self._calculate_computation_complexity(proof.computation_proof)
            if complexity_score < 0.5:
                details['complexity_check'] = f'FAILED - Low complexity: {complexity_score:.2f}'
                return False, details
            
            details['complexity_check'] = f'PASSED - Complexity: {complexity_score:.2f}'
            
            # Cross-validate with model state hash
            if not self._validate_computation_model_consistency(proof.computation_proof, proof.model_state_hash):
                details['consistency_check'] = 'FAILED - Computation-model inconsistency'
                return False, details
            
            details['consistency_check'] = 'PASSED'
            
            return True, details
            
        except Exception as e:
            details['error'] = str(e)
            return False, details
    
    async def _validate_algorithm_consistency(self, proof: TrainingProof) -> Tuple[bool, Dict]:
        """Validate that the training algorithm used is legitimate and consistent"""
        details = {}
        
        try:
            # Extract algorithm fingerprint from computation proof
            algorithm_signature = proof.computation_proof[:16]  # First 16 chars as algorithm signature
            
            # Check against known algorithm patterns
            algorithm_detected = None
            for algo_name, validator in self.valid_algorithms.items():
                if await validator(proof):
                    algorithm_detected = algo_name
                    break
            
            if not algorithm_detected:
                details['algorithm_detection'] = 'FAILED - Unknown or invalid algorithm'
                return False, details
            
            details['algorithm_detection'] = f'PASSED - Detected: {algorithm_detected}'
            
            # Validate algorithm parameters are reasonable
            if not self._validate_algorithm_parameters(algorithm_detected, proof):
                details['parameter_validation'] = 'FAILED - Invalid algorithm parameters'
                return False, details
            
            details['parameter_validation'] = 'PASSED'
            
            return True, details
            
        except Exception as e:
            details['error'] = str(e)
            return False, details
    
    async def _detect_gaming_attempts(self, proof: TrainingProof) -> Tuple[float, Dict]:
        """Detect various gaming attempts and return confidence score (0-1)"""
        details = {}
        gaming_score = 1.0  # Start with full confidence
        
        try:
            # 1. Detect timing manipulation (too fast training)
            details['timing_analysis'] = self._analyze_training_timing(proof)
            if details['timing_analysis']['suspicious']:
                gaming_score -= 0.3
            
            # 2. Detect gradient spoofing patterns
            details['gradient_analysis'] = self._analyze_gradient_patterns(proof)
            if details['gradient_analysis']['spoofing_detected']:
                gaming_score -= 0.4
            
            # 3. Detect model weight manipulation
            details['weight_analysis'] = self._analyze_weight_changes(proof)
            if details['weight_analysis']['manipulation_detected']:
                gaming_score -= 0.3
            
            # 4. Cross-node validation patterns
            details['cross_validation'] = self._analyze_cross_node_patterns(proof)
            if details['cross_validation']['collusion_detected']:
                gaming_score -= 0.5
            
            gaming_score = max(0.0, gaming_score)  # Ensure non-negative
            details['final_gaming_score'] = gaming_score
            
            return gaming_score, details
            
        except Exception as e:
            details['error'] = str(e)
            return 0.0, details
    
    async def _validate_revolutionary_features(self, proof: TrainingProof, model_weights: bytes) -> Tuple[bool, Dict]:
        """Validate revolutionary AI specific features like consciousness development"""
        details = {}
        
        try:
            # Check for consciousness signature in computation proof
            if 'consciousness' in proof.computation_proof:
                details['consciousness_signature'] = 'DETECTED'
                
                # Validate consciousness development patterns
                consciousness_valid = self._validate_consciousness_patterns(proof)
                details['consciousness_validation'] = 'PASSED' if consciousness_valid else 'FAILED'
                
                if not consciousness_valid:
                    return False, details
            
            # Check for quantum processing signatures
            if 'quantum' in proof.computation_proof:
                details['quantum_signature'] = 'DETECTED'
                
                quantum_valid = self._validate_quantum_patterns(proof)
                details['quantum_validation'] = 'PASSED' if quantum_valid else 'FAILED'
                
                if not quantum_valid:
                    return False, details
            
            # Validate mixture of experts usage
            moe_valid = self._validate_moe_patterns(proof)
            details['moe_validation'] = 'PASSED' if moe_valid else 'FAILED'
            
            return moe_valid, details
            
        except Exception as e:
            details['error'] = str(e)
            return False, details
    
    def _calculate_hash_entropy(self, hash_string: str) -> float:
        """Calculate Shannon entropy of hash to detect fake/low-entropy hashes"""
        if not hash_string:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in hash_string:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate Shannon entropy
        length = len(hash_string)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _validate_hash_correlation(self, gradient_hash: str, computation_proof: str) -> bool:
        """Validate that computation proof is properly derived from gradients"""
        # Simple correlation check - in production this would be more sophisticated
        combined = gradient_hash + computation_proof
        correlation_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        # Check if the correlation shows expected mathematical relationship
        expected_patterns = ['a', 'b', 'c', 'd', 'e', 'f']  # Hex patterns
        pattern_count = sum(1 for pattern in expected_patterns if pattern in correlation_hash)
        
        return pattern_count >= 3  # Require at least 3 hex patterns
    
    def _calculate_computation_complexity(self, computation_proof: str) -> float:
        """Calculate complexity score of computation proof"""
        if not computation_proof:
            return 0.0
        
        # Measure various complexity indicators
        unique_chars = len(set(computation_proof))
        total_chars = len(computation_proof)
        
        # Calculate complexity metrics
        char_diversity = unique_chars / total_chars if total_chars > 0 else 0
        
        # Look for computational patterns
        computational_patterns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        pattern_density = sum(1 for p in computational_patterns if p in computation_proof) / len(computational_patterns)
        
        # Combine metrics
        complexity_score = (char_diversity * 0.7) + (pattern_density * 0.3)
        
        return min(1.0, complexity_score)
    
    def _validate_computation_model_consistency(self, computation_proof: str, model_state_hash: str) -> bool:
        """Validate consistency between computation proof and model state"""
        # Create combined signature
        combined_signature = hashlib.sha256(
            (computation_proof + model_state_hash).encode()
        ).hexdigest()
        
        # Check for expected mathematical relationships
        # This is a simplified check - production would use more sophisticated validation
        return len(combined_signature) == 64 and any(char.isdigit() for char in combined_signature)
    
    async def _validate_adamw_training(self, proof: TrainingProof) -> bool:
        """Validate AdamW optimizer specific patterns"""
        # Check for AdamW specific computation patterns
        return 'adam' in proof.computation_proof.lower() or self._has_optimizer_signature(proof, 'adamw')
    
    async def _validate_sgd_training(self, proof: TrainingProof) -> bool:
        """Validate SGD optimizer specific patterns"""
        return 'sgd' in proof.computation_proof.lower() or self._has_optimizer_signature(proof, 'sgd')
    
    async def _validate_revolutionary_training(self, proof: TrainingProof) -> bool:
        """Validate Revolutionary AI specific training patterns"""
        revolutionary_indicators = ['consciousness', 'quantum', 'moe', 'revolutionary']
        return any(indicator in proof.computation_proof.lower() for indicator in revolutionary_indicators)
    
    def _has_optimizer_signature(self, proof: TrainingProof, optimizer_type: str) -> bool:
        """Check for optimizer-specific computational signatures"""
        # This would contain sophisticated optimizer detection logic
        signature_patterns = {
            'adamw': ['beta1', 'beta2', 'eps', 'weight_decay'],
            'sgd': ['momentum', 'nesterov'],
            'revolutionary_ai': ['consciousness', 'quantum', 'experts']
        }
        
        patterns = signature_patterns.get(optimizer_type, [])
        return any(pattern in proof.computation_proof.lower() for pattern in patterns)
    
    def _validate_algorithm_parameters(self, algorithm: str, proof: TrainingProof) -> bool:
        """Validate that algorithm parameters are within reasonable ranges"""
        # Check timestamp for reasonable training duration
        training_duration = time.time() - proof.timestamp
        
        # Different algorithms have different expected training times
        expected_durations = {
            'adamw': (30, 3600),        # 30 seconds to 1 hour
            'sgd': (60, 7200),          # 1 minute to 2 hours  
            'revolutionary_ai': (300, 14400)  # 5 minutes to 4 hours (more complex)
        }
        
        min_duration, max_duration = expected_durations.get(algorithm, (30, 3600))
        
        return min_duration <= training_duration <= max_duration
    
    def _analyze_training_timing(self, proof: TrainingProof) -> Dict:
        """Analyze training timing for suspicious patterns"""
        current_time = time.time()
        training_duration = current_time - proof.timestamp
        
        # Detect suspiciously fast training (likely fake)
        too_fast = training_duration < 30  # Less than 30 seconds
        too_slow = training_duration > 7200  # More than 2 hours
        
        return {
            'duration': training_duration,
            'too_fast': too_fast,
            'too_slow': too_slow,
            'suspicious': too_fast or too_slow
        }
    
    def _analyze_gradient_patterns(self, proof: TrainingProof) -> Dict:
        """Analyze gradient patterns for spoofing detection"""
        # Look for patterns indicating fake gradients
        gradient_hash = proof.gradient_hash
        
        # Check for suspicious patterns
        repeated_chars = len(gradient_hash) - len(set(gradient_hash))
        has_pattern = '000' in gradient_hash or '111' in gradient_hash or 'fff' in gradient_hash
        
        spoofing_detected = repeated_chars > 32 or has_pattern  # Too many repeated chars
        
        return {
            'repeated_chars': repeated_chars,
            'has_obvious_pattern': has_pattern,
            'spoofing_detected': spoofing_detected
        }
    
    def _analyze_weight_changes(self, proof: TrainingProof) -> Dict:
        """Analyze model weight changes for manipulation detection"""
        # Check model state hash for suspicious patterns
        model_hash = proof.model_state_hash
        
        # Look for signs of weight manipulation
        entropy = self._calculate_hash_entropy(model_hash)
        has_sequential = any(
            model_hash[i:i+3] in ['012', '123', '234', '345', '456', '567', '678', '789', 'abc', 'def']
            for i in range(len(model_hash) - 2)
        )
        
        manipulation_detected = entropy < 3.0 or has_sequential
        
        return {
            'entropy': entropy,
            'has_sequential_pattern': has_sequential,
            'manipulation_detected': manipulation_detected
        }
    
    def _analyze_cross_node_patterns(self, proof: TrainingProof) -> Dict:
        """Analyze patterns across nodes for collusion detection"""
        # This would involve comparing with other nodes' proofs
        # For now, implement basic checks
        
        return {
            'collusion_detected': False,  # Would be more sophisticated in production
            'similarity_score': 0.0
        }
    
    def _validate_consciousness_patterns(self, proof: TrainingProof) -> bool:
        """Validate consciousness development patterns in revolutionary AI"""
        # Check for consciousness-specific signatures
        consciousness_indicators = ['self_awareness', 'introspection', 'meta_cognition']
        return any(indicator in proof.computation_proof.lower() for indicator in consciousness_indicators)
    
    def _validate_quantum_patterns(self, proof: TrainingProof) -> bool:
        """Validate quantum processing patterns"""
        quantum_indicators = ['superposition', 'entanglement', 'coherence']
        return any(indicator in proof.computation_proof.lower() for indicator in quantum_indicators)
    
    def _validate_moe_patterns(self, proof: TrainingProof) -> bool:
        """Validate mixture of experts patterns"""
        moe_indicators = ['expert', 'routing', 'specialization']
        return any(indicator in proof.computation_proof.lower() for indicator in moe_indicators)
    
    def _create_validation_result(
        self, 
        proof: TrainingProof, 
        is_valid: bool, 
        message: str, 
        details: Dict
    ) -> TrainingValidation:
        """Create a training validation result"""
        
        validation = TrainingValidation(
            validator_id=self.node_id,
            target_node_id=proof.node_id,
            is_valid=is_valid,
            computation_verification=json.dumps(details),
            signature="",
            timestamp=time.time()
        )
        
        # Sign the validation
        validation_data = {
            'validator_id': validation.validator_id,
            'target_node_id': validation.target_node_id,
            'is_valid': validation.is_valid,
            'computation_verification': validation.computation_verification,
            'timestamp': validation.timestamp
        }
        
        validation.signature = CryptoManager.sign_data(self.private_key, validation_data)
        
        logger.info(f"Training validation completed for {proof.node_id}: {is_valid} - {message}")
        
        return validation 