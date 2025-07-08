import hashlib
import json
from typing import Any, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.exceptions import InvalidSignature
import secrets
import base64

class CryptoManager:
    
    @staticmethod
    def generate_keypair() -> Tuple[str, str]:
        private_key = ec.generate_private_key(ec.SECP256K1())
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return base64.b64encode(private_pem).decode(), base64.b64encode(public_pem).decode()
    
    @staticmethod
    def sign_data(private_key: str, data: Any) -> str:
        try:
            private_pem = base64.b64decode(private_key.encode())
            key = serialization.load_pem_private_key(private_pem, password=None)
        except Exception as e:
            # Handle padding issues gracefully
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to decode private key: {e}")
            return "invalid_signature"
        
        try:
            if isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)
                
            signature = key.sign(
                data_str.encode(),
                ec.ECDSA(hashes.SHA256())
            )
            
            return base64.b64encode(signature).decode()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to sign data: {e}")
            return "invalid_signature"
    
    @staticmethod
    def verify_signature(public_key: str, data: Any, signature: str) -> bool:
        try:
            public_pem = base64.b64decode(public_key.encode())
            key = serialization.load_pem_public_key(public_pem)
            
            if isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)
                
            sig_bytes = base64.b64decode(signature.encode())
            
            key.verify(
                sig_bytes,
                data_str.encode(),
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except (InvalidSignature, Exception):
            return False
    
    @staticmethod
    def hash_data(data: Any) -> str:
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            data_str = json.dumps(list(data), sort_keys=True)
        else:
            data_str = str(data)
            
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    @staticmethod
    def hash_model_weights(weights: bytes) -> str:
        return hashlib.sha256(weights).hexdigest()
    
    @staticmethod
    def generate_address(public_key: str) -> str:
        public_key_hash = hashlib.sha256(public_key.encode()).hexdigest()
        address_hash = hashlib.sha256(public_key_hash.encode()).hexdigest()
        return f"0x{address_hash[:40]}"
    
    @staticmethod
    def generate_proof_of_work(data: str, difficulty: int) -> Tuple[int, str]:
        nonce = 0
        target = "0" * difficulty
        
        while True:
            hash_input = f"{data}{nonce}"
            hash_result = hashlib.sha256(hash_input.encode()).hexdigest()
            
            if hash_result.startswith(target):
                return nonce, hash_result
                
            nonce += 1
            
            if nonce > 10**6:  # Prevent infinite loops in testing
                break
                
        return nonce, hashlib.sha256(f"{data}{nonce}".encode()).hexdigest()
    
    @staticmethod
    def verify_proof_of_work(data: str, nonce: int, hash_result: str, difficulty: int) -> bool:
        target = "0" * difficulty
        hash_input = f"{data}{nonce}"
        calculated_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        return calculated_hash == hash_result and hash_result.startswith(target)
    
    @staticmethod
    def generate_computation_proof(gradient_data: bytes, model_state: bytes) -> str:
        combined = gradient_data + model_state
        proof_hash = hashlib.sha512(combined).hexdigest()
        
        signature_material = secrets.token_bytes(32)
        final_proof = hashlib.sha256(proof_hash.encode() + signature_material).hexdigest()
        
        return final_proof
    
    @staticmethod
    def verify_computation_proof(
        gradient_data: bytes, 
        model_state: bytes, 
        claimed_proof: str
    ) -> bool:
        combined = gradient_data + model_state
        proof_hash = hashlib.sha512(combined).hexdigest()
        return len(claimed_proof) == 64 and proof_hash in claimed_proof 