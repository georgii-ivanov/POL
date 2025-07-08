import time
import json
from typing import List, Optional, Dict, Any
from .types import Block, Transaction, ConsensusProof, TrainingValidation
from .crypto import CryptoManager
import logging

logger = logging.getLogger(__name__)

class Blockchain:
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.mining_reward = 10.0
        self.training_reward = 50.0
        self.balances: Dict[str, float] = {}
        
        self.create_genesis_block()
    
    def create_genesis_block(self) -> None:
        genesis_block = Block(
            index=0,
            previous_hash="0",
            timestamp=time.time(),
            transactions=[],
            training_epoch=0,
            model_state_hash="genesis",
            consensus_proof=ConsensusProof(
                authority_nodes=[],
                training_validations=[],
                aggregated_model_hash="genesis",
                consensus_signature="genesis",
                epoch=0
            ),
            nonce=0,
            hash=""
        )
        
        genesis_block.hash = self.calculate_hash(genesis_block)
        self.chain.append(genesis_block)
        logger.info("Genesis block created")
    
    def calculate_hash(self, block: Block) -> str:
        block_data = {
            "index": block.index,
            "previous_hash": block.previous_hash,
            "timestamp": block.timestamp,
            "transactions": [tx.to_dict() for tx in block.transactions],
            "training_epoch": block.training_epoch,
            "model_state_hash": block.model_state_hash,
            "nonce": block.nonce
        }
        return CryptoManager.hash_data(block_data)
    
    def get_latest_block(self) -> Block:
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        if not self.validate_transaction(transaction):
            logger.warning(f"Invalid transaction rejected: {transaction.id}")
            return False
            
        self.pending_transactions.append(transaction)
        logger.info(f"Transaction added to pending pool: {transaction.id}")
        return True
    
    def validate_transaction(self, transaction: Transaction) -> bool:
        if transaction.amount <= 0 and not transaction.training_proof:
            return False
            
        if transaction.from_address == transaction.to_address:
            return False
            
        sender_balance = self.get_balance(transaction.from_address)
        total_cost = transaction.amount + transaction.fee
        
        if sender_balance < total_cost and not transaction.training_proof:
            logger.warning(f"Insufficient balance for transaction {transaction.id}")
            return False
            
        return True
    
    def mine_pending_transactions(
        self, 
        mining_reward_address: str,
        consensus_proof: ConsensusProof,
        model_state_hash: str,
        training_epoch: int
    ) -> Block:
        reward_transaction = Transaction(
            id=f"reward_{time.time()}",
            from_address="system",
            to_address=mining_reward_address,
            amount=self.mining_reward,
            fee=0.0,
            timestamp=time.time()
        )
        
        self.pending_transactions.append(reward_transaction)
        
        block = Block(
            index=len(self.chain),
            previous_hash=self.get_latest_block().hash,
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            training_epoch=training_epoch,
            model_state_hash=model_state_hash,
            consensus_proof=consensus_proof,
            nonce=0
        )
        
        block.hash = self.mine_block(block)
        
        self.chain.append(block)
        self.update_balances(block)
        self.pending_transactions = []
        
        logger.info(f"Block {block.index} mined and added to chain")
        return block
    
    def mine_block(self, block: Block) -> str:
        target = "0" * self.difficulty
        
        while True:
            block.hash = self.calculate_hash(block)
            
            if block.hash.startswith(target):
                logger.info(f"Block mined with nonce: {block.nonce}")
                return block.hash
                
            block.nonce += 1
    
    def update_balances(self, block: Block) -> None:
        for transaction in block.transactions:
            if transaction.from_address != "system":
                self.balances[transaction.from_address] = (
                    self.balances.get(transaction.from_address, 0) - 
                    transaction.amount - transaction.fee
                )
            
            self.balances[transaction.to_address] = (
                self.balances.get(transaction.to_address, 0) + 
                transaction.amount
            )
            
            if transaction.training_proof:
                training_reward = self.calculate_training_reward(transaction.training_proof)
                self.balances[transaction.to_address] += training_reward
    
    def calculate_training_reward(self, training_proof) -> float:
        base_reward = self.training_reward
        validation_bonus = len(training_proof.validation_signatures) * 5.0
        return base_reward + validation_bonus
    
    def get_balance(self, address: str) -> float:
        return self.balances.get(address, 0.0)
    
    def is_chain_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            if current_block.hash != self.calculate_hash(current_block):
                logger.error(f"Invalid hash for block {current_block.index}")
                return False
                
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Invalid previous hash for block {current_block.index}")
                return False
                
            if not self.validate_proof_of_learning(current_block):
                logger.error(f"Invalid proof of learning for block {current_block.index}")
                return False
        
        return True
    
    def validate_proof_of_learning(self, block: Block) -> bool:
        consensus_proof = block.consensus_proof
        
        if len(consensus_proof.authority_nodes) < 3:
            return False
            
        valid_validations = sum(
            1 for validation in consensus_proof.training_validations 
            if validation.is_valid
        )
        
        required_validations = len(consensus_proof.authority_nodes) * 2 // 3 + 1
        
        return valid_validations >= required_validations
    
    def get_transaction_history(self, address: str) -> List[Transaction]:
        transactions = []
        for block in self.chain:
            for tx in block.transactions:
                if tx.from_address == address or tx.to_address == address:
                    transactions.append(tx)
        return transactions
    
    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None
    
    def get_block_by_index(self, index: int) -> Optional[Block]:
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain": [block.to_dict() for block in self.chain],
            "pending_transactions": [tx.to_dict() for tx in self.pending_transactions],
            "difficulty": self.difficulty,
            "balances": self.balances
        } 