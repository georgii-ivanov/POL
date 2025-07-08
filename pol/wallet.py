import json
import os
from typing import Optional, Dict, List
import logging
from .types import Wallet, Transaction
from .crypto import CryptoManager

logger = logging.getLogger(__name__)

class WalletManager:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.wallets: Dict[str, Wallet] = {}
        
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Wallet manager initialized with data dir: {data_dir}")
    
    def create_wallet(self, wallet_name: str = "default") -> Wallet:
        private_key, public_key = CryptoManager.generate_keypair()
        address = CryptoManager.generate_address(public_key)
        
        wallet = Wallet(
            private_key=private_key,
            public_key=public_key,
            address=address,
            balance=0.0
        )
        
        self.wallets[wallet_name] = wallet
        self.save_wallet(wallet_name)
        
        logger.info(f"Created new wallet '{wallet_name}' with address: {address}")
        return wallet
    
    def load_wallet(self, wallet_name: str) -> Optional[Wallet]:
        wallet_file = os.path.join(self.data_dir, f"{wallet_name}.json")
        
        if not os.path.exists(wallet_file):
            logger.warning(f"Wallet file not found: {wallet_file}")
            return None
        
        try:
            with open(wallet_file, 'r') as f:
                wallet_data = json.load(f)
            
            wallet = Wallet(**wallet_data)
            self.wallets[wallet_name] = wallet
            
            logger.info(f"Loaded wallet '{wallet_name}' with address: {wallet.address}")
            return wallet
            
        except Exception as e:
            logger.error(f"Failed to load wallet '{wallet_name}': {e}")
            return None
    
    def save_wallet(self, wallet_name: str) -> bool:
        if wallet_name not in self.wallets:
            logger.error(f"Wallet '{wallet_name}' not found")
            return False
        
        wallet = self.wallets[wallet_name]
        wallet_file = os.path.join(self.data_dir, f"{wallet_name}.json")
        
        try:
            wallet_data = {
                'private_key': wallet.private_key,
                'public_key': wallet.public_key,
                'address': wallet.address,
                'balance': wallet.balance
            }
            
            with open(wallet_file, 'w') as f:
                json.dump(wallet_data, f, indent=2)
            
            logger.info(f"Saved wallet '{wallet_name}' to {wallet_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save wallet '{wallet_name}': {e}")
            return False
    
    def get_wallet(self, wallet_name: str) -> Optional[Wallet]:
        if wallet_name in self.wallets:
            return self.wallets[wallet_name]
        
        return self.load_wallet(wallet_name)
    
    def get_or_create_wallet(self, wallet_name: str = "default") -> Wallet:
        wallet = self.get_wallet(wallet_name)
        if wallet is None:
            wallet = self.create_wallet(wallet_name)
        return wallet
    
    def sign_transaction(self, wallet_name: str, transaction: Transaction) -> bool:
        wallet = self.get_wallet(wallet_name)
        if not wallet:
            logger.error(f"Wallet '{wallet_name}' not found")
            return False
        
        try:
            transaction_data = {
                'id': transaction.id,
                'from': transaction.from_address,
                'to': transaction.to_address,
                'amount': transaction.amount,
                'fee': transaction.fee,
                'timestamp': transaction.timestamp
            }
            
            if transaction.training_proof:
                transaction_data['training_proof'] = transaction.training_proof.to_dict()
            
            transaction.signature = CryptoManager.sign_data(wallet.private_key, transaction_data)
            
            logger.info(f"Signed transaction {transaction.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sign transaction: {e}")
            return False
    
    def verify_transaction_signature(self, transaction: Transaction) -> bool:
        try:
            transaction_data = {
                'id': transaction.id,
                'from': transaction.from_address,
                'to': transaction.to_address,
                'amount': transaction.amount,
                'fee': transaction.fee,
                'timestamp': transaction.timestamp
            }
            
            if transaction.training_proof:
                transaction_data['training_proof'] = transaction.training_proof.to_dict()
            
            sender_wallet = None
            for wallet in self.wallets.values():
                if wallet.address == transaction.from_address:
                    sender_wallet = wallet
                    break
            
            if not sender_wallet:
                logger.warning(f"Sender wallet not found for transaction {transaction.id}")
                return False
            
            return CryptoManager.verify_signature(
                sender_wallet.public_key,
                transaction_data,
                transaction.signature
            )
            
        except Exception as e:
            logger.error(f"Failed to verify transaction signature: {e}")
            return False
    
    def update_balance(self, wallet_name: str, new_balance: float) -> bool:
        wallet = self.get_wallet(wallet_name)
        if not wallet:
            return False
        
        wallet.balance = new_balance
        return self.save_wallet(wallet_name)
    
    def get_balance(self, wallet_name: str) -> float:
        wallet = self.get_wallet(wallet_name)
        return wallet.balance if wallet else 0.0
    
    def list_wallets(self) -> List[str]:
        wallet_files = [
            f.replace('.json', '') for f in os.listdir(self.data_dir)
            if f.endswith('.json')
        ]
        return wallet_files
    
    def export_wallet(self, wallet_name: str, export_path: str) -> bool:
        wallet = self.get_wallet(wallet_name)
        if not wallet:
            return False
        
        try:
            export_data = {
                'wallet_name': wallet_name,
                'private_key': wallet.private_key,
                'public_key': wallet.public_key,
                'address': wallet.address,
                'balance': wallet.balance,
                'export_timestamp': CryptoManager.hash_data(str(wallet.balance))
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported wallet '{wallet_name}' to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export wallet: {e}")
            return False
    
    def import_wallet(self, import_path: str, wallet_name: str = None) -> Optional[Wallet]:
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            if not wallet_name:
                wallet_name = import_data.get('wallet_name', 'imported')
            
            wallet = Wallet(
                private_key=import_data['private_key'],
                public_key=import_data['public_key'],
                address=import_data['address'],
                balance=import_data.get('balance', 0.0)
            )
            
            self.wallets[wallet_name] = wallet
            self.save_wallet(wallet_name)
            
            logger.info(f"Imported wallet '{wallet_name}' from {import_path}")
            return wallet
            
        except Exception as e:
            logger.error(f"Failed to import wallet: {e}")
            return None
    
    def generate_address_from_public_key(self, public_key: str) -> str:
        return CryptoManager.generate_address(public_key)
    
    def validate_address(self, address: str) -> bool:
        return (
            address.startswith("0x") and 
            len(address) == 42 and 
            all(c in "0123456789abcdefABCDEF" for c in address[2:])
        ) 