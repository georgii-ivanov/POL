__version__ = "1.0.0"
__author__ = "POL Team"
__description__ = "Proof of Learning consensus blockchain with distributed GPT AI training"

from .blockchain import Blockchain, Block, Transaction
from .node import ProofOfLearningNode
from .consensus import ProofOfLearningConsensus
from .ai_engine import AITrainingEngine
from .network import P2PNetwork
from .wallet import Wallet
from .api import Web3API

__all__ = [
    "Blockchain",
    "Block", 
    "Transaction",
    "ProofOfLearningNode",
    "ProofOfLearningConsensus",
    "AITrainingEngine",
    "P2PNetwork",
    "Wallet",
    "Web3API"
] 