from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging
from .node import ProofOfLearningNode
from .types import NodeConfig, AITrainingConfig, Transaction
from .crypto import CryptoManager

logger = logging.getLogger(__name__)

# Web3 compatible request/response models
class TransactionRequest(BaseModel):
    to: str
    value: float
    gas: Optional[float] = 0.1
    data: Optional[str] = None

class TransactionResponse(BaseModel):
    hash: str
    from_address: str
    to: str
    value: float
    gas: float
    status: str

class BlockRequest(BaseModel):
    block_number: Optional[int] = None
    block_hash: Optional[str] = None

class BlockResponse(BaseModel):
    number: int
    hash: str
    parent_hash: str
    timestamp: float
    transactions: List[Dict[str, Any]]
    training_epoch: int
    model_state_hash: str

class BalanceResponse(BaseModel):
    address: str
    balance: float

# OpenAI compatible models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "pol-gpt"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class Web3API:
    def __init__(self, node: ProofOfLearningNode):
        self.node = node
        self.app = FastAPI(
            title="Proof of Learning Blockchain API",
            description="Web3 and OpenAI compatible API for POL blockchain",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        logger.info("Web3 API initialized")
    
    def _setup_routes(self):
        # Web3 JSON-RPC compatible endpoints
        @self.app.post("/")
        async def json_rpc_handler(request: Dict[str, Any]):
            method = request.get("method")
            params = request.get("params", [])
            request_id = request.get("id", 1)
            
            try:
                if method == "eth_getBalance":
                    address = params[0]
                    balance = self.node.blockchain.get_balance(address)
                    return {
                        "id": request_id,
                        "jsonrpc": "2.0",
                        "result": hex(int(balance * 10**18))  # Convert to wei
                    }
                
                elif method == "eth_sendTransaction":
                    tx_data = params[0]
                    transaction = await self.node.create_transaction(
                        to_address=tx_data["to"],
                        amount=float(int(tx_data["value"], 16)) / 10**18,
                        fee=float(int(tx_data.get("gas", "0x5208"), 16)) / 10**18
                    )
                    
                    if transaction:
                        await self.node.broadcast_transaction(transaction)
                        return {
                            "id": request_id,
                            "jsonrpc": "2.0",
                            "result": transaction.id
                        }
                    else:
                        raise HTTPException(status_code=400, detail="Failed to create transaction")
                
                elif method == "eth_getBlockByNumber":
                    block_number = int(params[0], 16) if params[0] != "latest" else len(self.node.blockchain.chain) - 1
                    block = self.node.blockchain.get_block_by_index(block_number)
                    
                    if block:
                        return {
                            "id": request_id,
                            "jsonrpc": "2.0",
                            "result": {
                                "number": hex(block.index),
                                "hash": block.hash,
                                "parentHash": block.previous_hash,
                                "timestamp": hex(int(block.timestamp)),
                                "transactions": [tx.to_dict() for tx in block.transactions],
                                "trainingEpoch": block.training_epoch,
                                "modelStateHash": block.model_state_hash
                            }
                        }
                    else:
                        return {
                            "id": request_id,
                            "jsonrpc": "2.0",
                            "result": None
                        }
                
                elif method == "eth_blockNumber":
                    return {
                        "id": request_id,
                        "jsonrpc": "2.0",
                        "result": hex(len(self.node.blockchain.chain) - 1)
                    }
                
                elif method == "net_version":
                    return {
                        "id": request_id,
                        "jsonrpc": "2.0",
                        "result": "1337"  # POL network ID
                    }
                
                else:
                    raise HTTPException(status_code=400, detail=f"Method {method} not supported")
            
            except Exception as e:
                logger.error(f"JSON-RPC error: {e}")
                return {
                    "id": request_id,
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }
        
        # REST API endpoints
        @self.app.get("/api/v1/status")
        async def get_status():
            return self.node.get_node_status()
        
        @self.app.get("/api/v1/balance/{address}", response_model=BalanceResponse)
        async def get_balance(address: str):
            balance = self.node.blockchain.get_balance(address)
            return BalanceResponse(address=address, balance=balance)
        
        @self.app.post("/api/v1/transaction", response_model=TransactionResponse)
        async def send_transaction(tx_request: TransactionRequest):
            transaction = await self.node.create_transaction(
                to_address=tx_request.to,
                amount=tx_request.value,
                fee=tx_request.gas
            )
            
            if not transaction:
                raise HTTPException(status_code=400, detail="Failed to create transaction")
            
            success = await self.node.broadcast_transaction(transaction)
            
            return TransactionResponse(
                hash=transaction.id,
                from_address=transaction.from_address,
                to=transaction.to_address,
                value=transaction.amount,
                gas=transaction.fee,
                status="pending" if success else "failed"
            )
        
        @self.app.get("/api/v1/block/{block_identifier}")
        async def get_block(block_identifier: str):
            try:
                if block_identifier == "latest":
                    block = self.node.blockchain.get_latest_block()
                elif block_identifier.startswith("0x"):
                    block = self.node.blockchain.get_block_by_hash(block_identifier)
                else:
                    block_number = int(block_identifier)
                    block = self.node.blockchain.get_block_by_index(block_number)
                
                if not block:
                    raise HTTPException(status_code=404, detail="Block not found")
                
                return BlockResponse(
                    number=block.index,
                    hash=block.hash,
                    parent_hash=block.previous_hash,
                    timestamp=block.timestamp,
                    transactions=[tx.to_dict() for tx in block.transactions],
                    training_epoch=block.training_epoch,
                    model_state_hash=block.model_state_hash
                )
            
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid block identifier")
        
        @self.app.get("/api/v1/transactions/{address}")
        async def get_transaction_history(address: str):
            transactions = self.node.blockchain.get_transaction_history(address)
            return [tx.to_dict() for tx in transactions]
        
        @self.app.get("/api/v1/peers")
        async def get_peers():
            return {
                "connected_peers": self.node.network.get_connected_peers(),
                "peer_count": self.node.network.get_peer_count(),
                "authority_count": self.node.network.get_authority_count()
            }
        
        @self.app.get("/api/v1/training/status")
        async def get_training_status():
            status = {
                "current_epoch": self.node.current_training_epoch,
                "training_enabled": self.node.config.training_enabled,
                "model_state": self.node.ai_engine.get_model_state().__dict__,
                "consensus_status": self.node.consensus.get_consensus_status()
            }
            
            # Add revolutionary AI capabilities if available
            if hasattr(self.node.ai_engine, 'get_revolutionary_capabilities'):
                status["revolutionary_capabilities"] = self.node.ai_engine.get_revolutionary_capabilities()
            
            return status
        
        # Revolutionary AI endpoints
        @self.app.get("/api/v1/consciousness/status")
        async def get_consciousness_status():
            """Get current consciousness and reasoning levels"""
            if hasattr(self.node.ai_engine, 'get_revolutionary_capabilities'):
                caps = self.node.ai_engine.get_revolutionary_capabilities()
                return {
                    "consciousness_level": caps.get("consciousness_level", 0),
                    "reasoning_quality": caps.get("reasoning_quality", 0),
                    "quantum_coherence": caps.get("quantum_coherence", 0),
                    "consciousness_history": getattr(self.node.ai_engine, 'consciousness_history', [])[-100:],
                    "reasoning_history": getattr(self.node.ai_engine, 'reasoning_quality_history', [])[-100:],
                    "quantum_history": getattr(self.node.ai_engine, 'quantum_coherence_history', [])[-100:]
                }
            return {"error": "Revolutionary AI not available"}
        
        @self.app.post("/api/v1/consciousness/chat")
        async def consciousness_chat(request: ChatCompletionRequest):
            """Chat with full consciousness insights"""
            try:
                user_messages = [msg for msg in request.messages if msg.role == "user"]
                if not user_messages:
                    raise HTTPException(status_code=400, detail="No user message found")
                
                prompt = user_messages[-1].content
                
                # Generate with consciousness insights
                if hasattr(self.node.ai_engine, 'model') and hasattr(self.node.ai_engine.model, 'generate_with_consciousness'):
                    inputs = self.node.ai_engine.tokenizer(prompt, return_tensors='pt', truncation=True)
                    input_ids = inputs['input_ids'].to(self.node.ai_engine.device)
                    
                    result = self.node.ai_engine.model.generate_with_consciousness(
                        input_ids=input_ids,
                        max_new_tokens=request.max_tokens or 100,
                        temperature=request.temperature or 0.7,
                        return_insights=True
                    )
                    
                    generated_text = self.node.ai_engine.tokenizer.decode(
                        result["generated_tokens"][0][input_ids.shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    
                    return {
                        "response": generated_text,
                        "consciousness_insights": result.get("insights", []),
                        "consciousness_level": result.get("final_consciousness_level", 0),
                        "generation_steps": len(result.get("insights", []))
                    }
                else:
                    # Fallback to regular generation
                    generated_text = self.node.ai_engine.generate_text(prompt, request.max_tokens or 100)
                    return {"response": generated_text, "insights": "Standard generation used"}
                    
            except Exception as e:
                logger.error(f"Error in consciousness chat: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # OpenAI compatible chat completions endpoint
        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest):
            try:
                # Extract the last user message
                user_messages = [msg for msg in request.messages if msg.role == "user"]
                if not user_messages:
                    raise HTTPException(status_code=400, detail="No user message found")
                
                prompt = user_messages[-1].content
                
                # Generate response using the AI engine
                generated_text = self.node.ai_engine.generate_text(
                    prompt=prompt,
                    max_length=request.max_tokens or 100
                )
                
                # Remove the original prompt from the generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                response = ChatCompletionResponse(
                    id=f"chatcmpl-{CryptoManager.hash_data(prompt)[:8]}",
                    created=int(asyncio.get_event_loop().time()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }],
                    usage={
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(generated_text.split()),
                        "total_tokens": len(prompt.split()) + len(generated_text.split())
                    }
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Error in chat completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "pol-gpt",
                        "object": "model",
                        "created": 1677610602,
                        "owned_by": "proof-of-learning",
                        "permission": [],
                        "root": "pol-gpt",
                        "parent": None
                    }
                ]
            }
        
        # Mining control endpoints
        @self.app.post("/api/v1/mining/start")
        async def start_mining():
            self.node.mining_enabled = True
            return {"status": "Mining started"}
        
        @self.app.post("/api/v1/mining/stop")
        async def stop_mining():
            self.node.mining_enabled = False
            return {"status": "Mining stopped"}
        
        # Wallet endpoints
        @self.app.get("/api/v1/wallet/address")
        async def get_wallet_address():
            return {"address": self.node.wallet.address}
        
        @self.app.get("/api/v1/wallet/balance")
        async def get_wallet_balance():
            return {"balance": self.node.wallet.balance}
        
        # Network information
        @self.app.get("/api/v1/network/info")
        async def get_network_info():
            return {
                "node_id": self.node.node_id,
                "network_stats": self.node.network.get_network_stats(),
                "blockchain_height": len(self.node.blockchain.chain),
                "total_supply": sum(self.node.blockchain.balances.values()),
                "difficulty": self.node.blockchain.difficulty
            }

def create_api_server(node: ProofOfLearningNode) -> FastAPI:
    api = Web3API(node)
    return api.app

async def run_api_server(node: ProofOfLearningNode, host: str = "0.0.0.0", port: int = 8080):
    import uvicorn
    
    app = create_api_server(node)
    
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    logger.info(f"Starting API server on {host}:{port}")
    await server.serve() 