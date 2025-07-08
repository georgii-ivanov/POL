# Proof of Learning Blockchain

A blockchain system that uses **Proof of Learning** consensus where network nodes earn coins by training AI models. This creates a decentralized AI training network with blockchain incentives.

## 🚀 Features

- **Proof of Learning Consensus**: Nodes earn rewards by training AI models
- **GPT-like AI Models**: Transformer architecture for language modeling
- **Distributed Training**: Collaborative AI model training across network nodes
- **Web3 Compatible**: JSON-RPC and REST API support
- **OpenAI Compatible**: Drop-in replacement for OpenAI API endpoints
- **Authority-based Validation**: Designated nodes validate training authenticity
- **Auto-scaling**: Adjusts model size based on available hardware
- **Modern Architecture**: RMSNorm, SwiGLU, RoPE, Grouped Query Attention
- **Memory Efficient**: Mixed precision training, gradient checkpointing
- **Secure Cryptography**: ECDSA signatures and cryptographic proofs

## 🏗️ Architecture

### Core Components

1. **Blockchain Layer**: Custom blockchain with proof-of-learning consensus
2. **AI Training Engine**: Distributed GPT model training with PyTorch
3. **P2P Network**: WebSocket-based peer-to-peer communication
4. **Consensus Mechanism**: Authority nodes validate training proofs
5. **Web3 API**: Ethereum-compatible JSON-RPC interface
6. **OpenAI API**: Compatible chat completions endpoint

### How It Works

1. **Training**: Nodes train GPT models on shared datasets
2. **Proof Generation**: Training generates cryptographic proofs
3. **Validation**: Authority nodes verify training authenticity
4. **Consensus**: Network reaches agreement on valid training
5. **Rewards**: Successful trainers earn POL coins
6. **Mining**: Blocks are mined with training consensus proofs

## 📦 Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd proof-of-learning-blockchain

# Install Python dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Alternative: Using Virtual Environment

```bash
python -m venv pol_env
source pol_env/bin/activate  # On Windows: pol_env\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 🚀 Quick Start

### 1. Create Network Configuration

```bash
# Generate a sample network with authority and regular nodes
pol-node create-network
```

This creates configuration files for 6 nodes (3 authority + 3 regular).

### 2. Start Authority Nodes

```bash
# Terminal 1: Start first authority node
pol-node run --config-file authority_0_config.json

# Terminal 2: Start second authority node  
pol-node run --config-file authority_1_config.json

# Terminal 3: Start third authority node
pol-node run --config-file authority_2_config.json
```

### 3. Start Regular Nodes

```bash
# Terminal 4: Start regular node
pol-node run --config-file node_3_config.json

# Terminal 5: Start another regular node
pol-node run --config-file node_4_config.json
```

### 4. Interact with the Network

```bash
# Check node status
pol-node status --port 8080

# Check balance
pol-node balance --port 8080

# Send transaction
pol-node send --port 8080 --to 0x1234... --amount 10.0

# Chat with AI model
pol-node chat --port 8080 --prompt "Explain how blockchain consensus works"
```

## 🔧 Manual Node Setup

### Single Node

```bash
pol-node run \
  --node-id my-node \
  --port 8000 \
  --api-port 8080 \
  --training \
  --data-dir ./data
```

### Authority Node

```bash
pol-node run \
  --node-id authority-1 \
  --port 8000 \
  --api-port 8080 \
  --is-authority \
  --training \
  --boot-nodes localhost:8001,localhost:8002
```

### Generate Custom Configuration

```bash
pol-node generate-config \
  --output my_config.json \
  --node-id my-custom-node \
  --port 8000 \
  --is-authority
```

## 🌐 API Usage

### Web3 JSON-RPC (Ethereum Compatible)

```bash
# Get balance
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_getBalance", 
    "params": ["0x1234..."],
    "id": 1
  }'

# Send transaction
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_sendTransaction",
    "params": [{
      "to": "0x5678...",
      "value": "0x9184e72a000"
    }],
    "id": 1
  }'
```

### REST API

```bash
# Node status
curl http://localhost:8080/api/v1/status

# Get balance
curl http://localhost:8080/api/v1/balance/0x1234...

# Network info
curl http://localhost:8080/api/v1/network/info

# Training status
curl http://localhost:8080/api/v1/training/status
```

### OpenAI Compatible API

```bash
# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pol-gpt",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'

# List models
curl http://localhost:8080/v1/models
```

### Python SDK Example

```python
import requests

# Using as OpenAI replacement
class POLClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
    
    def chat_completion(self, messages, max_tokens=100):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "pol-gpt",
                "messages": messages,
                "max_tokens": max_tokens
            }
        )
        return response.json()

# Usage
client = POLClient()
response = client.chat_completion([
    {"role": "user", "content": "Explain quantum computing"}
])
print(response["choices"][0]["message"]["content"])
```

## 🔐 Security Features

### Training Proof Verification

- **Gradient Hashing**: Cryptographic hashes of model gradients
- **Computation Proofs**: Proof of actual training computation
- **Authority Validation**: Multiple authority nodes validate each proof
- **Reputation System**: Nodes build reputation through honest training
- **Anti-Gaming**: Measures prevent fake training submissions

### Blockchain Security

- **ECDSA Signatures**: Standard cryptographic signatures
- **Block Validation**: Full block and transaction validation
- **Consensus Rules**: Consensus requirements (67% agreement)
- **Network Integrity**: P2P network with peer reputation tracking

## ⚙️ Configuration

### Node Configuration

```json
{
  "node_id": "my-node",
  "port": 8000,
  "is_authority": false,
  "boot_nodes": ["localhost:8001", "localhost:8002"],
  "data_dir": "./data/my-node",
  "training_enabled": true,
  "model_size": 1000000000,
  "batch_size": 32,
  "learning_rate": 0.0001
}
```

### AI Training Configuration

**Model Architecture:**
- **Model Size**: Configurable parameter count (auto-scaled based on hardware)
- **Architecture**: Standard transformer with modern techniques:
  - **RMSNorm**: Layer normalization variant
  - **SwiGLU**: Activation function from recent research
  - **RoPE**: Rotary position embeddings
  - **Grouped Query Attention**: Memory-efficient attention mechanism
  - **Mixed Precision**: FP16/BF16 training for memory efficiency
  - **Gradient Checkpointing**: Reduced memory usage for larger models

**Training Features:**
- **Context Length**: Configurable sequence length
- **Vocabulary**: Standard tokenizer vocabulary
- **Batch Size**: Auto-adjusted based on GPU memory
- **Learning Rate**: Configurable with scheduling options
- **Gradient Accumulation**: Dynamic based on hardware
- **Hardware Scaling**: Automatically scales model based on available resources

## 📊 Monitoring

### Node Metrics

```bash
# Real-time status
watch -n 5 'curl -s http://localhost:8080/api/v1/status | jq'

# Training progress
curl http://localhost:8080/api/v1/training/status

# Network health
curl http://localhost:8080/api/v1/network/info
```

### Blockchain Explorer

Access the API to build custom blockchain explorers:

- Block information: `/api/v1/block/{number}`
- Transaction history: `/api/v1/transactions/{address}`
- Network statistics: `/api/v1/network/info`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black mypy flake8

# Run tests
pytest

# Format code
black pol/

# Type checking
mypy pol/
```

## 📋 System Requirements

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 10GB SSD
- **Network**: Stable internet connection
- **GPU**: CUDA-compatible (optional but recommended)

### Recommended for Training

- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 100GB+ NVMe SSD
- **GPU**: NVIDIA RTX 3090/4090 or better
- **Network**: High-speed internet (100+ Mbps)

## 🔍 Troubleshooting

### Common Issues

1. **CUDA not found**: Install CUDA toolkit and PyTorch with CUDA support
2. **Connection refused**: Check firewall settings and port availability
3. **Out of memory**: Reduce batch size or model size
4. **Peer discovery issues**: Verify boot node addresses

### Debug Mode

```bash
# Run with verbose logging
PYTHONPATH=. python -m pol.cli run --config-file config.json --log-level DEBUG
```

### Health Checks

```bash
# Check if node is responsive
curl -f http://localhost:8080/api/v1/status || echo "Node not healthy"

# Verify blockchain integrity
curl http://localhost:8080/api/v1/network/info | jq '.blockchain_height'
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the ML framework
- Hugging Face for transformer models
- FastAPI for the web framework
- The blockchain and AI research communities

## 🧠 AI Training Features

The system implements a distributed AI training network with the following capabilities:

### **🧠 Model Architecture**
- **Transformer Models**: Standard GPT-like architecture with modern improvements
- **Configurable Size**: Model parameters scale based on available hardware
- **Memory Optimization**: Mixed precision training and gradient checkpointing
- **Advanced Components**: RMSNorm, SwiGLU activations, RoPE embeddings

### **🎯 Training Features**
- **Distributed Learning**: Multiple nodes contribute to model training
- **Data Acquisition**: Automated dataset loading from HuggingFace and other sources
- **Training Validation**: Cryptographic proofs verify actual training occurred
- **Consensus Mechanism**: Network validates training quality and authenticity
- **Reward System**: Nodes earn coins for contributing valid training

### **📊 Monitoring & APIs**
- **Training Status**: Real-time monitoring of training progress
- **Model Performance**: Track loss, accuracy, and other metrics
- **Network Health**: Monitor peer connections and blockchain state
- **OpenAI Compatibility**: Standard chat completion API endpoints

### **Usage Examples**

```bash
# Start a training node
pol-node run --config-file config/production_config.json

# Monitor training progress
pol-node status

# Chat with the trained model
pol-node chat --prompt "Explain machine learning concepts" --max-tokens 200

# Check training metrics
curl http://localhost:8080/api/v1/training/status
```

**API Usage:**
```python
import requests

# Chat with trained model
response = requests.post("http://localhost:8080/v1/chat/completions", json={
    "model": "pol-gpt",
    "messages": [{"role": "user", "content": "Explain blockchain technology"}],
    "max_tokens": 150
})

data = response.json()
print(f"AI Response: {data['choices'][0]['message']['content']}")

# Monitor training status
status = requests.get("http://localhost:8080/api/v1/training/status").json()
print(f"Current Epoch: {status['current_epoch']}")
print(f"Training Loss: {status['current_loss']:.4f}")
```

## 🔮 Roadmap

### ✅ **Current Features**
- [x] **Proof of Learning Consensus** - Blockchain consensus based on AI training
- [x] **Distributed Training** - Multiple nodes training collaboratively
- [x] **Modern AI Architecture** - Transformer models with latest techniques
- [x] **OpenAI Compatibility** - Standard API endpoints for model interaction
- [x] **Security & Validation** - Cryptographic proof verification
- [x] **Web3 Integration** - Ethereum-compatible JSON-RPC interface

### 🚀 **Future Development**
- [ ] **🧠 Enhanced Model Architectures**: Larger models and improved architectures
- [ ] **🌐 Multimodal Support**: Vision, audio, and text processing
- [ ] **🔗 Cross-chain Compatibility**: Support for multiple blockchain networks
- [ ] **📱 Mobile Node Support**: Lightweight nodes for mobile devices
- [ ] **🤝 Advanced Collaboration**: Enhanced multi-node training coordination
- [ ] **🔧 Developer Tools**: Better debugging and monitoring tools
- [ ] **📊 Analytics Dashboard**: Web-based training and network monitoring

---

## 🎯 Proof of Learning System

**A practical implementation of blockchain-based AI training incentives**

This system demonstrates how blockchain technology can incentivize distributed AI training, creating a network where participants earn rewards for contributing computational resources to train machine learning models.

```bash
# Get started with proof-of-learning
pol-node run --config-file config/production_config.json

# Interact with the trained model
pol-node chat --prompt "Hello, how does this system work?"
```

**The system combines established blockchain and AI technologies to create a working proof-of-learning network.** 🔗 