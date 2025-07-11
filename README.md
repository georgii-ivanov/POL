# POL-AI Hybrid Chain

A proof-of-concept blockchain combining **Proof-of-Work (PoW) proposer** + **Prysm PoS finality** + **Sybil-proof ACK-quorum** for AI model training validation.

## Architecture

- **ai/trainer.py**: Hivemind + Petals + LoRA training with DeepSeek-R1
- **ai/local_averager.py**: Loss gate validation + Krum + FedAvg aggregation  
- **dht_daemon.py**: libp2p-Kademlia bootstrap (port 13337, k=20, AutoNAT)
- **engine/ai_engine.go**: External engine for Geth with ACK-quorum validation
- **prysm/ack_quorum.go**: Prysm patch for ACK quorum counting and slashing

## Key Features

✅ **Single-node genesis**: Runs from genesis with just one validator  
✅ **Auto-scaling**: Committee size and quorum adapt as validators join  
✅ **Model pre-caching**: DeepSeek-R1 downloaded during Docker build  
✅ **Deterministic genesis**: Same GENESIS_SEED = same validator keys  
✅ **30s ACK timeout**: Empty AI blocks if quorum not reached  
✅ **Loss gate validation**: LoRA updates must improve model performance  
✅ **Automatic slashing**: Invalid AI blocks trigger proposer penalties  

## Ports

Default ports (configurable in `.env` file):

| Service | Port | Protocol |
|---------|------|----------|
| DHT | 13337 | libp2p |
| Geth P2P | 30303 | devp2p |
| Geth HTTP | 8545 | JSON-RPC |
| Geth AuthRPC | 8551 | JWT |
| Prysm gRPC | 4000 | gRPC |
| Prysm REST | 3500 | HTTP |
| Prysm P2P | 13000 | libp2p |
| AI Engine | 8552 | HTTP |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- 16GB+ RAM (for DeepSeek-R1 model)
- NVIDIA GPU (recommended)

### 1. First Startup

**Create `.env` file with your configuration:**
```bash
# Create .env file
cat > .env << 'EOF'
# POL-AI Hybrid Chain Configuration

# Genesis seed for deterministic validator generation (64 hex characters)
# Generate with: openssl rand -hex 32
GENESIS_SEED=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

# Network configuration
GENESIS_HOST=localhost

# AI Configuration
AI_QUORUM=auto

# Optional: Custom ports (defaults shown)
DHT_PORT=13337
GETH_HTTP_PORT=8545
GETH_P2P_PORT=30303
GETH_AUTH_PORT=8551
PRYSM_GRPC_PORT=4000
PRYSM_REST_PORT=3500
PRYSM_P2P_PORT=13000
AI_ENGINE_PORT=8552
EOF

# Generate a unique GENESIS_SEED (recommended)
GENESIS_SEED=$(openssl rand -hex 32)
sed -i "s/GENESIS_SEED=.*/GENESIS_SEED=$GENESIS_SEED/" .env
```

**For Windows users:**
```cmd
REM Create .env file manually with these contents:
echo # POL-AI Hybrid Chain Configuration > .env
echo. >> .env
echo # Genesis seed for deterministic validator generation >> .env
echo GENESIS_SEED=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef >> .env
echo. >> .env
echo # Network configuration >> .env
echo GENESIS_HOST=localhost >> .env
echo. >> .env
echo # AI Configuration >> .env
echo AI_QUORUM=auto >> .env

REM Then replace the default GENESIS_SEED with a unique one if desired
```

**Build and start the network:**
```bash
# Build Docker images
docker-compose build

# Start all services
docker-compose up -d

# Watch logs
docker-compose logs -f
```

### 2. First Block Finalization

The network will automatically:
1. Start DHT daemon in solo mode
2. Load DeepSeek-R1 model (cached)
3. Generate deterministic genesis validator 
4. Begin PoW mining with AI validation
5. **First block finalizes with 1-node quorum**

### 3. Common Commands

**Monitor logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f geth
docker-compose logs -f prysm-beacon
docker-compose logs -f ai-trainer
```

**Manage services:**
```bash
# Stop all services
docker-compose down

# Restart a specific service
docker-compose restart geth

# View running containers
docker-compose ps
```

### 4. Add Validators

```bash
# Additional validators join via standard deposits
# Committee size scales automatically: quorum = max(1, ceil(validators/2))
```

## Environment Variables

All configuration is stored in `.env` file (created from `env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GENESIS_SEED` | generated | Deterministic seed for genesis validator (64 hex chars) |
| `GENESIS_HOST` | localhost | Bootstrap peer for DHT discovery |
| `AI_QUORUM` | auto | Manual ACK quorum override |
| `DHT_PORT` | 13337 | DHT service port |
| `GETH_HTTP_PORT` | 8545 | Geth JSON-RPC port |
| `GETH_AUTH_PORT` | 8551 | Geth AuthRPC port |
| `PRYSM_GRPC_PORT` | 4000 | Prysm gRPC port |
| `PRYSM_REST_PORT` | 3500 | Prysm REST API port |
| `AI_ENGINE_PORT` | 8552 | AI Engine port |
| `SLASHER_PORT` | 4003 | Prysm Slasher gRPC port |

## Architecture Details

### LoRA Training
- **r=8, α=16, dropout=0.05**
- Target modules: Q/K/V/O + MLP layers
- AdamW optimizer (lr=1e-4, β1=0.9, β2=0.95)
- LoRA δW ≤150KB FP16, published to DHT

### Loss Gate Validation
- 512 DEV_TEXTS (readable sentences ≤256 chars)
- Accept if `loss_after < 0.999 × loss_before`
- Skip gate if `loss_before < 0.05`
- Slashing for failed validation

### ACK Quorum Protocol
1. **Committee formation**: Prysm computes committee for block's slot (currently all active validators)
2. **Peer ID derivation**: `peerID = sha256(pubkey)[:8]` for each committee member  
3. **ML validation**: `local_averager.py` runs loss gate on committee members
4. **ACK publishing**: Valid updates → `ack:{updateID[:16]}:{peerID}` = `0x01` (hex format, TTL 3600s)
5. **Block format**: Binary ExtraData = `[32 bytes parentHash][32 bytes updateID]` (64 bytes total)
6. **Consensus validation**: Prysm counts ACKs only from committee members for that slot
7. **Quorum requirement**: votes ≥ max(1, ceil(committee_size/2)) to accept, otherwise slash proposer
8. **Timeout handling**: 30s timeout → empty AI block (updateID = 32 zero bytes)

### Consensus Flow
```
Training Layer:    Trainer → LoRA δW → DHT
Validation Layer:  Averager → Loss Gate → ACK Publication  
Consensus Layer:   Engine → PoW Block → Prysm ACK Count → Finality
```

## Volume Mounts

| Volume | Purpose |
|--------|---------|
| `chain_data/` | Blockchain state |
| `dht_db/` | DHT persistence |
| `model_cache/` | DeepSeek-R1 + LoRA deltas |
| `secrets_data/` | JWT secrets |

## CLI Examples

### Geth
```bash
geth --externalcl ./engine/ai_engine --jwtsecret jwt.hex
```

### Prysm  
```bash
prysm.sh beacon-chain --execution-endpoint http://localhost:8551 \
  --jwt-secret jwt.hex --ai-quorum auto
```

### Trainer
```bash
python3 ai/trainer.py --peer-id <hex> --steps 1000
```

### DHT Daemon
```bash
python3 dht_daemon.py --listen 0.0.0.0:13337 --solo
```

## Development

### Build Individual Components
```bash
# AI components
docker build -f Dockerfile.ai -t pol-ai .

# Engine
docker build -f Dockerfile.engine -t pol-engine .

# DHT
docker build -f Dockerfile.dht -t pol-dht .

# Prysm
docker build -f Dockerfile.prysm -t pol-prysm .
```

### Manual Testing
```bash
# Test DHT connectivity
curl http://localhost:13337/dht/peers

# Check Geth sync
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  http://localhost:8545

# Prysm beacon status  
curl http://localhost:3500/eth/v1/beacon/headers/head
```

## Network Behavior

### Genesis (1 Validator)
- Committee size: 1
- ACK quorum: 1  
- Solo DHT bootstrap
- Immediate finalization

### Multi-Validator (N Validators)
- Committee size: N
- ACK quorum: ceil(N/2)
- DHT mesh network
- Byzantine fault tolerance

### AI Validation
- **Committee members**: Run loss gate in `local_averager.py`
- ✅ **Valid**: Loss improvement → ACK publication → Block acceptance  
- ❌ **Invalid**: No improvement → No ACK → Insufficient quorum → Slashing
- ⏰ **Timeout**: No ACKs within 30s → Empty AI block → Continue

## Security

### Sybil Resistance
- **Committee membership**: Only active validators (32 POL stake) selected for specific slot
- **ACK authenticity**: libp2p ED25519 signatures from DHT 
- **Vote restriction**: Prysm only counts ACKs from exact committee members (no random peers)
- **Loss gate validation**: Each committee member independently verifies ML improvements

### Slashing Conditions
- **Invalid AI update**: Failed ACK quorum → ProposerSlashing via gRPC to Prysm slasher
- **Proposer equivocation**: Standard double-signing detection
- **Committee malfeasance**: Invalid ML validation by committee members
- **Slashing integration**: Connects to Prysm slasher on port 4003, broadcasts via gossipsub

## Performance

### Throughput
- Block time: 12 seconds (Prysm standard)
- AI validation: ~5-30 seconds per update
- LoRA δW: ≤150KB per update

### Resource Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+ (model loading)
- **Storage**: 100GB+ (chain + model data)
- **GPU**: NVIDIA recommended (training)

## Troubleshooting

### Common Issues

1. **Model download fails**
   ```bash
   # Check internet connectivity and HuggingFace access
   docker-compose logs ai-trainer
   ```

2. **DHT connectivity problems**
   ```bash
   # Verify port 13337 is accessible
   docker-compose logs dht-daemon
   ```

3. **Genesis not starting**
   ```bash
   # Check GENESIS_SEED format (64-char hex)
   echo $GENESIS_SEED
   # Should be exactly 64 hexadecimal characters (0-9, a-f, A-F)
   ```

4. **Prysm won't sync**
   ```bash
   # Verify JWT secret matches between Geth and Prysm
   docker-compose exec geth cat /secrets/jwt.hex
   docker-compose exec prysm-beacon cat /secrets/jwt.hex
   ```

### Log Analysis
```bash
# AI ACK quorum validation logs
docker-compose logs prysm-beacon | grep "ACK quorum"

# ML validation logs (loss gate)  
docker-compose logs ai-averager | grep "Loss gate"

# ACK publication logs
docker-compose logs ai-averager | grep "ACK"

# Training progress
docker-compose logs ai-trainer | grep "Step"
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Test with single-node setup
4. Submit pull request

---

**Note**: This is a proof-of-concept implementation. Production use requires additional security audits, testing, and optimizations. 