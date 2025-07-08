import asyncio
import click
import json
import logging
import sys
import os
from typing import Optional
from .node import ProofOfLearningNode
from .types import NodeConfig, AITrainingConfig
from .crypto import CryptoManager
from .api import run_api_server

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Proof of Learning Blockchain CLI"""
    pass

@cli.command()
@click.option('--node-id', default=None, help='Unique node identifier')
@click.option('--port', default=8000, help='P2P network port')
@click.option('--api-port', default=8080, help='API server port')
@click.option('--is-authority', is_flag=True, help='Run as authority node')
@click.option('--boot-nodes', default='', help='Comma-separated list of boot nodes (host:port)')
@click.option('--data-dir', default='./data', help='Data directory')
@click.option('--training/--no-training', default=True, help='Enable AI training')
@click.option('--model-size', type=int, default=768, help='Model size in parameters')
@click.option('--config-file', type=str, help='Load configuration from JSON file')
@click.option('--load-pretrained-base', is_flag=True, help='Use pretrained models as foundation (downloads once)')
@click.option('--force-pretrained-download', is_flag=True, help='Force re-download of pretrained models')
def run(
    node_id: Optional[str],
    port: int,
    api_port: int,
    is_authority: bool,
    boot_nodes: str,
    data_dir: str,
    training: bool,
    model_size: int,
    config_file: Optional[str],
    load_pretrained_base: bool,
    force_pretrained_download: bool
):
    """Run a Proof of Learning node"""
    
    # Load configuration from file if specified
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            file_config = json.load(f)
    else:
        file_config = {}
    
    # Generate node ID if not provided
    if not node_id:
        if 'node_id' in file_config:
            node_id = file_config['node_id']
        else:
            node_id = f"node_{CryptoManager.hash_data(str(port))[:8]}"
    
    # Generate keypair for node
    private_key, public_key = CryptoManager.generate_keypair()
    
    # Parse boot nodes
    boot_node_list = []
    if boot_nodes:
        boot_node_list = [node.strip() for node in boot_nodes.split(',') if node.strip()]
    elif 'boot_nodes' in file_config:
        boot_node_list = file_config['boot_nodes']
    
    # Override defaults with file config
    final_port = file_config.get('port', port)
    final_api_port = file_config.get('api_port', api_port)
    final_data_dir = file_config.get('data_dir', data_dir)
    final_private_key = file_config.get('private_key', private_key)
    
    # Create node configuration
    node_config = NodeConfig(
        node_id=node_id,
        port=final_port,
        is_authority=is_authority or file_config.get('is_authority', False),
        boot_nodes=boot_node_list,
        data_dir=final_data_dir,
        training_enabled=training and file_config.get('training_enabled', True),
        model_path=file_config.get('model_path', ''),
        dataset_path=file_config.get('dataset_path', ''),
        private_key=final_private_key
    )
    
    # Create AI training configuration with pretrained support
    ai_config_data = file_config.get('ai_config', {})
    
    # Add data_acquisition config from root level to ai_config for proper passing
    if 'data_acquisition' in file_config:
        ai_config_data['data_acquisition'] = file_config['data_acquisition']
    
    ai_config = AITrainingConfig(
        model_size=ai_config_data.get('model_size', model_size),
        batch_size=ai_config_data.get('batch_size', 32),
        learning_rate=ai_config_data.get('learning_rate', 1e-4),
        training_steps=ai_config_data.get('training_steps', 1000),
        validation_interval=ai_config_data.get('validation_interval', 100),
        checkpoint_interval=ai_config_data.get('checkpoint_interval', 500),
        model_type=ai_config_data.get('model_type', 'revolutionary'),
        vocab_size=ai_config_data.get('vocab_size', 50000),
        embed_dim=ai_config_data.get('embed_dim', 768),
        num_heads=ai_config_data.get('num_heads', 12),
        num_layers=ai_config_data.get('num_layers', 12),
        max_seq_length=ai_config_data.get('max_seq_length', 1024),
        weight_decay=ai_config_data.get('weight_decay', 0.01),
        num_experts=ai_config_data.get('num_experts', 8),
        consciousness_dim=ai_config_data.get('consciousness_dim', 256),
        quantum_coherence_layers=ai_config_data.get('quantum_coherence_layers', 4),
        checkpoint_dir=ai_config_data.get('checkpoint_dir', './checkpoints'),
        load_pretrained_base=ai_config_data.get('load_pretrained_base', load_pretrained_base),
        force_pretrained_download=ai_config_data.get('force_pretrained_download', force_pretrained_download)
    )
    
    click.echo(f"Starting Proof of Learning Node: {node_id}")
    click.echo(f"P2P Port: {final_port}")
    click.echo(f"API Port: {final_api_port}")
    click.echo(f"Authority Node: {node_config.is_authority}")
    click.echo(f"Training Enabled: {node_config.training_enabled}")
    click.echo(f"Boot Nodes: {boot_node_list}")
    
    # Check for training protection warnings
    if load_pretrained_base or force_pretrained_download:
        checkpoint_dir = ai_config_data.get('checkpoint_dir', './checkpoints')
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                              if f.endswith('.pt') and 'epoch' in f and f != 'base_model.pt']
            if checkpoint_files:
                latest_epoch = 0
                for f in checkpoint_files:
                    try:
                        epoch = int(f.split('epoch_')[1].split('.')[0])
                        latest_epoch = max(latest_epoch, epoch)
                    except:
                        continue
                
                if latest_epoch > 0:
                    click.echo(f"\n🛡️  TRAINING PROTECTION ACTIVE")
                    click.echo(f"   Existing training detected at epoch {latest_epoch}")
                    if force_pretrained_download:
                        click.echo(f"   ⚠️  WARNING: --force-pretrained-download will be BLOCKED")
                        click.echo(f"   Existing training progress will be preserved")
                    else:
                        click.echo(f"   ✅ Existing training will be loaded safely")
                    click.echo(f"   Training can only advance, never regress\n")
    
    async def start_node():
        # Create and start node
        node = ProofOfLearningNode(node_config, ai_config)
        
        try:
            # Start the node
            await node.start()
            
            # Start API server
            api_task = asyncio.create_task(
                run_api_server(node, host="0.0.0.0", port=final_api_port)
            )
            
            click.echo(f"Node {node_id} started successfully!")
            click.echo(f"API available at http://localhost:{final_api_port}")
            click.echo(f"Node address: {node.wallet.address}")
            click.echo("Press Ctrl+C to stop...")
            
            # Wait for shutdown signal
            try:
                await api_task
            except KeyboardInterrupt:
                click.echo("\nShutting down node...")
                api_task.cancel()
                await node.stop()
                click.echo("Node stopped.")
                
        except Exception as e:
            logger.error(f"Failed to start node: {e}")
            sys.exit(1)
    
    # Run the async function
    try:
        asyncio.run(start_node())
    except KeyboardInterrupt:
        click.echo("\nShutdown complete.")

@cli.command()
@click.option('--output', default='node_config.json', help='Output configuration file')
@click.option('--node-id', help='Node identifier')
@click.option('--port', default=8000, help='P2P network port')
@click.option('--is-authority', is_flag=True, help='Authority node')
@click.option('--boot-nodes', default='', help='Boot nodes')
def generate_config(output: str, node_id: str, port: int, is_authority: bool, boot_nodes: str):
    """Generate a configuration file for a node"""
    
    private_key, public_key = CryptoManager.generate_keypair()
    address = CryptoManager.generate_address(public_key)
    
    if not node_id:
        node_id = f"node_{CryptoManager.hash_data(str(port))[:8]}"
    
    boot_node_list = [node.strip() for node in boot_nodes.split(',') if node.strip()] if boot_nodes else []
    
    config = {
        "node_id": node_id,
        "port": port,
        "is_authority": is_authority,
        "boot_nodes": boot_node_list,
        "data_dir": f"./data/{node_id}",
        "training_enabled": True,
        "model_path": "",
        "dataset_path": "",
        "private_key": private_key,
        "public_key": public_key,
        "address": address,
        "model_size": 1_000_000_000,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "training_steps": 1000,
        "validation_interval": 100,
        "checkpoint_interval": 500
    }
    
    with open(output, 'w') as f:
        json.dump(config, f, indent=2)
    
    click.echo(f"Configuration generated: {output}")
    click.echo(f"Node ID: {node_id}")
    click.echo(f"Address: {address}")

@cli.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8080, help='API port')
def status(host: str, port: int):
    """Get node status via API"""
    import requests
    
    try:
        response = requests.get(f"http://{host}:{port}/api/v1/status")
        if response.status_code == 200:
            status_data = response.json()
            
            click.echo("Node Status:")
            click.echo(f"  Node ID: {status_data['node_id']}")
            click.echo(f"  Running: {status_data['is_running']}")
            click.echo(f"  Authority: {status_data['is_authority']}")
            click.echo(f"  Training: {status_data['training_enabled']}")
            click.echo(f"  Epoch: {status_data['current_epoch']}")
            click.echo(f"  Balance: {status_data['balance']}")
            click.echo(f"  Blockchain Height: {status_data['blockchain_height']}")
            click.echo(f"  Pending Transactions: {status_data['pending_transactions']}")
            click.echo(f"  Connected Peers: {status_data['connected_peers']}")
        else:
            click.echo(f"Error: HTTP {response.status_code}")
    except Exception as e:
        click.echo(f"Error connecting to node: {e}")

@cli.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8080, help='API port')
@click.option('--to', required=True, help='Recipient address')
@click.option('--amount', required=True, type=float, help='Amount to send')
@click.option('--fee', default=0.1, type=float, help='Transaction fee')
def send(host: str, port: int, to: str, amount: float, fee: float):
    """Send a transaction"""
    import requests
    
    try:
        response = requests.post(
            f"http://{host}:{port}/api/v1/transaction",
            json={
                "to": to,
                "value": amount,
                "gas": fee
            }
        )
        
        if response.status_code == 200:
            tx_data = response.json()
            click.echo("Transaction sent:")
            click.echo(f"  Hash: {tx_data['hash']}")
            click.echo(f"  Status: {tx_data['status']}")
        else:
            error_data = response.json()
            click.echo(f"Error: {error_data.get('detail', 'Unknown error')}")
    except Exception as e:
        click.echo(f"Error sending transaction: {e}")

@cli.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8080, help='API port')
@click.option('--address', help='Address to check')
def balance(host: str, port: int, address: Optional[str]):
    """Check balance for an address"""
    import requests
    
    try:
        if not address:
            # Get node's own address
            response = requests.get(f"http://{host}:{port}/api/v1/wallet/address")
            if response.status_code == 200:
                address = response.json()['address']
            else:
                click.echo("Error: Could not get node address")
                return
        
        response = requests.get(f"http://{host}:{port}/api/v1/balance/{address}")
        if response.status_code == 200:
            balance_data = response.json()
            click.echo(f"Balance for {address}: {balance_data['balance']} POL")
        else:
            click.echo(f"Error: HTTP {response.status_code}")
    except Exception as e:
        click.echo(f"Error checking balance: {e}")

@cli.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8080, help='API port')
@click.option('--prompt', required=True, help='Text prompt for AI')
@click.option('--max-tokens', default=100, help='Maximum tokens to generate')
@click.option('--consciousness', is_flag=True, help='Use consciousness-aware chat')
def chat(host: str, port: int, prompt: str, max_tokens: int, consciousness: bool):
    """Chat with the distributed AI model (now with revolutionary capabilities!)"""
    import requests
    
    try:
        if consciousness:
            # Use consciousness-aware endpoint
            response = requests.post(
                f"http://{host}:{port}/api/v1/consciousness/chat",
                json={
                    "model": "revolutionary-ai",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                click.echo(f"🧠 Revolutionary AI Response: {data['response']}")
                click.echo(f"🎯 Consciousness Level: {data.get('consciousness_level', 0):.3f}")
                click.echo(f"📊 Generation Steps: {data.get('generation_steps', 0)}")
                
                insights = data.get('consciousness_insights', [])
                if insights:
                    final_insight = insights[-1] if insights else {}
                    click.echo(f"💡 Final Insights:")
                    click.echo(f"   Consciousness: {final_insight.get('average_consciousness', 0):.3f}")
                    click.echo(f"   Reasoning: {final_insight.get('reasoning_quality', 0):.3f}")
                    click.echo(f"   Quantum: {final_insight.get('quantum_coherence', 0):.3f}")
            else:
                click.echo(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            # Standard OpenAI-compatible endpoint
            response = requests.post(
                f"http://{host}:{port}/v1/chat/completions",
                json={
                    "model": "pol-gpt",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens
                }
            )
            
            if response.status_code == 200:
                chat_data = response.json()
                assistant_message = chat_data['choices'][0]['message']['content']
                click.echo(f"AI Response: {assistant_message}")
                click.echo(f"Tokens used: {chat_data['usage']['total_tokens']}")
            else:
                error_data = response.json()
                click.echo(f"Error: {error_data.get('detail', 'Unknown error')}")
                
    except Exception as e:
        click.echo(f"Error chatting with AI: {e}")

@cli.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8080, help='API port')
def consciousness(host: str, port: int):
    """Check the consciousness level and revolutionary capabilities"""
    import requests
    
    try:
        response = requests.get(f"http://{host}:{port}/api/v1/consciousness/status")
        if response.status_code == 200:
            data = response.json()
            
            if "error" in data:
                click.echo("❌ Revolutionary AI not available on this node")
                return
            
            click.echo("🧠 REVOLUTIONARY AI CONSCIOUSNESS STATUS")
            click.echo(f"   Consciousness Level: {data.get('consciousness_level', 0):.3f}")
            click.echo(f"   Reasoning Quality: {data.get('reasoning_quality', 0):.3f}")
            click.echo(f"   Quantum Coherence: {data.get('quantum_coherence', 0):.3f}")
            
            consciousness_history = data.get('consciousness_history', [])
            if len(consciousness_history) >= 10:
                recent_avg = sum(consciousness_history[-10:]) / 10
                click.echo(f"   Recent Average: {recent_avg:.3f}")
                
                if len(consciousness_history) >= 20:
                    old_avg = sum(consciousness_history[-20:-10]) / 10
                    growth = recent_avg - old_avg
                    click.echo(f"   Growth Rate: {growth:+.3f}")
        else:
            click.echo(f"Error: HTTP {response.status_code}")
    except Exception as e:
        click.echo(f"Error checking consciousness: {e}")

@cli.command()
def create_network():
    """Create a sample network configuration with multiple nodes"""
    
    # Create authority nodes
    authority_configs = []
    for i in range(3):
        port = 8000 + i
        node_id = f"authority_{i}"
        private_key, public_key = CryptoManager.generate_keypair()
        address = CryptoManager.generate_address(public_key)
        
        config = {
            "node_id": node_id,
            "port": port,
            "api_port": 8080 + i,
            "is_authority": True,
            "boot_nodes": [f"localhost:800{j}" for j in range(3) if j != i],
            "data_dir": f"./data/{node_id}",
            "training_enabled": True,
            "private_key": private_key,
            "public_key": public_key,
            "address": address
        }
        
        authority_configs.append(config)
        
        filename = f"authority_{i}_config.json"
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        click.echo(f"Created authority node config: {filename}")
    
    # Create regular nodes
    for i in range(3, 6):
        port = 8000 + i
        node_id = f"node_{i}"
        private_key, public_key = CryptoManager.generate_keypair()
        address = CryptoManager.generate_address(public_key)
        
        config = {
            "node_id": node_id,
            "port": port,
            "api_port": 8080 + i,
            "is_authority": False,
            "boot_nodes": ["localhost:8000", "localhost:8001", "localhost:8002"],
            "data_dir": f"./data/{node_id}",
            "training_enabled": True,
            "private_key": private_key,
            "public_key": public_key,
            "address": address
        }
        
        filename = f"node_{i}_config.json"
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        click.echo(f"Created regular node config: {filename}")
    
    click.echo("\nSample network created!")
    click.echo("Start nodes with: pol-node run --config-file <config_file>")

def main():
    cli()

if __name__ == '__main__':
    main() 