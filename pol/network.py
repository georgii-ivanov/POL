import asyncio
import json
import websockets
import time
from typing import Dict, List, Set, Optional, Callable, Any
import logging
from .types import NetworkMessage, MessageType, PeerNode
from .crypto import CryptoManager

logger = logging.getLogger(__name__)

class P2PNetwork:
    def __init__(self, node_id: str, host: str = "0.0.0.0", port: int = 8000):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.peers: Dict[str, PeerNode] = {}
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.server = None
        self.running = False
        
        for msg_type in MessageType:
            self.message_handlers[msg_type] = []
        
        logger.info(f"P2P Network initialized for node {node_id} on {host}:{port}")
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered handler for {message_type}")
    
    async def start(self) -> None:
        self.running = True
        
        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        
        logger.info(f"P2P Network started on {self.host}:{self.port}")
        
        asyncio.create_task(self.peer_discovery_loop())
        asyncio.create_task(self.peer_maintenance_loop())
    
    async def stop(self) -> None:
        self.running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        for connection in self.connections.values():
            await connection.close()
        
        logger.info("P2P Network stopped")
    
    async def handle_connection(self, websocket, path):
        peer_id = None
        try:
            async for raw_message in websocket:
                try:
                    message_data = json.loads(raw_message)
                    message = NetworkMessage(**message_data)
                    
                    if not peer_id:
                        peer_id = message.from_node
                        self.connections[peer_id] = websocket
                        logger.info(f"New peer connected: {peer_id}")
                    
                    await self.handle_message(message)
                    
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON message")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Peer {peer_id} disconnected")
        finally:
            if peer_id and peer_id in self.connections:
                del self.connections[peer_id]
            if peer_id and peer_id in self.peers:
                self.peers[peer_id].last_seen = time.time()
    
    async def handle_message(self, message: NetworkMessage) -> None:
        logger.debug(f"Received message: {message.type} from {message.from_node}")
        
        if message.type == MessageType.PEER_DISCOVERY:
            await self.handle_peer_discovery(message)
        else:
            for handler in self.message_handlers[message.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
    
    async def handle_peer_discovery(self, message: NetworkMessage) -> None:
        peer_data = message.data
        
        if peer_data['node_id'] != self.node_id:
            peer = PeerNode(
                id=peer_data['node_id'],
                address=peer_data['address'],
                port=peer_data['port'],
                public_key=peer_data['public_key'],
                is_authority=peer_data.get('is_authority', False),
                reputation=peer_data.get('reputation', 0.0),
                training_capacity=peer_data.get('training_capacity', 0.0)
            )
            
            self.peers[peer.id] = peer
            logger.info(f"Discovered new peer: {peer.id}")
            
            await self.connect_to_peer(peer)
    
    async def connect_to_peer(self, peer: PeerNode) -> bool:
        if peer.id in self.connections:
            return True
        
        try:
            uri = f"ws://{peer.address}:{peer.port}"
            websocket = await websockets.connect(uri)
            self.connections[peer.id] = websocket
            
            asyncio.create_task(self.maintain_connection(peer.id, websocket))
            
            discovery_message = NetworkMessage(
                type=MessageType.PEER_DISCOVERY,
                from_node=self.node_id,
                data={
                    'node_id': self.node_id,
                    'address': self.host,
                    'port': self.port,
                    'public_key': 'placeholder_public_key',
                    'is_authority': False,
                    'reputation': 1.0,
                    'training_capacity': 1.0
                }
            )
            
            await self.send_message_to_peer(peer.id, discovery_message)
            
            logger.info(f"Connected to peer: {peer.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer.id}: {e}")
            return False
    
    async def maintain_connection(self, peer_id: str, websocket):
        try:
            async for raw_message in websocket:
                try:
                    message_data = json.loads(raw_message)
                    message = NetworkMessage(**message_data)
                    await self.handle_message(message)
                except Exception as e:
                    logger.error(f"Error processing message from {peer_id}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection to peer {peer_id} closed")
        except Exception as e:
            logger.error(f"Error in connection to {peer_id}: {e}")
        finally:
            if peer_id in self.connections:
                del self.connections[peer_id]
    
    async def send_message_to_peer(self, peer_id: str, message: NetworkMessage) -> bool:
        if peer_id not in self.connections:
            logger.warning(f"No connection to peer {peer_id}")
            return False
        
        try:
            message_json = json.dumps({
                'type': message.type.value,
                'from_node': message.from_node,
                'to_node': message.to_node,
                'data': message.data,
                'timestamp': message.timestamp,
                'signature': message.signature
            })
            
            await self.connections[peer_id].send(message_json)
            logger.debug(f"Sent message to {peer_id}: {message.type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {peer_id}: {e}")
            return False
    
    async def broadcast_message(self, message: NetworkMessage) -> int:
        message.from_node = self.node_id
        message.timestamp = time.time()
        
        sent_count = 0
        for peer_id in self.connections.keys():
            if await self.send_message_to_peer(peer_id, message):
                sent_count += 1
        
        logger.info(f"Broadcasted message {message.type} to {sent_count} peers")
        return sent_count
    
    async def send_to_authorities(self, message: NetworkMessage) -> int:
        authority_peers = [
            peer_id for peer_id, peer in self.peers.items()
            if peer.is_authority and peer_id in self.connections
        ]
        
        sent_count = 0
        for peer_id in authority_peers:
            if await self.send_message_to_peer(peer_id, message):
                sent_count += 1
        
        logger.info(f"Sent message {message.type} to {sent_count} authority nodes")
        return sent_count
    
    async def bootstrap(self, boot_nodes: List[str]) -> None:
        for boot_node in boot_nodes:
            try:
                host, port = boot_node.split(':')
                
                boot_peer = PeerNode(
                    id=f"boot_{host}_{port}",
                    address=host,
                    port=int(port),
                    public_key="boot_node_key",
                    is_authority=True,
                    reputation=1.0
                )
                
                await self.connect_to_peer(boot_peer)
                
            except Exception as e:
                logger.error(f"Failed to connect to boot node {boot_node}: {e}")
    
    async def peer_discovery_loop(self) -> None:
        while self.running:
            try:
                discovery_message = NetworkMessage(
                    type=MessageType.PEER_DISCOVERY,
                    from_node=self.node_id,
                    data={
                        'node_id': self.node_id,
                        'address': self.host,
                        'port': self.port,
                        'public_key': 'placeholder_public_key',
                        'is_authority': False,
                        'reputation': 1.0,
                        'training_capacity': 1.0
                    }
                )
                
                await self.broadcast_message(discovery_message)
                await asyncio.sleep(30)  # Discovery every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in peer discovery: {e}")
                await asyncio.sleep(10)
    
    async def peer_maintenance_loop(self) -> None:
        while self.running:
            try:
                current_time = time.time()
                stale_peers = []
                
                for peer_id, peer in self.peers.items():
                    if current_time - peer.last_seen > 300:  # 5 minutes timeout
                        stale_peers.append(peer_id)
                
                for peer_id in stale_peers:
                    if peer_id in self.connections:
                        await self.connections[peer_id].close()
                        del self.connections[peer_id]
                    del self.peers[peer_id]
                    logger.info(f"Removed stale peer: {peer_id}")
                
                await asyncio.sleep(60)  # Maintenance every minute
                
            except Exception as e:
                logger.error(f"Error in peer maintenance: {e}")
                await asyncio.sleep(30)
    
    def get_connected_peers(self) -> List[str]:
        return list(self.connections.keys())
    
    def get_peer_count(self) -> int:
        return len(self.peers)
    
    def get_authority_count(self) -> int:
        return sum(1 for peer in self.peers.values() if peer.is_authority)
    
    def get_network_stats(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'total_peers': self.get_peer_count(),
            'connected_peers': len(self.connections),
            'authority_nodes': self.get_authority_count(),
            'running': self.running
        } 