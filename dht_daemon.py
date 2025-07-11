#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
import logging
from typing import List, Optional
import signal
import time
import json
from dataclasses import dataclass, field
import hivemind
from hivemind import DHT, get_logger

logger = get_logger(__name__)

@dataclass
class DHTConfig:
    listen_port: int = 13337
    listen_host: str = "0.0.0.0"
    k_size: int = 20
    solo_mode: bool = False
    bootstrap_peers: List[str] = field(default_factory=list)
    autonat_enabled: bool = True
    persistence_dir: str = "/dht_db"

class DHTDaemon:
    def __init__(self, config: DHTConfig):
        self.config = config
        self.dht: Optional[DHT] = None
        self.running = False
        
        os.makedirs(config.persistence_dir, exist_ok=True)
        logger.info(f"DHT daemon initializing with config: {config}")
        
    async def start(self):
        logger.info("Starting DHT daemon...")
        
        initial_peers = []
        if not self.config.solo_mode and self.config.bootstrap_peers:
            initial_peers = self.config.bootstrap_peers
            logger.info(f"Connecting to bootstrap peers: {initial_peers}")
        elif self.config.solo_mode:
            logger.info("Starting in solo mode (genesis node)")
        else:
            genesis_host = os.getenv("GENESIS_HOST")
            if genesis_host and genesis_host != "localhost":
                initial_peers = [f"/ip4/{genesis_host}/tcp/13337"]
                logger.info(f"Auto-discovered genesis host: {genesis_host}")
        
        host_maddrs = [f"/ip4/{self.config.listen_host}/tcp/{self.config.listen_port}"]
        if self.config.listen_host == "0.0.0.0":
            host_maddrs.append(f"/ip4/127.0.0.1/tcp/{self.config.listen_port}")
            
        try:
            self.dht = DHT(
                start=True,
                initial_peers=initial_peers,
                host_maddrs=host_maddrs,
                client_mode=False
            )
            
            self.running = True
            logger.info(f"DHT daemon started successfully")
            if self.dht:
                logger.info(f"Node ID: {self.dht.peer_id}")
            logger.info(f"Listening on: {host_maddrs}")
            
            await self._periodic_maintenance()
            
        except Exception as e:
            logger.error(f"Failed to start DHT daemon: {e}")
            raise
            
    async def _periodic_maintenance(self):
        last_stats_time = time.time()
        
        while self.running:
            try:
                await asyncio.sleep(60)
                
                current_time = time.time()
                if current_time - last_stats_time >= 300:
                    await self._log_stats()
                    last_stats_time = current_time
                    
                await self._cleanup_expired_records()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance error: {e}")
                
    async def _log_stats(self):
        if not self.dht:
            return
            
        try:
            routing_table_size = len(self.dht.routing_table)
            logger.info(f"DHT Stats - Routing table size: {routing_table_size}")
            logger.info(f"DHT Stats - Node ID: {self.dht.peer_id}")
            
            if hasattr(self.dht, 'storage'):
                stored_records = len(self.dht.storage)
                logger.info(f"DHT Stats - Stored records: {stored_records}")
                
        except Exception as e:
            logger.warning(f"Failed to log stats: {e}")
            
    async def _cleanup_expired_records(self):
        if not self.dht or not hasattr(self.dht, 'storage'):
            return
            
        try:
            current_time = hivemind.get_dht_time()
            expired_keys = []
            
            for key, record in self.dht.storage.items():
                if hasattr(record, 'expiration_time') and record.expiration_time < current_time:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                del self.dht.storage[key]
                
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired records")
                
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
            
    async def store_data(self, key: str, value: bytes, ttl: int = 3600) -> bool:
        if not self.dht:
            return False
            
        try:
            success = await self.dht.store(
                key=key,
                value=value,
                expiration_time=hivemind.get_dht_time() + ttl
            )
            
            if success:
                logger.debug(f"Stored data for key: {key}")
            else:
                logger.warning(f"Failed to store data for key: {key}")
                
            return success
            
        except Exception as e:
            logger.error(f"Store error for key {key}: {e}")
            return False
            
    async def get_data(self, key: str) -> Optional[bytes]:
        if not self.dht:
            return None
            
        try:
            result = await self.dht.get(key)
            
            if result and result.value:
                logger.debug(f"Retrieved data for key: {key}")
                return result.value
            else:
                logger.debug(f"No data found for key: {key}")
                return None
                
        except Exception as e:
            logger.error(f"Get error for key {key}: {e}")
            return None
            
    async def list_peers(self) -> List[str]:
        if not self.dht:
            return []
            
        try:
            peers = []
            for peer_id in self.dht.routing_table:
                peers.append(str(peer_id))
            return peers
        except Exception as e:
            logger.error(f"Error listing peers: {e}")
            return []
            
    async def stop(self):
        logger.info("Stopping DHT daemon...")
        self.running = False
        
        if self.dht:
            try:
                await self.dht.shutdown()
                logger.info("DHT daemon stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping DHT: {e}")

class DHTDaemonServer:
    def __init__(self, config: DHTConfig):
        self.config = config
        self.daemon = DHTDaemon(config)
        self.shutdown_event = asyncio.Event()
        
    async def run(self):
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            await self.daemon.start()
            await self.shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"DHT daemon error: {e}")
        finally:
            await self.daemon.stop()
            
    async def shutdown(self):
        self.shutdown_event.set()

def main():
    parser = argparse.ArgumentParser(description="POL-AI DHT Daemon")
    parser.add_argument("--listen", type=str, default="0.0.0.0:13337", 
                       help="Listen address and port")
    parser.add_argument("--solo", action="store_true", 
                       help="Start in solo mode (genesis node)")
    parser.add_argument("--bootstrap", type=str, nargs="+", 
                       help="Bootstrap peer addresses")
    parser.add_argument("--k-size", type=int, default=20, 
                       help="Kademlia k parameter")
    parser.add_argument("--persistence-dir", type=str, default="/dht_db", 
                       help="Directory for persistent data")
    parser.add_argument("--no-autonat", action="store_true", 
                       help="Disable AutoNAT relay")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if ":" in args.listen:
        listen_host, listen_port = args.listen.rsplit(":", 1)
        listen_port = int(listen_port)
    else:
        listen_host = "0.0.0.0"
        listen_port = int(args.listen)
        
    config = DHTConfig(
        listen_host=listen_host,
        listen_port=listen_port,
        k_size=args.k_size,
        solo_mode=args.solo,
        bootstrap_peers=args.bootstrap or [],
        autonat_enabled=not args.no_autonat,
        persistence_dir=args.persistence_dir
    )
    
    server = DHTDaemonServer(config)
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("DHT daemon terminated by user")
    except Exception as e:
        logger.error(f"DHT daemon failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 