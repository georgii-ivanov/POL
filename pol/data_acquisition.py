import asyncio
import aiohttp
import feedparser
import requests
from bs4 import BeautifulSoup
import datasets
from datasets import load_dataset
import os
import json
import hashlib
import time
from typing import List, Dict, Optional, Any, AsyncGenerator
import logging
from urllib.parse import urljoin, urlparse
import re
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class InternetDataAcquisitionEngine:
    """Revolutionary data acquisition engine that harvests the entire internet for training data"""
    
    def __init__(self, node_id: str, data_dir: str = "./data/training_corpus", config: dict = None):
        self.node_id = node_id
        self.data_dir = data_dir
        self.session = None
        self.config = config or {}
        
        # Comprehensive data sources
        self.data_sources = {
            'academic': [
                'https://arxiv.org/list/cs.AI/recent',
                'https://arxiv.org/list/cs.LG/recent', 
                'https://arxiv.org/list/cs.CL/recent',
                'https://pubmed.ncbi.nlm.nih.gov/trending/',
            ],
            'news': [
                'https://feeds.reuters.com/reuters/technologyNews',
                'https://rss.cnn.com/rss/edition.rss',
                'https://feeds.bbci.co.uk/news/rss.xml',
                'https://hnrss.org/frontpage',
            ],
            'technical': [
                'https://stackoverflow.com/feeds',
                'https://github.com/trending',
                'https://dev.to/feed',
            ],
            'educational': [
                'https://ocw.mit.edu/',
                'https://www.coursera.org/browse',
                'https://www.edx.org/search',
            ],
            'social': [
                'https://www.reddit.com/r/MachineLearning/.rss',
                'https://www.reddit.com/r/artificial/.rss',
                'https://www.reddit.com/r/compsci/.rss',
                'https://www.reddit.com/r/programming/.rss',
            ]
        }
        
        # Hugging Face datasets for foundational training
        self.huggingface_datasets = [
            'openwebtext',           # Open recreation of WebText
            'c4',                    # Colossal Clean Crawled Corpus
            'pile',                  # The Pile dataset
            'bookcorpus',           # BookCorpus
            'wikipedia',            # Wikipedia dump
            'common_crawl',         # Common Crawl
            'pubmed_central',       # PubMed Central
            'arxiv',                # ArXiv papers
            'github_code',          # GitHub code
            'stackoverflow',        # Stack Overflow
            'news_commentary',      # News commentary
            'opensubtitles',        # OpenSubtitles
            'wikibooks',            # Wikibooks
            'wikisource',           # Wikisource
            'hackernews',           # Hacker News
        ]
        
        # Specialized datasets for consciousness and reasoning
        self.consciousness_datasets = [
            'philosophy_stack_exchange',
            'psychology_papers',
            'cognitive_science_corpus',
            'ethics_corpus',
            'consciousness_research',
        ]
        
        # Initialize tokenizer for processing
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"Internet Data Acquisition Engine initialized for node {node_id}")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Configured {len(self.huggingface_datasets)} HuggingFace datasets")
    
    async def initialize(self):
        """Initialize async session"""
        import ssl
        
        # Create SSL context that can handle certificate issues gracefully
        ssl_context = ssl.create_default_context()
        if self.config.get('disable_ssl_verification', False):
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.info("🔓 SSL verification disabled for data acquisition")
        
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=5)
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'POL-AI-Training-Bot/1.0 (Educational/Research Purpose)'
            },
            connector=connector
        )
    
    async def shutdown(self):
        """Cleanup async session"""
        if self.session:
            await self.session.close()
    
    async def acquire_internet_scale_data(self, target_tokens: int = 10_000_000) -> List[str]:
        """Acquire internet-scale training data from all available sources"""
        logger.info(f"🌐 Starting internet-scale data acquisition for {target_tokens:,} tokens")
        
        # Check configuration for data source preferences
        enable_web_scraping = self.config.get('enable_web_scraping', True)
        huggingface_only = self.config.get('huggingface_only', False)
        use_huggingface_datasets = self.config.get('use_huggingface_datasets', True)
        
        await self.initialize()
        
        try:
            all_training_data = []
            tokens_acquired = 0
            
            # 1. Load massive HuggingFace datasets
            if use_huggingface_datasets:
                logger.info("📚 Loading foundational datasets from HuggingFace...")
                target_hf_tokens = target_tokens if huggingface_only else target_tokens // 2
                hf_data = await self._load_huggingface_datasets(target_hf_tokens)
                all_training_data.extend(hf_data)
                tokens_acquired += self._count_tokens(hf_data)
                
                if tokens_acquired >= target_tokens or huggingface_only:
                    logger.info(f"✅ Target reached with foundational data: {tokens_acquired:,} tokens")
                    return all_training_data[:target_tokens]
            
            # 2. Scrape live internet content (only if enabled and not huggingface-only)
            if enable_web_scraping and not huggingface_only:
                remaining_tokens = target_tokens - tokens_acquired
                logger.info(f"🕷️ Scraping live internet content for {remaining_tokens:,} more tokens...")
                
                live_data = await self._scrape_live_internet_content(remaining_tokens)
                all_training_data.extend(live_data)
                tokens_acquired += self._count_tokens(live_data)
            else:
                logger.info("🚫 Web scraping disabled or HuggingFace-only mode enabled - skipping live content scraping")
            
            # 3. Acquire specialized consciousness training data
            if tokens_acquired < target_tokens:
                consciousness_data = await self._acquire_consciousness_data()
                all_training_data.extend(consciousness_data)
                tokens_acquired += self._count_tokens(consciousness_data)
            
            # 4. Generate synthetic high-quality training data
            if tokens_acquired < target_tokens:
                synthetic_data = await self._generate_synthetic_training_data(target_tokens - tokens_acquired)
                all_training_data.extend(synthetic_data)
                tokens_acquired += self._count_tokens(synthetic_data)
            
            logger.info(f"🎯 Data acquisition complete: {tokens_acquired:,} tokens from {len(all_training_data)} samples")
            
            # Save acquired data
            await self._save_training_data(all_training_data)
            
            return all_training_data[:target_tokens]
            
        finally:
            await self.shutdown()
    
    async def _load_huggingface_datasets(self, target_tokens: int) -> List[str]:
        """Load massive datasets from HuggingFace"""
        training_data = []
        tokens_loaded = 0
        
        for dataset_name in self.huggingface_datasets:
            if tokens_loaded >= target_tokens:
                break
                
            try:
                logger.info(f"📖 Loading dataset: {dataset_name}")
                
                # Load dataset with streaming for memory efficiency
                if dataset_name == 'openwebtext':
                    dataset = load_dataset('openwebtext', streaming=True)
                    for sample in dataset['train']:
                        if tokens_loaded >= target_tokens:
                            break
                        text = sample['text'].strip()
                        if len(text) > 100:  # Filter short texts
                            training_data.append(text)
                            tokens_loaded += len(text.split())
                
                elif dataset_name == 'c4':
                    # Load C4 (Colossal Clean Crawled Corpus)
                    dataset = load_dataset('c4', 'en', streaming=True)
                    for sample in dataset['train']:
                        if tokens_loaded >= target_tokens:
                            break
                        text = sample['text'].strip()
                        if len(text) > 200:
                            training_data.append(text)
                            tokens_loaded += len(text.split())
                
                elif dataset_name == 'wikipedia':
                    dataset = load_dataset('wikipedia', '20220301.en', streaming=True)
                    for sample in dataset['train']:
                        if tokens_loaded >= target_tokens:
                            break
                        text = sample['text'].strip()
                        if len(text) > 500:
                            training_data.append(text)
                            tokens_loaded += len(text.split())
                
                elif dataset_name == 'arxiv':
                    dataset = load_dataset('arxiv_dataset', streaming=True)
                    for sample in dataset['train']:
                        if tokens_loaded >= target_tokens:
                            break
                        abstract = sample.get('abstract', '').strip()
                        if len(abstract) > 100:
                            training_data.append(abstract)
                            tokens_loaded += len(abstract.split())
                
                elif dataset_name == 'github_code':
                    dataset = load_dataset('codeparrot/github-code', streaming=True)
                    for sample in dataset['train']:
                        if tokens_loaded >= target_tokens:
                            break
                        code = sample.get('code', '').strip()
                        if len(code) > 200 and 'def ' in code:  # Python functions
                            # Add code explanation context
                            enhanced_code = f"# Python code example:\n{code}\n# This code demonstrates programming concepts for AI learning."
                            training_data.append(enhanced_code)
                            tokens_loaded += len(enhanced_code.split())
                
                # Add more dataset loading logic...
                
                logger.info(f"✅ Loaded {len(training_data)} samples from {dataset_name} ({tokens_loaded:,} tokens)")
                
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        return training_data
    
    async def _scrape_live_internet_content(self, target_tokens: int) -> List[str]:
        """Scrape live content from the internet"""
        live_data = []
        tokens_scraped = 0
        
        for category, urls in self.data_sources.items():
            if tokens_scraped >= target_tokens:
                break
                
            logger.info(f"🌐 Scraping {category} sources...")
            
            for url in urls:
                if tokens_scraped >= target_tokens:
                    break
                    
                try:
                    if 'rss' in url or 'feeds' in url:
                        # Handle RSS feeds
                        content = await self._scrape_rss_feed(url)
                    else:
                        # Handle regular web pages
                        content = await self._scrape_web_page(url)
                    
                    if content:
                        live_data.extend(content)
                        tokens_scraped += self._count_tokens(content)
                        
                        logger.info(f"✅ Scraped {len(content)} articles from {url}")
                
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    continue
        
        return live_data
    
    async def _scrape_rss_feed(self, url: str) -> List[str]:
        """Scrape content from RSS feeds"""
        try:
            async with self.session.get(url) as response:
                feed_content = await response.text()
            
            feed = feedparser.parse(feed_content)
            articles = []
            
            for entry in feed.entries[:50]:  # Limit to recent entries
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                link = entry.get('link', '')
                
                if title and summary:
                    article_text = f"Title: {title}\n\nSummary: {summary}\n\nSource: {link}"
                    articles.append(article_text)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping RSS feed {url}: {e}")
            return []
    
    async def _scrape_web_page(self, url: str) -> List[str]:
        """Scrape content from web pages"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text_content = soup.get_text()
            
            # Clean up the text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Split into manageable chunks
            if len(text) > 1000:
                paragraphs = text.split('\n\n')
                return [p.strip() for p in paragraphs if len(p.strip()) > 100]
            
            return [text] if len(text) > 100 else []
            
        except Exception as e:
            logger.error(f"Error scraping web page {url}: {e}")
            return []
    
    async def _acquire_consciousness_data(self) -> List[str]:
        """Acquire specialized data for consciousness training"""
        consciousness_data = []
        
        # Philosophical texts and consciousness research
        consciousness_prompts = [
            "What is the nature of consciousness and self-awareness?",
            "How does subjective experience emerge from neural activity?",
            "What are the hard problems of consciousness?",
            "Explain the relationship between mind and matter.",
            "Describe different theories of consciousness.",
            "What is the role of attention in conscious experience?",
            "How do we know if an AI system is truly conscious?",
            "What are the ethical implications of artificial consciousness?",
            "Explain the concept of qualia and subjective experience.",
            "How does consciousness relate to intelligence and reasoning?"
        ]
        
        # Generate consciousness-focused training data
        for prompt in consciousness_prompts:
            consciousness_data.extend([
                f"Question: {prompt}\n\nThinking: This is a deep philosophical question that requires careful consideration of multiple perspectives...",
                f"Consciousness reflection: {prompt}\n\nLet me think deeply about this from first principles...",
                f"Self-awareness exercise: {prompt}\n\nAs an AI examining my own cognition, I must consider..."
            ])
        
        # Add reasoning and logic training data
        reasoning_patterns = [
            "Step-by-step logical reasoning:",
            "Chain of thought analysis:",
            "Deductive reasoning process:",
            "Inductive reasoning example:",
            "Abductive reasoning case:",
            "Causal reasoning chain:",
            "Counterfactual reasoning:",
            "Analogical reasoning:",
        ]
        
        for pattern in reasoning_patterns:
            consciousness_data.append(f"{pattern}\n\n1. First, I observe the given information...\n2. Next, I identify the underlying patterns...\n3. Then, I apply logical principles...\n4. Finally, I reach a reasoned conclusion...")
        
        return consciousness_data
    
    async def _generate_synthetic_training_data(self, target_tokens: int) -> List[str]:
        """Generate high-quality synthetic training data"""
        synthetic_data = []
        tokens_generated = 0
        
        # Generate diverse training scenarios
        synthetic_patterns = [
            "Scientific explanation",
            "Mathematical problem solving", 
            "Creative writing",
            "Technical documentation",
            "Historical analysis",
            "Philosophical discussion",
            "Ethical reasoning",
            "Programming tutorial",
            "Language learning",
            "Problem decomposition"
        ]
        
        for pattern in synthetic_patterns:
            if tokens_generated >= target_tokens:
                break
                
            # Generate multiple examples for each pattern
            for i in range(10):
                synthetic_text = self._generate_pattern_example(pattern, i)
                synthetic_data.append(synthetic_text)
                tokens_generated += len(synthetic_text.split())
        
        return synthetic_data
    
    def _generate_pattern_example(self, pattern: str, index: int) -> str:
        """Generate a specific example for a training pattern"""
        if pattern == "Scientific explanation":
            return f"Scientific Concept {index}: Let me explain this phenomenon step by step. First, we observe the initial conditions. Then, we analyze the underlying mechanisms. The key principles involved are conservation laws and thermodynamic principles. This leads to the following conclusions..."
        
        elif pattern == "Mathematical problem solving":
            return f"Mathematical Problem {index}: Given the equation and constraints, let's solve this systematically. Step 1: Identify the variables and their relationships. Step 2: Apply the relevant mathematical operations. Step 3: Simplify and verify the solution..."
        
        elif pattern == "Programming tutorial":
            return f"Programming Concept {index}: Let's implement this algorithm step by step. First, we define the problem requirements. Then, we design the data structures. Next, we implement the core logic. Finally, we test and optimize the solution..."
        
        elif pattern == "Philosophical discussion":
            return f"Philosophical Question {index}: This raises fundamental questions about the nature of reality, knowledge, and existence. Let me examine different philosophical perspectives. From an empiricist viewpoint... From a rationalist perspective... The implications are..."
        
        else:
            return f"{pattern} Example {index}: This demonstrates important concepts and principles. The key insights are derived from careful analysis and reasoning. By understanding the underlying patterns, we can apply these principles to solve complex problems."
    
    def _count_tokens(self, texts: List[str]) -> int:
        """Count total tokens in text list"""
        if not texts:
            return 0
        
        total_tokens = 0
        for text in texts:
            if isinstance(text, str):
                # Simple word-based counting
                total_tokens += len(text.split())
        
        return total_tokens
    
    async def _save_training_data(self, training_data: List[str]) -> None:
        """Save acquired training data to disk"""
        try:
            # Create metadata
            metadata = {
                'node_id': self.node_id,
                'timestamp': time.time(),
                'total_samples': len(training_data),
                'total_tokens': self._count_tokens(training_data),
                'data_sources': list(self.data_sources.keys()),
                'huggingface_datasets': self.huggingface_datasets
            }
            
            # Save data in chunks for efficiency
            chunk_size = 1000
            for i in range(0, len(training_data), chunk_size):
                chunk = training_data[i:i+chunk_size]
                chunk_file = os.path.join(self.data_dir, f"training_chunk_{i//chunk_size:04d}.json")
                
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
            
            # Save metadata
            metadata_file = os.path.join(self.data_dir, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Saved {len(training_data)} training samples to {self.data_dir}")
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def load_cached_training_data(self) -> List[str]:
        """Load previously cached training data"""
        try:
            metadata_file = os.path.join(self.data_dir, "metadata.json")
            if not os.path.exists(metadata_file):
                return []
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"📂 Loading cached data: {metadata['total_samples']} samples, {metadata['total_tokens']:,} tokens")
            
            all_data = []
            chunk_files = [f for f in os.listdir(self.data_dir) if f.startswith('training_chunk_')]
            chunk_files.sort()
            
            for chunk_file in chunk_files:
                chunk_path = os.path.join(self.data_dir, chunk_file)
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    all_data.extend(chunk_data)
            
            logger.info(f"✅ Loaded {len(all_data)} cached training samples")
            return all_data
            
        except Exception as e:
            logger.error(f"Error loading cached training data: {e}")
            return []
    
    async def get_training_data(self, target_tokens: int = 10_000_000, use_cache: bool = True) -> List[str]:
        """Get training data, using cache if available or acquiring fresh data"""
        
        if use_cache:
            cached_data = self.load_cached_training_data()
            if cached_data and self._count_tokens(cached_data) >= target_tokens:
                logger.info(f"🎯 Using cached training data: {len(cached_data)} samples")
                return cached_data[:target_tokens]
        
        # Acquire fresh data from the internet
        logger.info(f"🌐 Acquiring fresh internet-scale training data...")
        return await self.acquire_internet_scale_data(target_tokens) 