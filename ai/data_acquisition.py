#!/usr/bin/env python3

import os
import random
from typing import List, Dict, Optional, Union, Iterator
from datasets import load_dataset, Dataset, concatenate_datasets
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    name: str
    path: str
    subset: Optional[str] = None
    split: str = "train"
    streaming: bool = False
    max_samples: Optional[int] = None
    text_column: str = "text"
    description: str = ""

class DataAcquisition:
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.available_datasets = {
            "wikipedia_en": DatasetConfig(
                name="Wikipedia English",
                path="wikimedia/wikipedia",
                subset="20231101.en",
                text_column="text",
                description="Wikipedia articles in English - high quality encyclopedic content"
            ),
            "wikipedia_simple": DatasetConfig(
                name="Wikipedia Simple English",
                path="wikimedia/wikipedia", 
                subset="20231101.simple",
                text_column="text",
                description="Simple English Wikipedia - easier to understand content"
            ),
            "wikipedia_de": DatasetConfig(
                name="Wikipedia German",
                path="wikimedia/wikipedia",
                subset="20231101.de",
                text_column="text",
                description="Wikipedia articles in German"
            ),
            "wikipedia_fr": DatasetConfig(
                name="Wikipedia French",
                path="wikimedia/wikipedia",
                subset="20231101.fr",
                text_column="text",
                description="Wikipedia articles in French"
            ),
            "wikipedia_es": DatasetConfig(
                name="Wikipedia Spanish",
                path="wikimedia/wikipedia",
                subset="20231101.es",
                text_column="text",
                description="Wikipedia articles in Spanish"
            ),
            "wikipedia_it": DatasetConfig(
                name="Wikipedia Italian",
                path="wikimedia/wikipedia",
                subset="20231101.it",
                text_column="text",
                description="Wikipedia articles in Italian"
            ),
            "wikipedia_pt": DatasetConfig(
                name="Wikipedia Portuguese",
                path="wikimedia/wikipedia",
                subset="20231101.pt",
                text_column="text",
                description="Wikipedia articles in Portuguese"
            ),
            "wikipedia_ru": DatasetConfig(
                name="Wikipedia Russian",
                path="wikimedia/wikipedia",
                subset="20231101.ru",
                text_column="text",
                description="Wikipedia articles in Russian"
            ),
            "wikipedia_zh": DatasetConfig(
                name="Wikipedia Chinese",
                path="wikimedia/wikipedia",
                subset="20231101.zh",
                text_column="text",
                description="Wikipedia articles in Chinese"
            ),
            "wikipedia_ja": DatasetConfig(
                name="Wikipedia Japanese",
                path="wikimedia/wikipedia",
                subset="20231101.ja",
                text_column="text",
                description="Wikipedia articles in Japanese"
            ),
            "wikipedia_ar": DatasetConfig(
                name="Wikipedia Arabic",
                path="wikimedia/wikipedia",
                subset="20231101.ar",
                text_column="text",
                description="Wikipedia articles in Arabic"
            ),
            "c4_en": DatasetConfig(
                name="C4 English",
                path="allenai/c4",
                subset="en",
                text_column="text",
                description="Colossal Clean Crawled Corpus - English web pages"
            ),
            "c4_de": DatasetConfig(
                name="C4 German",
                path="allenai/c4",
                subset="de",
                text_column="text",
                description="C4 dataset in German"
            ),
            "c4_fr": DatasetConfig(
                name="C4 French",
                path="allenai/c4",
                subset="fr",
                text_column="text",
                description="C4 dataset in French"
            ),
            "c4_es": DatasetConfig(
                name="C4 Spanish",
                path="allenai/c4",
                subset="es",
                text_column="text",
                description="C4 dataset in Spanish"
            ),
            "fineweb": DatasetConfig(
                name="FineWeb",
                path="HuggingFaceFW/fineweb",
                subset="sample-10BT",
                text_column="text",
                description="High-quality web content from Common Crawl"
            ),
            "fineweb_sample": DatasetConfig(
                name="FineWeb Sample",
                path="HuggingFaceFW/fineweb",
                subset="sample-100BT",
                text_column="text",
                description="Smaller sample of FineWeb for quick testing"
            ),
            "wikisource_en": DatasetConfig(
                name="Wikisource English",
                path="wikimedia/wikisource",
                subset="20231201.en",
                text_column="text",
                description="Public domain texts and documents"
            ),
            "common_corpus": DatasetConfig(
                name="Common Corpus",
                path="PleIAs/common_corpus",
                text_column="text",
                description="Large multilingual open training dataset (2T tokens)"
            ),
            "long_data": DatasetConfig(
                name="Long Data Collections",
                path="togethercomputer/Long-Data-Collections",
                text_column="text",
                description="Long-form text data for training"
            ),
        }
        
    def list_available_datasets(self) -> None:
        print("Available datasets:")
        for key, config in self.available_datasets.items():
            print(f"  {key}: {config.name}")
            print(f"    Description: {config.description}")
            print(f"    Path: {config.path}")
            if config.subset:
                print(f"    Subset: {config.subset}")
            print()
            
    def load_dataset_config(self, dataset_key: str, **kwargs) -> Dataset:
        if dataset_key not in self.available_datasets:
            raise ValueError(f"Dataset '{dataset_key}' not found. Available: {list(self.available_datasets.keys())}")
            
        config = self.available_datasets[dataset_key]
        
        logger.info(f"Loading dataset: {config.name}")
        
        load_kwargs = {
            "path": config.path,
            "split": config.split,
            "cache_dir": self.cache_dir,
            "streaming": config.streaming,
        }
        
        if config.subset:
            load_kwargs["name"] = config.subset
            
        load_kwargs.update(kwargs)
        
        try:
            dataset = load_dataset(**load_kwargs)
            logger.info(f"Successfully loaded {config.name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load {config.name}: {e}")
            raise
            

        
    def get_text_samples(self, dataset: Dataset, num_samples: int = 1000, 
                        text_column: str = "text", min_length: int = 100) -> List[str]:
        samples = []
        
        if hasattr(dataset, '__iter__'):
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                    
                text = item.get(text_column, "")
                if isinstance(text, str) and len(text) >= min_length:
                    samples.append(text)
                    
        else:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for i in indices[:num_samples]:
                item = dataset[i]
                text = item.get(text_column, "")
                if isinstance(text, str) and len(text) >= min_length:
                    samples.append(text)
                    
        logger.info(f"Extracted {len(samples)} text samples")
        return samples
        
    def load_datasets(self, dataset_keys: List[str], 
                     samples_per_dataset: Optional[int] = None,
                     total_samples: Optional[int] = None) -> List[str]:
        all_samples = []
        
        for key in dataset_keys:
            try:
                dataset = self.load_dataset_config(key)
                config = self.available_datasets[key]
                
                num_samples = samples_per_dataset or 1000
                if total_samples:
                    remaining = total_samples - len(all_samples)
                    if remaining <= 0:
                        break
                    num_samples = min(num_samples, remaining)
                    
                samples = self.get_text_samples(
                    dataset, 
                    num_samples=num_samples,
                    text_column=config.text_column
                )
                
                all_samples.extend(samples)
                logger.info(f"Added {len(samples)} samples from {config.name}")
                
            except Exception as e:
                logger.error(f"Failed to load samples from {key}: {e}")
                continue
                
        logger.info(f"Loaded dataset with {len(all_samples)} total samples")
        return all_samples
        
    def save_dataset_to_file(self, samples: List[str], filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample.strip() + '\n')
                
        logger.info(f"Saved {len(samples)} samples to {filename}")
        
    def load_dataset_from_file(self, filename: str) -> List[str]:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dataset file not found: {filename}")
            
        with open(filename, 'r', encoding='utf-8') as f:
            samples = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Loaded {len(samples)} samples from {filename}")
        return samples

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Acquisition for POL-AI Training")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--datasets", nargs="+", help="Dataset keys to load")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to extract")
    parser.add_argument("--samples-per-dataset", type=int, help="Samples per dataset")
    parser.add_argument("--output", type=str, help="Output file to save samples")
    parser.add_argument("--cache-dir", type=str, default="./data_cache", help="Cache directory")
    
    args = parser.parse_args()
    
    acquisition = DataAcquisition(cache_dir=args.cache_dir)
    
    if args.list:
        acquisition.list_available_datasets()
        return
        
    if not args.datasets:
        print("Please specify --datasets with one or more dataset keys")
        print("Use --list to see available datasets")
        return
        
    samples = acquisition.load_datasets(
        dataset_keys=args.datasets,
        samples_per_dataset=args.samples_per_dataset,
        total_samples=args.samples
    )
        
    if args.output:
        acquisition.save_dataset_to_file(samples, args.output)
    else:
        print(f"Loaded {len(samples)} samples")
        if samples:
            print(f"Sample text: {samples[0][:200]}...")

if __name__ == "__main__":
    main() 