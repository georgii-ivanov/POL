#!/usr/bin/env python3

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_model():
    model_name = "deepseek-ai/DeepSeek-R1"
    cache_dir = "/model_cache"
    
    print(f"Setting up model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(f"{cache_dir}/tokenizer")
        print("Tokenizer cached successfully")
        
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu"
        )
        model.save_pretrained(f"{cache_dir}/model")
        print("Model cached successfully")
        
        print("Creating deltas directory...")
        os.makedirs(f"{cache_dir}/deltas", exist_ok=True)
        
        print("Model setup completed successfully!")
        
    except Exception as e:
        print(f"Error setting up model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_model() 