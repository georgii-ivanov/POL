#!/bin/bash

echo "Setting up POL-AI development environment..."

pip3 install --user --upgrade pip

if [ -f "requirements-ai.txt" ]; then
    echo "Installing AI requirements..."
    pip3 install --user -r requirements-ai.txt
fi

if [ -f "requirements-dht.txt" ]; then
    echo "Installing DHT requirements..."
    pip3 install --user -r requirements-dht.txt
fi

echo "Setup complete!"
echo "You can now run the POL-AI system components." 