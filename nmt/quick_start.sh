#!/bin/bash
# Quick start script for Gemma LoRA fine-tuning

echo "=== Gemma LoRA Fine-tuning Quick Start ==="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python 3.8+"
    exit 1
fi

# Check if GPU is available
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Prepare data
echo "Preparing data..."
python prepare_data.py
python create_validation.py

# Start training
echo "Starting training..."
python train_gemma_lora.py

echo "Training completed!"
echo "To test the model, run: python inference.py test"
echo "For interactive mode, run: python inference.py interactive"