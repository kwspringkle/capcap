#!/usr/bin/env python3
"""
Script để tạo validation dataset từ training data
"""

import json
import random

def create_validation_split(train_file, validation_file, split_ratio=0.1, seed=42):
    """
    Tạo validation dataset từ training data
    
    Args:
        train_file: File JSON training data
        validation_file: File JSON validation data sẽ được tạo
        split_ratio: Tỷ lệ data cho validation (0.1 = 10%)
        seed: Random seed để reproducible
    """
    
    # Load training data
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total data points: {len(data)}")
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Split data
    val_size = int(len(shuffled_data) * split_ratio)
    
    validation_data = shuffled_data[:val_size]
    remaining_train_data = shuffled_data[val_size:]
    
    print(f"Training data: {len(remaining_train_data)} samples")
    print(f"Validation data: {len(validation_data)} samples")
    
    # Save validation data
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(validation_data, f, ensure_ascii=False, indent=2)
    
    # Update training data (optional - remove validation samples)
    update_train = input("Do you want to update training file to remove validation samples? (y/n): ").lower()
    if update_train == 'y':
        backup_file = train_file.replace('.json', '_backup.json')
        # Backup original
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Original training data backed up to: {backup_file}")
        
        # Save updated training data
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(remaining_train_data, f, ensure_ascii=False, indent=2)
        print(f"Updated training data saved to: {train_file}")
    
    print(f"Validation data saved to: {validation_file}")
    
    # Show sample
    print("\nSample validation data:")
    for i, sample in enumerate(validation_data[:2]):
        print(f"\nSample {i+1}:")
        print(f"Instruction: {sample['instruction']}")
        print(f"Input: {sample['input'][:100]}...")
        print(f"Output: {sample['output'][:100]}...")

def create_sample_validation():
    """
    Tạo validation data mẫu nếu chưa có training data
    """
    
    sample_data = [
        {
            "instruction": "Translate in ancient/period drama style, from Chinese to Vietnamese. Keep all <|xx.xx|> timestamps unchanged.",
            "input": "Chinese: <|59.04|> 小猫咪 <|60.32|> <|67.84|> 小妹妹 <|69.24|>\nVietnamese:",
            "output": "<|59.04|> Chú mèo nhỏ. <|60.32|> <|67.84|> Tiểu muội muội. <|69.24|>"
        },
        {
            "instruction": "Translate in ancient/period drama style, from Chinese to Vietnamese. Keep all <|xx.xx|> timestamps unchanged.", 
            "input": "Chinese: <|88.78|> 虎妖闯入禁地要吃人了 <|90.60|> <|92.44|> 菩萨显灵 <|93.50|>\nVietnamese:",
            "output": "<|88.78|> Hổ yêu xông vào cấm địa ăn thịt người rồi. <|90.60|> <|92.44|> Bồ tát hiển linh. <|93.50|>"
        }
    ]
    
    with open('gemma_sft_validation.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("Sample validation data created: gemma_sft_validation.json")

if __name__ == "__main__":
    import os
    
    train_file = "gemma_sft_train.json"
    validation_file = "gemma_sft_validation.json"
    
    if os.path.exists(train_file):
        print(f"Found training file: {train_file}")
        create_validation_split(train_file, validation_file)
    else:
        print(f"Training file not found: {train_file}")
        print("Creating sample validation data...")
        create_sample_validation()