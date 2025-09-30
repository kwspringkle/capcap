#!/usr/bin/env python3
"""
Cấu hình LoRA tùy chỉnh cho các task khác nhau
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json

@dataclass
class LoRAConfig:
    """Cấu hình LoRA"""
    r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling parameter
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.1
    bias: str = "none"
    modules_to_save: Optional[List[str]] = None

@dataclass  
class TrainingConfig:
    """Cấu hình training"""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    fp16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"

@dataclass
class DataConfig:
    """Cấu hình dữ liệu"""
    max_length: int = 2048
    train_file: str = "gemma_sft_train.json"
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    instruction: str = "Translate in ancient/period drama style, from Chinese to Vietnamese. Keep all <|xx.xx|> timestamps unchanged."

# Các preset cấu hình cho các task khác nhau
TRANSLATION_PRESET = {
    "lora": LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1
    ),
    "training": TrainingConfig(
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8
    ),
    "data": DataConfig(
        max_length=2048,
        instruction="Translate in ancient/period drama style, from Chinese to Vietnamese. Keep all <|xx.xx|> timestamps unchanged."
    )
}

CHAT_PRESET = {
    "lora": LoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05
    ),
    "training": TrainingConfig(
        num_train_epochs=5,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4
    ),
    "data": DataConfig(
        max_length=1024,
        instruction="You are a helpful assistant."
    )
}

SUMMARIZATION_PRESET = {
    "lora": LoRAConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1
    ),
    "training": TrainingConfig(
        num_train_epochs=2,
        learning_rate=3e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16
    ),
    "data": DataConfig(
        max_length=4096,
        instruction="Summarize the following text:"
    )
}

def get_preset(task_type: str):
    """Lấy preset cho task cụ thể"""
    presets = {
        "translation": TRANSLATION_PRESET,
        "chat": CHAT_PRESET, 
        "summarization": SUMMARIZATION_PRESET
    }
    
    if task_type not in presets:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(presets.keys())}")
    
    return presets[task_type]

def save_config(config_dict: dict, filepath: str):
    """Lưu cấu hình ra file JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def load_config(filepath: str) -> dict:
    """Load cấu hình từ file JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    # Tạo config files cho các task khác nhau
    
    # Translation config
    translation_config = get_preset("translation")
    save_config({
        "lora_config": translation_config["lora"].__dict__,
        "training_config": translation_config["training"].__dict__,
        "data_config": translation_config["data"].__dict__
    }, "translation_config.json")
    
    # Chat config  
    chat_config = get_preset("chat")
    save_config({
        "lora_config": chat_config["lora"].__dict__,
        "training_config": chat_config["training"].__dict__,
        "data_config": chat_config["data"].__dict__
    }, "chat_config.json")
    
    print("Config files created:")
    print("- translation_config.json")
    print("- chat_config.json")