#!/usr/bin/env python3
"""
Fine-tune Gemma-2B với LoRA cho task dịch thuật với BLEU metric và validation
"""
from dotenv import load_dotenv
from huggingface_hub import login
import os
import json
import torch
import numpy as np
import sacrebleu
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset, load_from_disk
import wandb
from datetime import datetime
print("Logging in to Hugging Face...")
login(token=os.getenv("hf_write_token"))
class CustomTrainer(Trainer):
    """Custom Trainer với BLEU evaluation"""
    
    def __init__(self, *args, eval_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_tokenizer = eval_tokenizer
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate để thêm BLEU score"""
        
        # Gọi evaluate gốc
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Nếu có eval dataset, tính BLEU score
        if eval_dataset is not None and hasattr(eval_dataset, 'select'):
            bleu_score = self.compute_bleu_score(eval_dataset)
            eval_result[f"{metric_key_prefix}_bleu"] = bleu_score
            
            # Log BLEU score
            if self.state.is_world_process_zero:
                print(f"BLEU Score: {bleu_score:.4f}")
        
        return eval_result
    
    def compute_bleu_score(self, eval_dataset, num_samples=50):
        """Tính BLEU score trên validation set"""
        if len(eval_dataset) < num_samples:
            num_samples = len(eval_dataset)
        
        # Sample random examples
        indices = np.random.choice(len(eval_dataset), num_samples, replace=False)
        sample_dataset = eval_dataset.select(indices)
        
        predictions = []
        references = []
        
        self.model.eval()
        with torch.no_grad():
            for example in sample_dataset:
                # Tạo prompt
                prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
                
                # Tokenize input
                inputs = self.eval_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(self.model.device)
                
                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.eval_tokenizer.eos_token_id,
                    eos_token_id=self.eval_tokenizer.eos_token_id,
                )
                
                # Decode prediction
                generated_text = self.eval_tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_start = generated_text.find("### Response:\n") + len("### Response:\n")
                prediction = generated_text[response_start:].strip()
                
                predictions.append(prediction)
                references.append(example['output'])
        
        # Tính BLEU score
        try:
            bleu = sacrebleu.corpus_bleu(predictions, [references])
            return bleu.score
        except Exception:
            return 0.0

def setup_model_and_tokenizer(model_name="google/gemma-2b", use_4bit=True):
    """
    Setup model và tokenizer với quantization
    """
    print(f"Loading model: {model_name}")
    
    # Quantization config
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config(target_modules=None):
    """
    Cấu hình LoRA
    """
    if target_modules is None:
        # Target modules cho Gemma
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,  # scaling parameter
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    return lora_config

def load_and_prepare_dataset(data_path, tokenizer, max_length=2048, validation_path=None):
    """
    Load và chuẩn bị dataset
    """
    def load_json_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format dữ liệu
        formatted_data = []
        for item in data:
            text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}{tokenizer.eos_token}"
            formatted_data.append({
                "text": text,
                "instruction": item['instruction'],
                "input": item['input'],
                "output": item['output']
            })
        return Dataset.from_list(formatted_data)
    
    if data_path.endswith('.json'):
        train_dataset = load_json_data(data_path)
    elif os.path.isdir(data_path):
        train_dataset = load_from_disk(data_path)
    else:
        raise ValueError("Data path must be JSON file or directory")
    
    val_dataset = None
    if validation_path and validation_path.endswith('.json'):
        val_dataset = load_json_data(validation_path)
    
    # def tokenize_function(examples):
    #     # Tokenize without return_tensors để tránh lỗi batching
    #     tokenized = tokenizer(
    #         examples["text"],
    #         truncation=True,
    #         padding=True,
    #         max_length=max_length,
    #         return_tensors=None  # Không return tensors ở đây
    #     )
    #     # Labels = input_ids cho causal LM
    #     tokenized["labels"] = tokenized["input_ids"].copy()
    #     return tokenized
    
    def tokenize_function(examples):
        """
        Tokenize function với fix cho nested structures
        """
    # Đảm bảo examples["text"] là list của strings, không phải nested lists
        texts = examples["text"]
        
        # Nếu có bất kỳ nested structure nào, flatten nó
        if texts and isinstance(texts[0], list):
            print("WARNING: Found nested list in texts, flattening...")
            flattened_texts = []
            for text_list in texts:
                if isinstance(text_list, list):
                    flattened_texts.extend(text_list)
                else:
                    flattened_texts.append(text_list)
            texts = flattened_texts
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,  # Let DataCollator handle padding
            max_length=max_length,
            return_tensors=None
        )
        
        # Labels = input_ids cho causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    # Determine columns to drop after tokenization (keep only model inputs)
    train_remove_columns = train_dataset.column_names
    val_remove_columns = val_dataset.column_names if val_dataset is not None else None

    # Tokenize train dataset
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
    remove_columns=train_remove_columns
    )
    
    # Tokenize validation dataset if provided
    tokenized_val = None
    if val_dataset is not None:
        tokenized_val = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_remove_columns
        )
    
    return tokenized_train, tokenized_val

def train_model(
    model_name="google/gemma-2b",
    data_path="gemma_sft_train.json",
    validation_path=None,
    output_dir="./gemma-translation-lora",
    num_epochs=3,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_length=2048,
    use_wandb=False
):
    """
    Main training function
    """
    
    # Setup wandb
    if use_wandb:
        run_name = f"gemma-translation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project="gemma-translation", name=run_name)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load dataset
    train_dataset, val_dataset = load_and_prepare_dataset(
        data_path, tokenizer, max_length, validation_path
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # Data collator với padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Giúp tối ưu trên GPU
        return_tensors="pt",
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        max_steps=-1,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        metric_for_best_model="eval_bleu" if val_dataset else None,
        greater_is_better=True if val_dataset else None,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb" if use_wandb else "none",
        load_best_model_at_end=True if val_dataset else False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",
    )
    
    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        eval_tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Final evaluation if validation data exists
    if val_dataset:
        print("\nFinal evaluation:")
        final_metrics = trainer.evaluate()
        print(f"Final BLEU Score: {final_metrics.get('eval_bleu', 0):.4f}")
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save LoRA weights
    model.save_pretrained(output_dir)
    
    print(f"Model saved to: {output_dir}")
    
    if use_wandb:
        wandb.finish()
    
    return model, tokenizer

if __name__ == "__main__":
    # Cấu hình training
    config = {
        "model_name": "google/gemma-2b",
        "data_path": "gemma_sft_train.json",
        "validation_path": "gemma_sft_validation.json",  # Thêm validation data
        "output_dir": "./gemma-translation-lora",
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "max_length": 2048,
        "use_wandb": False
    }
    
    # Train model
    model, tokenizer = train_model(**config)
    
    print("Training completed!")