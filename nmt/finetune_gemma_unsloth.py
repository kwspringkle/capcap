#!/usr/bin/env python3
"""
Fine-tune Gemma-2B với Unsloth cho task dịch thuật Chinese -> Vietnamese
Converted from Jupyter notebook to Python script
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from transformers import EarlyStoppingCallback
# Configuration
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True
MODEL_NAME = "unsloth/gemma-2-2b"

# Data paths - adjust these according to your setup
TRAIN_DATA_PATH = "gemma_sft_train.json"
VALIDATION_DATA_PATH = "gemma_sft_validation.json"
TEST_DATA_PATH = "gemma_sft_test.json"

# Alpaca prompt template
ALPACA_PROMPT = """Below is an instruction for a translation task. Write a translation that accurately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def setup_model_and_tokenizer():
    """Load and setup the model and tokenizer"""
    print("Loading model and tokenizer...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.1,  # Supports any, but = 0 is optimized
        bias="none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    
    return model, tokenizer


def formatting_prompts_func(examples, tokenizer):
    """Format the examples into alpaca prompt format"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ALPACA_PROMPT.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    
    return {"text": texts}


def load_and_prepare_dataset(tokenizer):
    """Load and prepare the dataset"""
    print("Loading dataset...")
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_DATA_PATH,
            "validation": VALIDATION_DATA_PATH,
            "test": TEST_DATA_PATH
        }
    )
    
    # Apply formatting
    dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)
    
    return dataset


def train_model(model, tokenizer, dataset):
    """Train the model"""
    print("Starting training...")
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            num_train_epochs=10,  # Set this for 1 full training run.
            learning_rate=1e-4,
            logging_steps=20,
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_type="inverse_sqrt",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
        callbacks=[early_stopping], 
    )
    
    # Show memory stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    return model, trainer_stats


def generate_output(model, tokenizer, instruction, inp):
    """Generate output from the model"""
    prompt = ALPACA_PROMPT.format(instruction, inp, "")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=False,
        use_cache=True,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the response part
    if "### Response:" in decoded:
        decoded = decoded.split("### Response:")[-1].strip()
    return decoded


def evaluate_model(model, tokenizer, dataset, split_name):
    """Evaluate the model and return DataFrame with results"""
    print(f"Evaluating on {split_name} set...")
    
    data = dataset[split_name]
    preds, refs, instructions, inputs = [], [], [], []
    
    smoothie = SmoothingFunction().method4
    
    for sample in tqdm(data, desc=f"Generating on {split_name} set"):
        pred = generate_output(model, tokenizer, sample["instruction"], sample["input"])
        preds.append(pred)
        refs.append(sample["output"])
        instructions.append(sample["instruction"])
        inputs.append(sample["input"])
    
    # Calculate BLEU scores for each sentence
    sentence_bleus = [
        sentence_bleu([r.split()], p.split(), smoothing_function=smoothie)
        for r, p in zip(refs, preds)
    ]
    
    # Calculate corpus BLEU
    corpus_bleu_score = corpus_bleu(
        [[r.split()] for r in refs], [p.split() for p in preds], smoothing_function=smoothie
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        "instruction": instructions,
        "input": inputs,
        "reference": refs,
        "prediction": preds,
        "sentence_bleu": sentence_bleus,
    })
    df["corpus_bleu"] = corpus_bleu_score
    
    print(f"✅ {split_name.upper()} corpus BLEU: {corpus_bleu_score:.4f}")
    return df, corpus_bleu_score


def save_results_to_excel(val_df, test_df, val_bleu, test_bleu, output_path="bleu_results.xlsx"):
    """Save evaluation results to Excel file"""
    print(f"Saving results to {output_path}...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        val_df.to_excel(writer, sheet_name='Validation', index=False)
        test_df.to_excel(writer, sheet_name='Test', index=False)
        
        # Summary sheet
        summary_df = pd.DataFrame({
            'Dataset': ['Validation', 'Test'],
            'Corpus BLEU': [round(val_bleu, 4), round(test_bleu, 4)]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"📘 Results saved to: {output_path}")
    print(f"Validation corpus BLEU = {val_bleu:.4f}")
    print(f"Test corpus BLEU = {test_bleu:.4f}")


def save_model(model, tokenizer, output_dir="lora_model"):
    """Save the trained model"""
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def main():
    """Main execution function"""
    print("Starting Gemma-2B fine-tuning with Unsloth...")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(tokenizer)
    
    # Show a sample
    print("\nSample from dataset:")
    print(dataset['train'][0]['text'][:500] + "...")
    
    # Train the model
    trained_model, trainer_stats = train_model(model, tokenizer, dataset)
    
    # Switch to inference mode
    FastLanguageModel.for_inference(trained_model)
    
    # Evaluate the model
    val_df, val_bleu = evaluate_model(trained_model, tokenizer, dataset, "validation")
    test_df, test_bleu = evaluate_model(trained_model, tokenizer, dataset, "test")
    
    # Save results
    save_results_to_excel(val_df, test_df, val_bleu, test_bleu)
    
    # Save the model
    save_model(trained_model, tokenizer)
    
    print("\n🎉 Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()