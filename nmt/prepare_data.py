import json
from datasets import Dataset
from transformers import AutoTokenizer

def prepare_gemma_dataset(json_file_path, output_dir="./prepared_data"):
    """
    Chuẩn bị dataset cho fine-tuning Gemma-2B với LoRA
    """
    
    # Load dataset
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Chuẩn bị format chat cho training
    formatted_data = []
    
    for item in data:
        # Format theo chuẩn Gemma chat template
        conversation = [
            {"role": "user", "content": item["instruction"] + "\n\n" + item["input"]},
            {"role": "assistant", "content": item["output"]}
        ]
        
        # Convert to text format
        text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        formatted_data.append({
            "text": text,
            "input_ids": tokenizer.encode(text, truncation=True, max_length=2048)
        })
    
    # Tạo dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Save dataset
    dataset.save_to_disk(output_dir)
    
    print(f"Dataset đã được chuẩn bị và lưu tại: {output_dir}")
    print(f"Số lượng samples: {len(dataset)}")
    
    # Hiển thị ví dụ
    print("\nVí dụ đầu tiên:")
    print("Text:", formatted_data[0]["text"][:500] + "...")
    print("Input IDs length:", len(formatted_data[0]["input_ids"]))
    
    return dataset

def create_simple_format(json_file_path, output_file="train_data.jsonl"):
    """
    Tạo format đơn giản hơn cho training
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # Format prompt đơn giản
            formatted_item = {
                "prompt": f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n",
                "completion": item['output'],
                "text": f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            }
            f.write(json.dumps(formatted_item, ensure_ascii=False) + '\n')
    
    print(f"Simple format dataset saved to: {output_file}")

if __name__ == "__main__":
    # Chạy chuẩn bị dữ liệu
    dataset = prepare_gemma_dataset("gemma_sft_train.json")
    
    # Tạo format đơn giản
    create_simple_format("gemma_sft_train.json")