#!/usr/bin/env python3
"""
Inference script cho Gemma-2B đã fine-tune với LoRA
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

class GemmaTranslator:
    def __init__(self, base_model_name="google/gemma-2b", lora_path="./gemma-translation-lora"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        print("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def create_prompt(self, instruction, input_text):
        """
        Tạo prompt theo format training
        """
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return prompt
    
    def translate(self, chinese_text, instruction=None, max_length=1024, temperature=0.7, do_sample=True):
        """
        Dịch văn bản từ Tiếng Trung sang Tiếng Việt
        """
        if instruction is None:
            instruction = "Translate in ancient/period drama style, from Chinese to Vietnamese. Keep all <|xx.xx|> timestamps unchanged."
        
        # Tạo input text với format giống training data
        input_text = f"Chinese: {chinese_text}\nVietnamese:"
        
        prompt = self.create_prompt(instruction, input_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response
        response_start = generated_text.find("### Response:\n") + len("### Response:\n")
        response = generated_text[response_start:].strip()
        
        return response
    
    def translate_with_timestamps(self, chinese_text_with_timestamps):
        """
        Dịch văn bản có timestamps
        """
        instruction = "Translate in ancient/period drama style, from Chinese to Vietnamese. Keep all <|xx.xx|> timestamps unchanged."
        return self.translate(chinese_text_with_timestamps, instruction)
    
    def batch_translate(self, texts, instruction=None):
        """
        Dịch nhiều văn bản cùng lúc
        """
        results = []
        for text in texts:
            result = self.translate(text, instruction)
            results.append(result)
        return results

def test_translator():
    """
    Test function cho translator
    """
    # Khởi tạo translator
    translator = GemmaTranslator()
    
    # Test cases
    test_cases = [
        "<|59.04|> 小猫咪 <|60.32|> <|67.84|> 小妹妹 <|69.24|> <|69.96|> 来跟哥哥一起玩儿啊 <|72.28|>",
        "<|88.78|> 虎妖闯入禁地要吃人了 <|90.60|> <|92.44|> 菩萨显灵 <|93.50|>",
        "你好，今天天气怎么样？",
        "我们一起去吃饭吧。"
    ]
    
    print("=== Testing Gemma Translator ===\n")
    
    for i, chinese_text in enumerate(test_cases):
        print(f"Test {i+1}:")
        print(f"Chinese: {chinese_text}")
        
        # Dịch
        vietnamese = translator.translate_with_timestamps(chinese_text)
        print(f"Vietnamese: {vietnamese}")
        print("-" * 50)

def interactive_mode():
    """
    Chế độ tương tác
    """
    translator = GemmaTranslator()
    
    print("=== Gemma Translator - Interactive Mode ===")
    print("Nhập văn bản tiếng Trung để dịch (hoặc 'quit' để thoát):")
    
    while True:
        chinese_text = input("\nChinese: ").strip()
        
        if chinese_text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not chinese_text:
            continue
        
        try:
            vietnamese = translator.translate_with_timestamps(chinese_text)
            print(f"Vietnamese: {vietnamese}")
        except Exception as e:
            print(f"Error: {e}")

def translate_file(input_file, output_file):
    """
    Dịch file JSON
    """
    translator = GemmaTranslator()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    for i, item in enumerate(data):
        print(f"Translating {i+1}/{len(data)}...")
        
        chinese_text = item.get('input', '').replace('Chinese: ', '').replace('\nVietnamese:', '')
        vietnamese = translator.translate_with_timestamps(chinese_text)
        
        result = {
            'original': item,
            'chinese': chinese_text,
            'vietnamese_predicted': vietnamese,
            'vietnamese_ground_truth': item.get('output', '')
        }
        results.append(result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_translator()
        elif sys.argv[1] == "interactive":
            interactive_mode()
        elif sys.argv[1] == "file" and len(sys.argv) == 4:
            translate_file(sys.argv[2], sys.argv[3])
        else:
            print("Usage:")
            print("  python inference.py test")
            print("  python inference.py interactive")
            print("  python inference.py file input.json output.json")
    else:
        # Default: interactive mode
        interactive_mode()