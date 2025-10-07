import json
import re
import os
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("hf_write_token")
login(token=hf_token)
def remove_timestamps_keep_spacing(text: str) -> str:
    """
    Xóa timestamp dạng <|xx.xx|> nhưng giữ nguyên vị trí bằng cách thay bằng 2 dấu cách.
    """
    # Thay thế mỗi timestamp bằng 2 dấu cách
    text = re.sub(r"<\|[^|]*\|>", "  ", str(text))
    # Chuẩn hóa khoảng trắng (chỉ 1 space ở đầu/cuối)
    text = re.sub(r" {3,}", "  ", text)  # tránh thừa khoảng trắng
    return text.strip()

def clean_text_for_instruction(text: str) -> str:
    """Làm sạch text nhưng giữ nguyên timestamp (nếu còn) để dùng trong instruction."""
    return re.sub(r"\s+", " ", str(text)).strip()

def map_style(tag: str) -> str:
    """Map tag tiếng Việt -> instruction tiếng Anh"""
    mapping = {
        "co_trang": "Translate in ancient/period Chinese drama style, from Chinese to Vietnamese.",
        "hien_dai": "Translate in modern style, from Chinese to Vietnamese.",
    }
    return mapping.get(tag, f"Translate in {tag} style.")

def build_record(zh: str, vi: str, tags, movie_name: str = None) -> dict:
    """Tạo 1 record instruction-style"""
    # Ưu tiên kiểm tra co_trang và hien_dai trong list tags
    selected_tag = None
    
    if isinstance(tags, list):
        # Ưu tiên co_trang trước
        if "co_trang" in tags:
            selected_tag = "co_trang"
        elif "hien_dai" in tags:
            selected_tag = "hien_dai"
        # Nếu không có 2 tag chính, lấy tag đầu tiên
        elif len(tags) > 0:
            selected_tag = tags[0]
    elif isinstance(tags, str):
        selected_tag = tags

    style_instruction = map_style(selected_tag) if selected_tag else "Translate from Chinese to Vietnamese."
    record = {
        "instruction": f"{style_instruction}",
        "input": f"Chinese: {remove_timestamps_keep_spacing(zh)}\nVietnamese:",
        "output": remove_timestamps_keep_spacing(vi),
    }
    
    # Thêm movie_name nếu có
    if movie_name:
        record["movie_name"] = movie_name
    
    return record

def convert_to_sft(dataset, output_path="gemma_sft_dataset.json"):
    records = []
    # đảm bảo không shuffle — duyệt tuần tự
    for i in range(len(dataset)):
        ex = dataset[i]
        zh = ex.get("chinese_text", "")
        vi = ex.get("vietnamese_text", "")
        tags = ex.get("tags", None)
        movie_name = ex.get("movie_name", None)
        if not zh or not vi:
            continue
        records.append(build_record(zh, vi, tags, movie_name))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(records)} samples to {output_path}")

# Load dataset từ HF
dataset = load_dataset("kwspringkles/film_final")

for split_name, split_data in dataset.items():
    output_file = f"gemma_sft_{split_name}.json"
    convert_to_sft(split_data, output_file)
