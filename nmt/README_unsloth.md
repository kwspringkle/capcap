## Cài đặt

### 1. Cài đặt PyTorch với CUDA support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Cài đặt Unsloth và dependencies
```bash
pip install unsloth
pip install transformers==4.55.4
pip install --no-deps trl==0.22.2
pip install nltk
pip install pandas openpyxl
pip install datasets
pip install tqdm
pip install sacrebleu

```