# Quantization-Aware Knowledge Distillation (QKD)

This repository implements Quantization-Aware Knowledge Distillation for Vision Transformers using PyTorch.

## Folder Structure

```
QKD-Quantization-aware-Knowledge-Distillation/
│
├── QKD.py                  # Main training script
├── models/
│   ├── vit.py              # Vision Transformer definition
│   └── quant_vit.py        # Quantized Vision Transformer
├── utils/
│   └── quant_utils.py      # Quantization helper functions
├── data/
│   └── imagenet_loader.py  # Data loading utilities
├── README.md
```

## Requirements

- Python 3.8+
- torch
- torchvision
- numpy

Install dependencies:
```
pip install torch torchvision numpy
```

## Usage

1. Prepare your ImageNet dataset and update the dataset path in `QKD.py`.
2. Run the main script:
   ```
   python QKD.py
   ```
3. The best quantized student model will be saved as `quantized_vit.pth`.

## Project Phases

- **Self-studying:** Train the quantized student model alone.
- **Co-studying:** Jointly train teacher and student with mutual distillation.
- **Tutoring:** Fine-tune the student with the fixed teacher.

---

Feel free to modify this README for your specific needs!