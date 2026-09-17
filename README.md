# EuroSAT Classification using Transfer Learning

A hands-on exploration of transfer learning using VGG16 pretrained on ImageNet for multi-class satellite image classification. The model was progressively adapted through freezing, fine-tuning, Global Average Pooling, and dropout.

## Dataset

- 27,000 satellite images
- Image size: 224 × 224 × 3
- 10-class classification
- 21,600 training / 5,400 validation images

## Results

| Model | Validation Accuracy |
|---|---:|
| VGG16 from Scratch | 86.24% |
| ImageNet + Frozen Backbone | 92.30% |
| Fine-Tune Block 5 | 93.80% |
| Fine-Tune Blocks 4–5 | 95.98% |
| + Global Average Pooling | 97.43% |
| + Dropout | **97.78%** |

## Final Architecture

```text
Input (224 × 224 × 3)
        ↓
VGG16 (ImageNet)
        ↓
Blocks 1–3 → Frozen
        ↓
Blocks 4–5 → Fine-Tuned
        ↓
Global Average Pooling
        ↓
Dense (256) → ReLU
        ↓
Dropout (0.5)
        ↓
Dense (10) → Softmax
```

The project focuses on understanding how pretrained CNN representations can be adapted to a different visual domain through selective fine-tuning and architectural changes.

**Tech:** Python, TensorFlow, Keras, NumPy, Matplotlib
