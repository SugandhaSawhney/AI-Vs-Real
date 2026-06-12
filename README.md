# 🔍 AI vs Real Image Detector

Binary image classifier to detect **AI-generated vs real images** using fine-tuned EfficientNetB0 on the CIFAKE dataset.

---

## 📌 Overview

With the rise of AI image generation tools like Stable Diffusion and DALL-E, detecting synthetic images has become a critical problem in content moderation, journalism, and digital forensics. This project builds a deep learning pipeline to classify images as **REAL** or **AI-GENERATED** with high accuracy.

---

## 🧠 Model Architecture

- **Backbone**: EfficientNetB0 (pretrained on ImageNet)
- **Training Strategy**: Two-phase fine-tuning
  - Phase 1: Frozen backbone, train custom head only
  - Phase 2: Partial unfreeze (MBConv blocks 4–8) + head
- **Custom Head**: BN → Dropout → Linear(1280, 512) → GELU → BN → Dropout → Linear(512, 128) → GELU → Linear(128, 2)
- **Loss**: Cross-Entropy with label smoothing (0.1)
- **Optimizer**: AdamW with OneCycleLR scheduler

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 85%+ |
| AUC-ROC | 0.89 |
| F1 Score | 0.84 |

> Training in progress — Phase 2 fine-tuning pending. Expected final accuracy: 93–95%

---

## 🗂️ Dataset

**CIFAKE: Real and AI-Generated Synthetic Images**
- 60,000 real images (from CIFAR-10)
- 60,000 AI-generated images (Stable Diffusion)
- Total: 120,000 images | Train/Test split included

Download from Kaggle:
```
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
```

Place dataset as:
```
data/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

---

## 📁 Project Structure

```
AI-Vs-Real/
├── src/
│   ├── model.py        # EfficientNetB0 + custom head
│   ├── dataset.py      # Data loading + augmentation pipeline
│   ├── train.py        # Two-phase training loop
│   ├── evaluate.py     # Metrics + confusion matrix + ROC curve
│   └── predict.py      # Single image inference
├── app.py              # Gradio demo UI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/SugandhaSawhney/AI-Vs-Real.git
cd AI-Vs-Real

# 2. Create virtual environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python src/train.py \
  --data_dir data \
  --output_dir outputs \
  --batch_size 32 \
  --num_workers 0 \
  --phase1_epochs 5 \
  --phase2_epochs 15
```

Training logs saved to `outputs/results/training_log.csv`

---

## 📈 Evaluation

```bash
python src/evaluate.py --data_dir data --checkpoint outputs/best_model.pth
```

Outputs saved to `outputs/results/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `classification_report.txt`

---

## 🖥️ Demo

```bash
python app.py
```

Opens a Gradio interface in browser — upload any image to get a Real/Fake prediction with confidence score.

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Deep Learning | PyTorch |
| Model | EfficientNetB0 (torchvision) |
| Data Augmentation | torchvision.transforms |
| Evaluation | scikit-learn |
| Demo UI | Gradio |
| Training GPU | Google Colab T4 |

---

## 📚 Key Deep Learning Concepts

- Transfer Learning & fine-tuning
- Two-phase training strategy
- OneCycleLR scheduling
- Label smoothing regularization
- Mixed precision training (AMP)
- WeightedRandomSampler for class balance
- Grad-CAM visualization (coming soon)

---

## 🔮 Future Work

- Add Grad-CAM visualizations to show which image regions indicate AI generation
- Extend to detect images from newer generators (Midjourney, DALL-E 3)
- Deploy on HuggingFace Spaces
