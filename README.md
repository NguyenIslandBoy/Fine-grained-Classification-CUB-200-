# Fine-Grained Bird Classification - CUB-200-2011

Image classification of 200 bird species using transfer learning (EfficientNetB5, **86% test accuracy**) and a custom CNN trained from scratch (**50% test accuracy**) on 5,994 training images.

## Results

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|:---:|:---:|:---:|:---:|
| EfficientNetB5 (transfer learning) | **86%** | 0.86 | 0.86 | 0.86 |
| Custom CNN (from scratch) | **50%** | 0.51 | 0.50 | 0.49 |

## Dataset

[CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) - 11,788 images across 200 North American bird species with bounding box annotations, 15 part locations per image, and 312 binary attributes per class.

- **Train:** 5,394 images (90% of official train split)
- **Validation:** 600 images (10% stratified split)
- **Test:** 5,794 images (official test split)

## Approach

### Model 1: EfficientNetB5 (Transfer Learning)

Two-phase progressive unfreezing strategy:

- **Phase 1** - Frozen backbone, train classification head only (GAP -> BN -> Dense(512) -> Dense(200)) with Adam (lr=1e-3) for 20 epochs
- **Phase 2** - Unfreeze blocks 6, 7, and top conv layers; keep all BatchNorm frozen. Fine-tune with RMSprop + cosine decay (lr=5e-5) for 20 epochs

Key decisions:
- Bounding box crops with LANCZOS interpolation to focus on bird regions
- BatchNorm layers kept frozen during fine-tuning to preserve ImageNet statistics
- RMSprop chosen over Adam for more stable fine-tuning of pretrained weights
- Data augmentation: horizontal flip, rotation, zoom, translation, brightness, contrast

### Model 2: Custom CNN (From Scratch)

ResNet-style architecture (~7.2M parameters) with modern components:

- **Residual blocks** with pre-activation BatchNorm for gradient flow through 20+ layers
- **Squeeze-and-Excitation attention** for channel-wise feature recalibration
- **Depthwise separable convolutions** in Stage 4 for parameter efficiency (~8× reduction)
- **Global Average Pooling** instead of Flatten (25K vs 5M classifier parameters)

Architecture: Stem (7×7 conv) -> 4 stages (64->128->256->512 channels) -> SE -> GAP -> Dense(512) -> Dense(200)

Training: AdamW (weight_decay=1e-4) with 5-epoch warmup + cosine annealing (peak lr=1e-3)

## Project Structure

```
├── Coursework_Fine_Grained_Classification.ipynb   # Full training + evaluation notebook
├── b5_phase2.keras                                 # EfficientNetB5 checkpoint (87%)
├── custom_model_best.keras                         # Custom CNN checkpoint
├── bird_CUB_200_2011.zip                           # Dataset
└── README.md
```

## How to Run

### Requirements
- Google Colab with GPU runtime (L4 or A100 recommended)
- CUB-200-2011 dataset in Google Drive

### Full Training
1. Upload dataset to `My Drive/Colab Notebooks/Deep Learning/CUB_200_2011/`
2. Open notebook in Colab, set runtime to GPU
3. Run all cells sequentially

### Inference Only
1. Run Sections 1–2 (Setup + Data Loading)
2. Run model loading cells to load saved checkpoints from Drive
3. Run evaluation cells (Sections 3.4 and 4.8)
4. Run Demo section (Section 6) with any provided test folder

### Demo
Set `DEMO_TEST_FOLDER` to the provided test folder path and run the demo cell. It loads both models and reports accuracy, precision, recall, F1-score, and confusion matrix.

Expected test folder structure:
```
Test/
├── 001.Black_footed_Albatross/
│   └── *.jpg
├── 002.Laysan_Albatross/
│   └── *.jpg
└── ...
```

## Reproducibility

All random seeds fixed to 42 (`tf.random.set_seed`, `np.random.seed`, `random_state`). Stratified train/val split ensures consistent class distributions across runs.

## Tools

Python, TensorFlow/Keras, EfficientNetB5 (ImageNet weights), Google Colab (GPU)
