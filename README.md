# STAMP-RI: Shape-Texture Augmentation with Mining Pipeline for Retinal Images

A deep learning pipeline for **Diabetic Retinopathy (DR) grading** that addresses CNN texture bias through multi-stage texture manipulation, contrastive learning, and online negative mining. The system learns shape-invariant retinal image representations that generalize across clinical sites and imaging conditions.

---

## Problem Statement

Convolutional neural networks exhibit **texture bias** — they classify based on local texture patterns rather than global shape and structure. In medical imaging, this causes models to latch onto camera artifacts, illumination differences, and site-specific color profiles instead of diagnostically meaningful features like lesion morphology and vessel patterns. This project designs augmentation strategies that systematically manipulate low-level texture while preserving semantic content, producing models with improved generalization and transferability.

---

## Pipeline Overview

```
                                    STAMP-RI Pipeline
                                    
  Messidor-2 Dataset                Painters-by-Numbers
  (1057 retinal images)             (artistic paintings)
         |                                  |
         v                                  v
  [1] Classify by DR Grade          Color Histogram Matching
  (Grade 0-4, 5 classes)            (Chi-squared distance)
         |                                  |
         +----------------------------------+
         |
         v
  [2] Neural Style Transfer (VGG-19)
      Content: retinal structure preserved
      Style: artistic texture injected
         |
         v
  [3] DiffuseMix Augmentation
      Concatenate original + NST
      Blend with fractal patterns (alpha-controlled)
         |
         v
  [4] Frequency-Domain Blending (FFT)
      Low-freq: 100% medical image (shape preserved)
      High-freq: blended with fractal (texture varied)
         |
         v
  [5] Contrastive Learning (ResNet-18)
      Triplet loss with online semi-hard negative mining
      Stratified train/val/test split (70/15/15)
         |
         v
  [6] Evaluation
      Silhouette | kNN | Linear Probe | Confusion Matrix | t-SNE
```

---

## Key Features

- **Multi-stage texture manipulation**: NST, pixel-space blending, and frequency-domain blending each provide different levels of texture-shape disentanglement
- **Frequency-domain blending**: FFT-based approach that mathematically guarantees preservation of low-frequency structural content while varying high-frequency texture
- **Stratified train/val/test split**: 70/15/15 split at the image level, consistent across all augmentation methods, with no data leakage
- **Online semi-hard negative mining**: Dynamic negative selection based on current embedding distances, replacing static pre-assigned negatives
- **Weighted sampling**: Handles severe class imbalance (468 Grade 0 vs 21 Grade 4) via `WeightedRandomSampler`
- **Comprehensive evaluation**: Silhouette score, kNN accuracy, linear probe, confusion matrices, classification reports, and t-SNE visualization — all on held-out test set

---

## Results

### Final 4-Way Comparison (Test Set)

| Method | Best Epoch | Silhouette | kNN Acc | Probe Acc |
|--------|:---------:|:----------:|:-------:|:---------:|
| **NST** | 75 | -0.0291 | 0.4843 | **0.5220** |
| Concat | 100 | -0.0459 | 0.4465 | 0.4906 |
| Pixel-Blend (a=0.05) | 100 | -0.0375 | 0.4151 | 0.4717 |
| Freq-Blend (a=0.10) | 125 | **-0.0285** | **0.4780** | 0.4654 |

**Overall Winner: NST** (highest linear probe accuracy on held-out test set)

### Key Findings

- NST augmentation provides the strongest generalization — texture replacement with artistic styles forces shape-based learning
- Frequency-domain blending achieves the best cluster quality (silhouette) and competitive kNN accuracy
- Low blending ratios (alpha=0.05) work best for medical images — subtle texture perturbation outperforms aggressive mixing
- All results are on a held-out test set with model selection via validation loss

---

## Dataset

### Messidor-2 (Classified)

| DR Grade | Description | Images |
|:--------:|-------------|:------:|
| 0 | No DR | 468 |
| 1 | Mild | 207 |
| 2 | Moderate | 290 |
| 3 | Severe | 71 |
| 4 | Proliferative | 21 |
| | **Total** | **1057** |

### Train/Val/Test Split (70/15/15)

| Grade | Train | Val | Test |
|:-----:|:-----:|:---:|:----:|
| 0 | 328 | 70 | 70 |
| 1 | 145 | 31 | 31 |
| 2 | 203 | 44 | 43 |
| 3 | 50 | 11 | 10 |
| 4 | 15 | 3 | 3 |
| **Total** | **739** | **159** | **159** |

### Data Sources

- **Messidor-2**: [Messidor-2 Dataset](https://www.adcis.net/en/third-party/messidor2/) — retinal fundus images with DR grade labels
- **Painters by Numbers**: [Kaggle Competition](https://www.kaggle.com/c/painter-by-numbers) — artistic paintings used as style sources for NST

> **Note**: Image datasets are not included in this repository due to size. Download them separately and place in the project root.

---

## Project Structure

```
beproject/
|
|-- pipeline.ipynb                  # Main pipeline (split + online mining)
|-- pipeline_clean.ipynb            # Clean version of the pipeline
|-- training_utils.py               # Utility functions
|-- PNGtoJPG.ipynb                  # Image format conversion
|
|-- Messidor_Original/
|   |-- messidor_data.csv           # DR grade labels
|
|-- Messidor_Classified/
|   |-- image_mapping.csv           # Image-to-grade mapping
|   |-- ptrbynum_histogram.csv      # Painters histogram features
|   |-- Grade_*_histogram.csv       # Per-grade histogram features
|   |-- Grade_*_matches.csv         # Per-grade painter matches
|   |-- Grade_0_No_DR/              # Original images (not in repo)
|   |-- Grade_0_No_DR_nst/          # NST augmented (not in repo)
|   |-- Grade_0_No_DR_concatenated/ # Concatenated (not in repo)
|   |-- Grade_0_No_DR_blended_*/    # Blended variants (not in repo)
|   |-- Grade_0_No_DR_freqblend_*/  # Freq-blended (not in repo)
|   |-- ... (same for Grades 1-4)
|
|-- Models/
|   |-- pipeline_wo_checkpoints.ipynb
|   |-- pipeline_w_checkpoints.ipynb
|   |-- pipeline_5DR_skewed dataset.ipynb
|   |-- pipeline_frequency_blending.ipynb
|   |-- pipeline_train_val_test.ipynb
|
|-- matches.csv                     # Global histogram matches
|-- messidor_histogram.csv          # Messidor features
|-- ptrbynum_histogram.csv          # Painters features
|-- .gitignore
```

---

## Architecture

### Embedding Network

- **Backbone**: ResNet-18 (pretrained on ImageNet)
- **Embedding dimension**: 128
- **Normalization**: L2-normalized output embeddings
- **Loss**: Triplet loss with margin=1.0

### Neural Style Transfer

- **Model**: VGG-19 (pretrained)
- **Content layer**: `conv4_2` (layer 21) — preserves spatial structure
- **Style layers**: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` — captures texture at multiple scales
- **Optimization**: L-BFGS on input image

### Frequency-Domain Blending

- **Transform**: 2D FFT per color channel
- **Low-pass cutoff**: 0.3 (ratio of frequency space)
- **Blending**: Low-freq 100% from medical image; high-freq blended with fractal at controlled alpha
- **Best alpha**: 0.10

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision opencv-python numpy pandas pillow matplotlib scikit-learn tqdm
```

### Quick Start (if augmented images already exist)

1. Open `pipeline.ipynb` in Jupyter
2. Set `best_freq_alpha = 0.10` (or run the frequency ablation cell)
3. Run the **Phase 2** cell — it is fully self-contained and will:
   - Create the stratified train/val/test split
   - Load datasets for all 4 augmentation methods
   - Train with validation loss tracking
   - Select best checkpoints
   - Evaluate on held-out test set
   - Generate all plots and metrics

### Full Pipeline (from scratch)

Run cells in order:
1. **Classify** Messidor images into DR grades
2. **Histogram matching** between retinal and painter images
3. **NST** for each grade
4. **DiffuseMix** concatenation + fractal blending
5. **Frequency-domain blending** at multiple alpha values
6. **Phase 1**: Alpha ablation study
7. **Phase 2**: Final 4-way comparison with train/val/test split

---

## Evolution of the Pipeline

| Version | Location | Key Addition |
|---------|----------|-------------|
| V1 | `Models/pipeline_wo_checkpoints.ipynb` | Basic unsupervised triplet learning |
| V2 | `Models/pipeline_w_checkpoints.ipynb` | Checkpoint saving for model selection |
| V3 | `Models/pipeline_5DR_skewed dataset.ipynb` | 5-class supervised grading |
| V4 | `Models/pipeline_frequency_blending.ipynb` | FFT-based blending |
| V5 | `Models/pipeline_train_val_test.ipynb` | Train/val/test split |
| V6 | `pipeline.ipynb` | Online semi-hard negative mining + all improvements |

---

## References

- **DiffuseMix**: Khandoker et al., "DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models" (2024)
- **Texture Bias**: Geirhos et al., "ImageNet-trained CNNs are biased towards textures" (ICLR 2019)
- **Messidor-2**: Decenciere et al., "Feedback on a publicly distributed database: the Messidor database" (2014)
- **Neural Style Transfer**: Gatys et al., "A Neural Algorithm of Artistic Style" (2015)
- **Triplet Loss**: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering" (CVPR 2015)
- **Semi-Hard Mining**: Schroff et al., FaceNet (2015) — online triplet selection strategy
