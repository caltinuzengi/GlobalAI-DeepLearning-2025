# GlobalAI-DeepLearning-2025

## Car Brand Classification — Custom CNN (from scratch) vs. Transfer Baselines

### 🎯 Goal
Classify **33 car brands** from images.  
Report not only working models but also **failures, root-cause analysis, and transfer learning baselines**.

---

## 📂 Dataset
- Source: *Car Brand Classification Dataset* (Kaggle).  
- Balanced: ~349 train, 75 validation, 75 test per class.  
- Structured in `train/`, `val/`, `test/`.  
- **Challenge factors:**
  - Logos are often **small, occluded, or absent**.  
  - Mixed viewpoints (front, side, rear, interior).  
  - A **fine-grained recognition** problem.

---

## 🧪 Approach

### 1. Custom CNN (from scratch)
- **Architecture**:  
  - Conv blocks (32 → 64 → 128 filters)  
  - ReLU activations  
  - MaxPooling  
  - Global Average Pooling  
  - Dropout  
  - Dense (256 → 33 outputs)  
- **Training setup**:  
  - Optimizer: AdamW (`lr=3e-3`, `wd=5e-5`)  
  - Loss: CrossEntropy + Label Smoothing (0.05)  
  - Scheduler: ReduceLROnPlateau  
  - Augmentation: Resize(256) → Random/CenterCrop(224), HFlip, mild ColorJitter  

### 2. Hyperparameter Optimization
- Random search over filters, kernel sizes, dense units, dropout, learning rate, batch size, optimizer.  
- Best configs did not escape **collapse (~0.03 accuracy)**.

### 3. BatchNorm Variant
- Added BatchNorm layers (SmallCarCNN_BN).  
- Training curves more stable but still collapsed.  

### 4. Transfer Learning Baseline
- **Model**: EfficientNet-B0 (`timm`, pretrained on ImageNet).  
- **Training setup**:  
  - AdamW (`lr=3e-4`, `wd=1e-4`)  
  - CosineAnnealingLR (T_max=8)  
  - 8 epochs (2-epoch frozen backbone, then fine-tuned)  
- **Augmentation**: same as custom CNN.  

---

## 📊 Results

### Custom CNN (from scratch)
- Validation/Test Accuracy: **~0.03** (random baseline).  
- Loss stuck near log(33) ≈ 3.49.  
- Confusion Matrix: collapsed into a single class.  
- Grad-CAM: irrelevant background focus.  

### Transfer Learning (EfficientNet-B0)
- **Val Accuracy**: 0.665  
- **Test Accuracy**: 0.667  
- **Macro F1**: 0.665  
- **Per-class**: Many brands reach 0.70–0.83 F1 (e.g. FIAT, GMC, Jeep, Ram).  
- Confusion Matrix: balanced diagonal, errors among visually similar brands (BMW vs Mercedes, Lexus vs Toyota).  
- Grad-CAM:  
  - Correct=1651, Wrong=824  
  - Heatmaps focus on **logos and front grilles**, confirming meaningful features.  

---

## ⚖️ Comparison

| Model                | Val Acc | Test Acc | CM Pattern          | Grad-CAM Focus        |
|----------------------|---------|----------|---------------------|-----------------------|
| SmallCarCNN          | ~0.03   | ~0.03    | Collapsed           | Background / random   |
| SmallCarCNN + BN     | ~0.03   | ~0.03    | Still collapsed     | Background            |
| EfficientNet-B0 (TL) | 0.665   | 0.667    | Balanced diagonal   | Logos / grilles       |

---

## 🔍 Analysis

### Why collapse in custom CNN?
1. **Model too small** (32–128 filters).  
2. **No pretrained features** for fine-grained cues.  
3. **Aggressive crops** initially cut logos out.  
4. **Scheduler** reduced LR too early, reinforcing stagnation.  
5. **Fine-grained nature**: logos too small for scratch CNN.

### Why TL succeeds?
- Pretrained filters already capture **edges, textures, logo shapes**.  
- Fine-tuning aligns them to car-specific patterns.  
- Larger backbone capacity (~4M params) enables discriminative learning.

---

## 🧠 Lessons Learned
- From-scratch CNNs fail on fine-grained datasets.  
- Transfer learning yields >0.66 accuracy with moderate training.  
- **Grad-CAM** is essential to verify feature focus (logos vs irrelevant).  
- Balanced datasets don’t guarantee learning without strong architectures.  
- **Reproducibility**: Documenting collapse is as important as success.

---

## 🚀 Next Steps
- Use larger models (EfficientNet-B2/B3, ConvNeXt).  
- Increase resolution (256–288) to capture small logos.  
- Progressive unfreezing + OneCycleLR.  
- Regularization: Mixup/CutMix with care.  
- Monitor with TensorBoard / Weights & Biases.  

---

## 📝 Reproduction
- Kaggle Notebook: [Car Brand Classification with Custom CNN](https://www.kaggle.com/code/altnzengi/car-brand-classification-with-custom-cnn)  
- Run order: Env → Paths → Transforms → EDA → CNN → HPO → BN Variant → Training → Evaluation → Grad-CAM → Transfer Learning.  
- Outputs: training curves, confusion matrices, Grad-CAM overlays.  

---

## 📜 License
Educational purpose only.  
Dataset: Kaggle Car Brand Classification.  
