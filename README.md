# GlobalAI-DeepLearning-2025

## Car Brand Classification — Custom CNN (from scratch) vs. Transfer Baselines

### Goal
Classify 33 car brands from images using a custom CNN.  
Document not only results but also **failures** and **lessons learned**.

---

### Dataset
- Source: *Car Brand Classification Dataset* (Kaggle).  
- Balanced: ~349 train, 75 val, 75 test per class.  
- Splits provided in folders (`train/val/test`).  
- Challenge: logos are often **small or absent**, making this a fine-grained recognition task.

---

### Approach

#### Custom CNN (from scratch)
- 3 Conv blocks (32→64→128), GAP, Dropout, FC(256→33).
- No pretrained weights; trained from scratch.
- Optimizer: AdamW (lr=3e-3, wd=5e-5).
- Loss: CrossEntropy + label smoothing 0.05.
- Augment: Resize→(Random|Center)Crop(224), HFlip, mild ColorJitter.

#### HPO
Random search across filters, kernel, dense units, dropout, lr, batch size, optimizer.  
Best configs still collapsed.

#### BatchNorm Variant
A SmallCarCNN_BN was tested — did not improve meaningfully; some runs worse.

---

### Results

- **Validation/Test Accuracy:** ~0.03 (random baseline).  
- **Confusion Matrix:** almost all predictions collapsed into one class.  
- **Grad-CAM:** heatmaps often ignored logos, focusing on background.

Screenshots:  
- `assets/confusion_matrix.png` (collapsed predictions)  
- `assets/gradcam.png` (logo ignored)

---

### Analysis — Why Collapse?

1. **Model capacity too small**: 32–128 conv filters cannot capture fine brand cues.  
2. **No BatchNorm (baseline)**: unstable gradients → collapse.  
3. **Aggressive augmentations**: initial RandomResizedCrop cut out logos.  
4. **Learning dynamics**: ReduceLROnPlateau shrank LR too early, halting learning.  
5. **Task difficulty**: fine-grained brand recognition, small cues.

---

### Comparison with Transfer Baseline
Pretrained models (ReXNet-150, EfficientNet-B2) on the same dataset achieve **~0.69 test accuracy** with balanced confusion matrices.  
This highlights the gap between **from-scratch CNNs** vs. **transfer learning**.

---

### Lessons Learned

- From-scratch small CNNs fail to train on fine-grained datasets.  
- Transfer learning is crucial when discriminative features (logos) are small and subtle.  
- Sanity tests like “overfit 32 images” are essential before full training.  
- Augmentation must preserve key signals (logos).  

---

### Next Steps

- Use pretrained models with transfer learning.  
- Increase resolution to 256/288.  
- Add BatchNorm/SE blocks, deeper filters.  
- Modern schedulers (Cosine, OneCycle).  
- Mixup/CutMix cautiously.

---

### Reproduction

- Kaggle Notebook: [https://www.kaggle.com/code/altnzengi/car-brand-classification-with-custom-cnn]  
- Run order: Env → Paths → Transforms → EDA → Model → Training → Save → Eval → Grad-CAM.  
- Outputs: confusion matrices, Grad-CAM visualizations.

---

## License
Educational purpose only. Dataset: Kaggle Car Brand Classification.
