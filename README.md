# Chest X-Ray Classification with CNNs

A proof-of-concept multi-class classifier distinguishing **COVID-19**, **Pneumonia**, and **Normal** chest X-rays, built with TensorFlow/Keras. The project is split into two stages: a lightweight CPU run for architecture validation, and a full GPU run on Google Colab for actual training.

---

## Dataset

[Chest X-Ray (COVID-19 & Pneumonia)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia) — Kaggle, via Prashant268.

| Class | Train | Test |
|-----------|-------|------|
| COVID-19 | 460 | 116 |
| Normal | 1266 | 317 |
| Pneumonia | 4273 | 855 |

The COVID-19 class is roughly **3× underrepresented** relative to Pneumonia. This is the single largest factor driving model performance and is discussed further under [Limitations](#limitations).

---

## Project Structure

```
Fun_With_CNNs_4_CPUEd_1.ipynb   # Main notebook (CPU architecture check + GPU training)
```

---

## Setup

**Kaggle credentials** — the notebook uses the Colab file upload approach to avoid hardcoding credentials:

```python
from google.colab import files
import os

files.upload()  # upload your kaggle.json when prompted

os.makedirs('/root/.kaggle', exist_ok=True)
!cp kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
```

Download your `kaggle.json` from kaggle.com → Settings → API → Create New Token.

**Dependencies:** TensorFlow, Keras, scikit-learn, NumPy, pandas, Matplotlib, seaborn.

---

## Approach

### Stage 1 — CPU Architecture Check (5 epochs, 64×64)

A smaller version of the network (`createModel`) trained for 5 epochs at 64×64 resolution on CPU. The purpose here is purely to validate that the architecture runs end-to-end before committing GPU time — loss curves and confusion matrix are included for completeness but not the focus.

Architecture: two conv blocks (32→64 filters) with BatchNorm, ReLU, MaxPooling and Dropout, followed by a 256-unit dense head.

### Stage 2 — GPU Training (20 epochs, 224×224)

The main model (`createModel1`) trained on Google Colab at full 224×224 resolution for up to 20 epochs, with `ReduceLROnPlateau` (factor 0.5, patience 3) and `EarlyStopping` (patience 5, best weights restored).

Architecture: four conv blocks (32→64→128→256 filters) with BatchNorm, ReLU, MaxPooling and Dropout, replacing the final `Flatten` + Dense with `GlobalAveragePooling2D` for better generalisation.

**Class imbalance handling:** balanced class weights computed via `sklearn.utils.class_weight.compute_class_weight` and passed to `model.fit`.

**Data augmentation (training only):** rotation (±15°), zoom (15%), width/height shift (15%), horizontal flip, shear (10%).

---

## Results (GPU Run)

| Class | Precision | Recall | F1 | Support |
|-----------|-----------|--------|----|---------|
| COVID-19 | 0.14 | 0.14 | 0.14 | 116 |
| Normal | 0.23 | 0.19 | 0.21 | 317 |
| Pneumonia | 0.67 | 0.71 | 0.69 | 855 |
| **Overall accuracy** | | | **0.53** | 1288 |
| Macro avg | 0.35 | 0.35 | 0.35 | |
| Weighted avg | 0.51 | 0.53 | 0.52 | |

The model performs reasonably on Pneumonia (F1 0.69) but struggles with COVID-19 and Normal — see Limitations.

---

## Limitations

This is intentionally a proof of concept, not a production-grade medical classifier.

- **Class imbalance.** The COVID-19 training set (~460 images) is about 3× smaller than the Pneumonia set (~4273). Class weighting partially compensates, but with this little COVID-19 data the model has limited ability to learn discriminative features for that class. The low COVID-19 F1 (0.14) reflects this directly.
- **No transfer learning.** The network is trained from scratch. A pretrained backbone (e.g. ResNet, EfficientNet) would almost certainly improve performance, especially on the minority class.
- **Image resolution.** 224×224 grayscale-to-RGB is a reasonable baseline but clinical X-ray analysis typically operates at higher resolution.
- A companion notebook covering **binary classification (Sick vs Normal)** is also available in this repository, which sidesteps the multi-class imbalance problem entirely and achieves stronger results.

---

## Potential Next Steps

- Transfer learning with a pretrained backbone
- Oversampling / SMOTE for the COVID-19 class
- Grad-CAM visualisations to inspect what regions the model attends to
- Threshold tuning per class given the clinical cost asymmetry between false negatives
