# Droplet Pinch-Off Machine Learning Framework

This repository contains the code used to perform dimensionality reduction, supervised learning, and clustering on high-speed images of droplet pinch-off.
The workflow consists of four major components:

1. **Autoencoder** for extracting low-dimensional representations of droplet contours.
2. **MLP models** for predicting liquid properties from latent variables.
3. **XGBoost models** for comparison with neural networks.
4. **Unsupervised clustering** for analyzing morphological families.

---

## **1. Autoencoder: Extracting Low-Dimensional Representations**

To begin, run:

```
Autoencoder.py
```

This script trains a convolutional autoencoder (CAE) on the droplet silhouettes and produces a **low-dimensional latent representation** (typically 14D) stored for downstream tasks.

Output includes:

* `latent_vectors.npy`
* reconstructed contour images

These serve as inputs for all supervised and unsupervised models.

---

## **2. Supervised Learning with MLPs**

After obtaining the latent representation, train the MLP models by running:

```
train_model1_viscosity.py
train_model2_tension.py
train_model3_tension&viscosity.py
train_model4_figure.py
```

Descriptions:

| Script                              | Prediction Target                              |
| ----------------------------------- | ---------------------------------------------- |
| `train_model1_viscosity.py`         | viscosity (μ)                                  |
| `train_model2_tension.py`           | surface tension (σ)                            |
| `train_model3_tension&viscosity.py` | both μ and σ simultaneously                    |
| `train_model4_figure.py`            | predictions for figure generation & comparison |

Each script loads the latent vectors from the autoencoder and fits an MLP regression model.

---

## **3. Supervised Learning with XGBoost**

To compare results with tree-based methods, run:

```
model1-XG.py
model2-XG.py
model3-XG.py
model4-XG.py
```

These scripts train XGBoost regressors corresponding to the MLP models above, using the same latent representations as input.

---

## **4. Unsupervised Clustering**

To explore morphological structure in the latent space, run:

```
unsupervised.py
```

This script performs clustering (KMeans and/or GMM) on the latent vectors, identifies shape families, and visualizes representative drop contours.

---

## Repository Structure

```
├── Autoencoder.py               # Train autoencoder for latent contour embedding
├── README.md                    # Project documentation
├── model1-XG.py                 # XGBoost: viscosity
├── model2-XG.py                 # XGBoost: surface tension
├── model3-XG.py                 # XGBoost: both viscosity & tension
├── model4-XG.py                 # XGBoost models for figure generation
├── train_model1_viscosity.py    # MLP: viscosity
├── train_model2_tension.py      # MLP: surface tension
├── train_model3_tension&viscosity.py
├── train_model4_figure.py       # MLP for figure/demo
├── unsupervised.py              # clustering analysis (KMeans/GMM)
├── blueonly                     # Contours for droplet shape
├── data.xlsx                    # Liquid property and experimental configuration data
```

---

## **Usage Summary**

1. **Run Autoencoder.py**
   → obtain latent representations of droplet contours.

2. **Train MLP models**

   ```
   train_model1_viscosity.py
   train_model2_tension.py
   train_model3_tension&viscosity.py
   train_model4_figure.py
   ```

3. **Train XGBoost models**

   ```
   model1-XG.py
   model2-XG.py
   model3-XG.py
   model4-XG.py
   ```

4. **Perform clustering analysis**

   ```
   unsupervised.py
   ```

