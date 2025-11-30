# Navier–Stokes — Physics-Informed & Multi-Fidelity Neural Networks

This repository explores **data-driven modeling for Navier–Stokes / CFD** using a **Multi-Fidelity Neural Network (MFNN)**.  
The objective is to learn an accurate approximation of a **high-fidelity CFD solution** by combining:

- **Low-fidelity** data (coarse solver, simplified physics),
- **High-fidelity** data (reference solution),
- Neural networks capable of fusing both.

This approach is inspired by modern **physics-informed** and **multi-fidelity** strategies used in scientific machine learning.

---

## 🚀 1. Algorithms Implemented

### 1.1 Multi-Fidelity Neural Network (MFNN)

The MFNN implemented in `MFNN.py` is trained on:

- low-fidelity data \( (x, y_{\text{low}}) \)
- high-fidelity data \( (x, y_{\text{high}}) \)

It learns to approximate:

$$
x \;\mapsto\; \hat{y}_{\text{low}}(x), \qquad
x \;\mapsto\; \hat{y}_{\text{high}}(x)
$$

---

### 🧠 a) Network Architecture

The MFNN consists of:

1. **Shared feature extractor**  
   Produces a latent representation:
   $$
   z(x) = f_{\theta}(x)
   $$

2. **Low-fidelity head**
   $$
   \hat{y}_{\text{low}} = g_{\theta_{\text{low}}}(z)
   $$

3. **High-fidelity correction head**
   $$
   \hat{y}_{\text{high}} = h_{\theta_{\text{high}}}(z,\, \hat{y}_{\text{low}})
   $$

The high-fidelity branch reuses learned information from the low-fidelity approximation.

---

### 📉 b) Multi-fidelity loss function

The total loss is the weighted sum of both fidelity levels:

$$
\mathcal{L}
= \lambda_{\text{low}}\,\mathrm{MSE}\!\left(\hat{y}_{\text{low}},\, y_{\text{low}}\right)
+ \lambda_{\text{high}}\,\mathrm{MSE}\!\left(\hat{y}_{\text{high}},\, y_{\text{high}}\right)
$$

- \( \lambda_{\text{low}}, \lambda_{\text{high}} \) weight the influence of each fidelity level.  
- The network learns a **good low-fidelity model**, then **corrects** it using high-fidelity data.

---

### ⚙️ c) Optimization

Training uses:

- **Adam optimizer**
- Combined batches of low- and high-fidelity samples
- Optional learning rate scheduling
- Model saved as `model.pth`

---

## 📂 2. Data Structure

The repository contains:

- `y_l.dat` → Low-fidelity data  
- `y_h.dat` → High-fidelity data  
- `y_test.dat` → Test set  
- `mfdata.mat`, `mfdata2.mat`, `Copy_of_mfdata.mat` → MATLAB datasets

Workflow:

1. Load data  
2. Normalize inputs/outputs  
3. Train MFNN  
4. Evaluate on `y_test.dat`  
5. Compare low, high, and MFNN predictions  

---

## 📊 3. Visualizations

Included plots:

- **True function** (`true function.png`)
- **Low-fidelity model** (`low.jpg`)
- **High-fidelity model** (`high.jpg`)
- **MFNN predictions** (`prediction.png`)
- Training history / comparisons (`150000epoch_500low_10high.png`, `Our.png`)

These show how the MFNN refines the low-fidelity approximation.

---

## 📁 4. Repository Structure

