# Navier–Stokes — Physics-Informed & Multi-Fidelity Neural Networks

This repository explores **data-driven modeling for Navier–Stokes / CFD** using a **Multi-Fidelity Neural Network (MFNN)**.

The goal is to combine **cheap, low-fidelity CFD data** with **expensive, high-fidelity data** to learn an accurate surrogate model — improving prediction quality while reducing simulation cost.

## 🔍 Motivation

Classical CFD solvers are accurate but computationally expensive.  
Low-fidelity solvers are faster, but introduce bias.

👉 A **Multi-Fidelity Neural Network** leverages *both*:

- **Low-fidelity data** → captures global structure  
- **High-fidelity data** → corrects local errors  

This approach is highly relevant for **surrogate modeling, optimization, digital twins, and engineering design**.

## 🎯 Objective

We learn two mappings from inputs \(x\):

$$
x \mapsto \hat{y}_{\mathrm{low}}(x)
$$

$$
x \mapsto \hat{y}_{\mathrm{high}}(x)
$$

The high-fidelity network **reuses information** from the low-fidelity model.

## 🧠 Model Architecture

The MFNN contains three main components:

### 1️⃣ Shared Feature Extractor
Transforms the input into a latent space:
$$
z(x)=f_{\theta}(x)
$$


### 2️⃣ Low-Fidelity Head
Predicts the coarse approximation:ù

$$
\hat{y}_{\mathrm{low}} = g_{\theta_{\mathrm{low}}}\!\left(z\right)
$$

### 3️⃣ High-Fidelity Correction Head
Refines the prediction using both the latent space and the LF estimate:

$$
\hat{y}_{\mathrm{high}} = h_{\theta_{\mathrm{high}}}\!\left(z,\,\hat{y}_{\mathrm{low}}\right)
$$

This sharing mechanism is what makes the network *multi-fidelity*.

## 📉 Loss Function

The network is trained using a weighted MSE loss:

$$
\mathcal{L} = \lambda_{\mathrm{low}}\, \mathrm{MSE}\!\left(\hat{y}_{\mathrm{low}},\, y_{\mathrm{low}}\right) + \lambda_{\mathrm{high}}\,  \mathrm{MSE}\!\left(\hat{y}_{\mathrm{high}},\,y_{\mathrm{high}}\right)
$$

Where:

- \( \lambda_{\text{low}} \) — weight of LF data  
- \( \lambda_{\text{high}} \) — weight of HF data  

Training proceeds such that the model first learns a **good low-fidelity approximation**, and then **refines it** with high-fidelity supervision.

## 📂 Dataset

Included files:

- `y_l.dat` — Low-fidelity data  
- `y_h.dat` — High-fidelity data  
- `y_test.dat` — Test dataset  
- `mfdata.mat`, `mfdata2.mat` — MATLAB multi-fidelity datasets  

Typical workflow:

1. Load datasets  
2. (Optional) Normalize inputs/outputs  
3. Train MFNN  
4. Evaluate against test data  

---

## 📊 Visualizations

The repository includes figures comparing:

✔ Low-fidelity vs. high-fidelity  
✔ MFNN predictions vs. ground truth  
✔ Error distributions  
✔ Training evolution  

These illustrate how the network progressively improves from:

**Low-Fidelity → High-Fidelity → Multi-Fidelity**

> If you run the notebooks/scripts, figures are automatically generated.

---

## 📁 Repository Structure

