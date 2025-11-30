# Navier–Stokes — Physics-Informed & Multi-Fidelity Neural Networks

This repository explores **data-driven modeling for Navier–Stokes / CFD** using a **Multi-Fidelity Neural Network (MFNN)**.  
The objective is to combine **low-fidelity** and **high-fidelity** data to learn an accurate surrogate of a CFD solution.

---

## 1. Objective

The MFNN learns the mappings:

$$
x \mapsto \hat{y}_{\text{low}}(x)
$$

$$
x \mapsto \hat{y}_{\text{high}}(x)
$$

---

## 2. Network Architecture

The MFNN consists of three components.

### 2.1 Shared feature extractor
Produces a latent representation:

$$
z(x) = f_{\theta}(x)
$$

### 2.2 Low-fidelity head

$$
\hat{y}_{\text{low}} = g_{\theta_{\text{low}}}(z)
$$

### 2.3 High-fidelity correction head

$$
\hat{y}_{\text{high}} = h_{\theta_{\text{high}}}(z,\, \hat{y}_{\text{low}})
$$

The high-fidelity branch reuses information learned from the low-fidelity data.

---

## 3. Multi-fidelity Loss Function

The total loss is the weighted sum:

$$
\mathcal{L}
= \lambda_{\text{low}}\,\mathrm{MSE}(\hat{y}_{\text{low}},\, y_{\text{low}})
+ \lambda_{\text{high}}\,\mathrm{MSE}(\hat{y}_{\text{high}},\, y_{\text{high}})
$$

Where:

- \( \lambda_{\text{low}} \) and \( \lambda_{\text{high}} \) control the influence of each fidelity level.  
- The model learns a **good low-fidelity approximation**, then **refines it** using high-fidelity data.

---

## 4. Data Included

- `y_l.dat` — Low-fidelity data  
- `y_h.dat` — High-fidelity data  
- `y_test.dat` — Test data  
- `mfdata.mat`, `mfdata2.mat` — MATLAB datasets  

Workflow:

1. Load datasets  
2. Normalize input/output (optional)  
3. Train MFNN  
4. Evaluate on test data  

---

## 5. Visualizations

The repository contains figures:

- low-fidelity vs high-fidelity  
- MFNN predictions  
- true function reference  
- training evolution  

These illustrate the refinement from LF → HF → MFNN.

---

## 6. Repository Structure

