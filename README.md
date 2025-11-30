# Navier-Stokes – Physics-Informed and CFD (Multi-Fidelity Neural Network)

This repository explores **data-driven modeling for CFD / Navier–Stokes problems** using a **multi-fidelity neural network (MFNN)**.  
The goal is to learn an accurate surrogate of a high-fidelity CFD solution by combining:

- Cheap **low-fidelity** data (coarse simulations / simplified models), and  
- Expensive **high-fidelity** data (fine CFD / reference solution).

The approach is inspired by **physics-informed** and **multi-fidelity** learning strategies frequently used in fluid dynamics and scientific machine learning.

---

## 1. Algorithms Implemented

### 1.1 Multi-Fidelity Neural Network (MFNN)

The core of the repository is `MFNN.py`, which implements a **multi-fidelity regression model** using a feed-forward neural network (PyTorch).

The idea is to learn a mapping
\[
x \mapsto y_{\text{low}}(x), \quad x \mapsto y_{\text{high}}(x)
\]
where:

- \( y_{\text{low}} \) comes from **low-fidelity data** (cheap approximation of the solution),
- \( y_{\text{high}} \) comes from **high-fidelity data** (expensive CFD / reference solution).

The MFNN uses both datasets jointly during training.

#### a) Network architecture

Conceptually, the model can be seen as:

1. A **shared feature extractor**:
   - Fully-connected layers with non-linear activations
   - Encodes the input \( x \) into a latent representation \( z(x) \).

2. A **low-fidelity head**:
   - Takes \( z(x) \) and outputs a prediction \(\hat{y}_{\text{low}}(x)\).

3. A **high-fidelity head**:
   - Uses \( z(x) \) *and* information about \( \hat{y}_{\text{low}}(x) \) (either explicitly or implicitly through shared weights)  
   - Outputs \(\hat{y}_{\text{high}}(x)\), which should approximate the true high-fidelity solution.

This structure allows the network to **reuse what it learns from cheap low-fidelity data** to improve the prediction of the expensive high-fidelity solution.

#### b) Multi-fidelity loss function

The loss combines the errors on both levels:

\[
\mathcal{L} = \lambda_{\text{low}} \, \text{MSE}\big( \hat{y}_{\text{low}}, y_{\text{low}} \big)
            + \lambda_{\text{high}} \, \text{MSE}\big( \hat{y}_{\text{high}}, y_{\text{high}} \big)
\]

- \( \lambda_{\text{low}}, \lambda_{\text{high}} \) weight the contribution of each fidelity level.
- This encourages the network to:
  - Learn a **good approximation of the low-fidelity model**, and
  - Correct / refine it to match the **high-fidelity target**.

Depending on the experiment, the loss can also be extended with:

- Regularization terms (weight decay, L2 on parameters),
- Physics-inspired penalties (e.g. enforcing smoothness or known constraints),
- Test errors on unseen points (`y_test.dat`) for evaluation.

#### c) Optimization

Training typically uses standard **stochastic gradient-based optimization**, e.g.:

- Optimizer: Adam / SGD with momentum,
- Mini-batch training on mixed low- and high-fidelity samples,
- Learning rate scheduling (optional).

The training loop iterates over the data until convergence and saves the trained weights in `model.pth`.

---

## 2. Data Handling

The repository includes several data files:

- `y_l.dat` : low-fidelity data (inputs / outputs),
- `y_h.dat` : high-fidelity data,
- `y_test.dat` : test data for evaluating generalization,
- `mfdata.mat`, `mfdata2.mat`, `Copy_of_mfdata.mat` : MATLAB data files containing low- and high-fidelity datasets.

Typical workflow:

1. **Load the data** from `.dat` or `.mat` files.
2. **Split** into training and test sets.
3. **Normalize / scale** inputs and outputs (if needed).
4. **Train** the MFNN on low- and high-fidelity data.
5. **Evaluate** on `y_test.dat` and compare predictions to the true function.

---

## 3. Visualization

To analyze the performance of the MFNN, the repository contains several plots:

- `true function.png`, `true_500pts.jpg` – reference / “true” target function,
- `prediction.png` – MFNN prediction after training,
- `low.jpg`, `low_500pts.jpg` – low-fidelity solution,
- `high.jpg`, `high_500pts.jpg` – high-fidelity solution,
- `150000epoch_500low_10high.png`, `Our.png` – training results and comparison between methods.

These figures illustrate:

- How the low-fidelity model deviates from the true function,
- How the high-fidelity data corrects it,
- How the MFNN combines both to approximate the true solution.

---

## 4. Repository Structure

```text
.
├── MFNN.py                # Multi-Fidelity Neural Network implementation (PyTorch)
├── model.pth              # Trained model weights
├── y_l.dat                # Low-fidelity dataset
├── y_h.dat                # High-fidelity dataset
├── y_test.dat             # Test dataset
├── mfdata.mat             # MATLAB multi-fidelity data
├── mfdata2.mat
├── Copy_of_mfdata.mat
├── true function.png      # Reference “true” function
├── prediction.png         # MFNN predictions
├── low.jpg / high.jpg     # Low- and high-fidelity solutions
├── *_500pts.jpg           # Same, sampled on 500 points
├── 150000epoch_500low_10high.png
├── Our.png
└── README.md              # This file
