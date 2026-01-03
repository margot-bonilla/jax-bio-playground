# jax-bio-playground 🧬

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![JAX](https://img.shields.io/badge/JAX-Accelerated-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**A JAX/Flax implementation of E(3)-Invariant Graph Neural Networks for Protein Structure Analysis.**

This repository serves as a research engineering sandbox to demonstrate robust implementation of Geometric Deep Learning concepts. It implements a custom Message Passing Neural Network (MPNN) capable of processing 3D molecular point clouds while respecting physical symmetries (Rotation and Translation invariance).

---

## 🚀 Key Features

* **Pure JAX/Flax Implementation:** Custom layers written from scratch using `flax.linen` and `jax.vmap` for efficient batching.
* **E(3) Invariance:** The model architecture is designed to be invariant to 3D rotations and translations—a critical requirement for accurate molecular property prediction.
* **Rigorous Testing:** Includes a specific test suite (`tests/test_invariance.py`) that mathematically verifies the model's symmetry properties to floating-point tolerance.
* **Mock Data Pipeline:** Generates synthetic 3D protein structures (node features + coordinates) for rapid prototyping without heavy PDB parsing overhead.

---

## 📂 Project Structure

```text
jax-bio-playground/
├── src/
│   ├── __init__.py
│   ├── data.py             # Synthetic 3D graph generation (Nodes, Coords, Edges)
│   ├── layers.py           # Custom GNN layers (Message Passing & Aggregation)
│   ├── model.py            # End-to-end Flax Model definition
│   └── train.py            # JIT-compiled training loop (using Optax)
├── tests/
│   ├── test_layers.py      # Shape and gradient checks
│   └── test_invariance.py  # 🔥 The Rotation Invariance verification
├── requirements.txt
└── README.md
