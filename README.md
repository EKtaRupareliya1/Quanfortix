# Tensor-Network Motivated Quantum Simulation of Tight-Binding Hamiltonians

## Overview
This repository presents a hybrid classical–quantum workflow for simulating many-body lattice Hamiltonians using tensor-network intuition and quantum algorithms. The project combines exact diagonalization (QuTiP) with variational quantum eigensolvers (Qiskit) to analyze ground-state properties of a 1D tight-binding fermionic chain.

The pipeline integrates:
Physics → Entanglement Structure → Tensor Compression → Quantum Simulation

---

## Objectives
- Construct tight-binding Hamiltonians
- Compute exact ground states using QuTiP
- Analyze entanglement entropy (area law behavior)
- Study correlation decay and locality
- Map fermionic operators to qubits (Jordan–Wigner)
- Implement VQE for ground-state estimation
- Benchmark quantum results against exact solutions

---

## Physics Model

The 1D tight-binding Hamiltonian:

\[
H = -t \sum_i (c_i^\dagger c_{i+1} + \text{h.c.})
\]

This system:
- exhibits short-range correlations
- obeys area-law entanglement
- admits efficient tensor-network representations (MPS/MPO)

---

## Methods

### Classical Simulation (QuTiP)
- Exact diagonalization
- Ground-state energy computation
- Reduced density matrices
- Von Neumann entropy
- Correlation functions

### Quantum Simulation (Qiskit)
- Jordan–Wigner fermion-to-qubit mapping
- Pauli Hamiltonian construction
- Variational Quantum Eigensolver (VQE)
- Circuit depth convergence studies

---

## Results

### Entanglement entropy
![Entropy](results/entropy_plot.png)

### Correlation decay
![Correlation](results/correlation_plot.png)

### VQE convergence
![VQE](results/vqe_convergence.png)

Increasing circuit depth systematically improves energy estimates toward the exact solution.

---

## Installation

```bash
pip install -r requirements.txt
