# Special Topics on Optimal Control and Learning — Fall 2025 (ISYE 8803 VAN)
*Georgia Institute of Technology – Fridays 2 pm ET*

**Designers:** Andrew Rosemberg & Michael Klamkin  
**Instructor:** Prof. Pascal Van Hentenryck

---

## Overview
This student-led course explores modern techniques for controlling — and learning to control — dynamical systems. Topics range from classical optimal control and numerical optimization to reinforcement learning, PDE-constrained optimization (finite-element methods, Neural DiffEq, PINNs, neural operators), and GPU-accelerated workflows.

## Prerequisites
* Solid linear-algebra background  
* Programming experience in Julia, Python, *or* MATLAB  
* Basic ODE familiarity  

## Grading
| Component | Weight |
|-----------|--------|
| Participation & paper critiques | **25 %** |
| In-class presentations | **50 %** |
| Projects | **25 %** |

## Weekly Schedule (Fall 2025 – Fridays 2 p.m. ET)

| #  | Date (MM/DD) | Format / Presenter | Topic & Learning Goals | Prep / Key Resources |
|----|--------------|--------------------|------------------------|----------------------|
| 1  | 08/22/2025   | Lecture — Andrew Rosemberg | Course map; why PDE-constrained **optimization**; tooling overview; stability & state-space dynamics; Lyapunov; discretization issues | |
| 2  | 08/29/2025   | Lecture | Numerical **optimization** for control (grad/SQP/QP); ALM vs. interior-point vs. penalty methods | |
| 3  | 09/05/2025   | Lecture | Pontryagin’s Maximum Principle; shooting & multiple shooting; LQR, Riccati, QP viewpoint (finite / infinite horizon) | |
| 4  | 09/12/2025   | Lecture | Dynamic Programming & Model-Predictive Control | |
| 5  | 09/19/2025   | Lecture | **Nonlinear** trajectory **optimization**; collocation; implicit integration | |
| 6  | 09/26/2025   | **External seminar 1** - Henrique Ferrolho | Trajectory **optimization** on robots in Julia Robotics | |
| 7  | 10/03/2025   | Lecture | Essentials of PDEs for control engineers; weak forms; FEM/FDM review | |
| 8  | 10/10/2025   | **External seminar 2** TBD (speaker to be confirmed) | Topology **optimization** | |
| 9  | 10/17/2025   | **External seminar 3 — François Pacaud** | GPU-accelerated optimal control | |
|10  | 10/24/2025   | Lecture - Michael Klamkin | Physics-Informed Neural Networks (PINNs): formulation & pitfalls | |
|11  | 10/31/2025   | **External seminar 4** - Chris Rackauckas | Neural Differential Equations: PINNs + classical solvers | |
|12  | 11/07/2025   | Lecture | Neural operators (FNO, Galerkin Transformer); large-scale surrogates | |
|13  | 11/14/2025   | **External seminar 5** - Charlelie Laurent | Scalable PINNs / neural operators; CFD & weather applications | |
|14  | 11/21/2025   | Lecture | Robust control & min-max DDP (incl. PDE cases); chance constraints; Data-driven control & RL-in-the-loop | |

## Reference Material

- https://optimalcontrol.ri.cmu.edu/
- https://github.com/SciML/SciMLBook
- https://underactuated.mit.edu/
- https://www.math.lmu.de/~philip/publications/lectureNotes/philipPeter_OptimalControlOfPDE.pdf
- https://castle.princeton.edu/sda/
- https://www.mit.edu/~dimitrib/dpbook.html

---

*Repository maintained by the 2025 cohort.*  
Feel free to open issues or pull requests for corrections and improvements.
