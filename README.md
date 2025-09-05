[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://learningtooptimize.github.io/LearningToControlClass/dev/)

# Special Topics on Optimal Control and Learning — Fall 2025 (ISYE 8803 VAN)
*Georgia Institute of Technology – Fridays 2-4:45 pm ET*

**Designers:** Andrew Rosemberg & Michael Klamkin  
**Instructor:** Prof. Pascal Van Hentenryck

---

## Overview
This student-led course explores modern techniques for controlling — and learning to control — dynamical systems. Topics range from classical optimal control and numerical optimization to reinforcement learning, PDE-constrained optimization (finite-element methods, Neural DiffEq, PINNs, neural operators), and GPU-accelerated workflows.

## Objective
Create an online book at the end using the materials from all lectures.

## Prerequisites
* Solid linear-algebra background  
* Programming experience in Julia, Python, *or* MATLAB  
* Basic ODE familiarity  

## Grading
| Component | Weight |
|-----------|--------|
| Participation | **25 %** |
| In-class Presentations and Chapter | **50 %** |
| Projects (Liaison work & Scribe & Admin & ...) | **25 %** |

**Class material is due one week before the lecture!** No exceptions apart from the first 2 lectures.

**Issues outlining references that will be used for lecture preparation are due at the end of the 3rd week (10/05/2025)!** 
20 minutes of research should give you an initial idea of what you need to read.

🎯🚲 **Guessing Game**

Here’s how the presentation grading works: we already know the lecture content we expect from you. Any deviations will be penalized **exponentially**. Your mission is twofold:  
1. **Check your understanding** — use [discussions](https://github.com/LearningToOptimize/LearningToControlClass/discussions) from previous lectures to ensure you’ve mastered earlier topics. We expect lectures to be extremely linked to each other.
2. **Test your hypotheses** — validate your lecture content by raising and resolving issues, focusing primarily on your *main task issue* (see this example from [class 03](https://github.com/LearningToOptimize/LearningToControlClass/issues/18)).  

All interactions will happen **only through GitHub** — no in-person hints will be given. 

## Weekly Schedule (Fall 2025 – Fridays 2 p.m. ET)

#### In-person:

| #  | Date (MM/DD) | Format / Presenter | Topic & Learning Goals | Prep / Key Resources |
|----|--------------|--------------------|------------------------|----------------------|
| 1  | 08/22/2025   | Lecture — Andrew Rosemberg | Course map; why PDE-constrained **optimization**; tooling overview; stability & state-space dynamics; Lyapunov; discretization issues | [📚](https://learningtooptimize.github.io/LearningToControlClass/dev/class01/class01/) |
| 2  | 08/29/2025   | Lecture - Arnaud Deza | Numerical **optimization** for control (grad/SQP/QP); ALM vs. interior-point vs. penalty methods | [📚](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/overview/) |
| 3  | 09/05/2025   | Lecture - Zaowei Dai | Pontryagin’s Maximum Principle; shooting & multiple shooting; LQR, Riccati, QP viewpoint (finite / infinite horizon) | |
| 4  | 09/12/2025   | **External seminar 1** - Joaquim Dias Garcia| Dynamic Programming & Model-Predictive Control | |
| 5  | 09/19/2025   | Lecture - Guancheng "Ivan" Qiu | **Nonlinear** trajectory **optimization**; collocation; implicit integration | |
| 6  | 09/26/2025   | **External seminar 2** - Henrique Ferrolho | Trajectory **optimization** on robots in Julia Robotics | |
| 7  | 10/03/2025   | Lecture - Jouke van Westrenen | Stochastic optimal control, Linear Quadratic Gaussian (LQG), Kalman filtering, robust control under uncertainty, unscented optimal control; | |
| 8  | 10/10/2025   | **External seminar 3** TBD (speaker to be confirmed) | Topology **optimization** | |
| 9  | 10/17/2025   | **External seminar 4** — François Pacaud | GPU-accelerated optimal control | |
|10  | 10/24/2025   | Lecture - Michael Klamkin | Physics-Informed Neural Networks (PINNs): formulation & pitfalls | |
|11  | 10/31/2025   | **External seminar 5** - Chris Rackauckas | Neural Differential Equations: PINNs + classical solvers | |
|12  | 11/07/2025   | Lecture - Pedro Paulo | Neural operators (FNO, Galerkin Transformer); large-scale surrogates | |
|13  | 11/14/2025   | **External seminar 6** - Charlelie Laurent | Scalable PINNs / neural operators; CFD & weather applications | |
|14  | 11/21/2025   | Lecture - TBD | TBD from the pool | |

#### Pool of additional topics

If there are more students than slots, we will select from the following topics for recorded lectures. 
Students must provide materials equivalent to those used in an in-person session by `10/03/2025`.

| # | Format / Presenter | Topic & Learning Goals | Prep / Key Resources |
|---|--------------------|------------------------|----------------------|
| 15 | Lecture - TBD | Quaternions, Lie groups, and Lie algebras; attitude control; LQR with Attitude, Quadrotors; | |
| 16 | Lecture - TBD | Stochastic optimal control, Linear Quadratic Gaussian (LQG), Kalman filtering, robust control under uncertainty, unscented optimal control; | |
| 17 | Lecture - TBD | Trajectory Optimization with Obstacles; Convexification of Non-Convex Constraints; | |
| 18 | Lecture - Joe Ye | Robust control & min-max DDP (incl. PDE cases); chance constraints; Data-driven control & Model-Based RL-in-the-loop | |
| 19 | Lecture - TBD | Contact Explict and Contact Implicit; Trajectory Optimization for Hybrid and Composed Systems;  | |
| 20 | Lecture - TBD | Probabilistic Programming; Bayesian numerical methods; Variational Inference; probabilistic solvers for ODEs/PDEs; Bayesian optimization in control;  | |
| 21 | Lecture - TBD | Distributed optimal control & multi-agent coordination; Consensus, distributed MPC, and optimization over graphs (ADMM). | |
| 22 | Lecture - Shuaicheng (Allen) Tong | Dynamic Optimal Control of Power Systems; Generators swing equations, Transmission lines electromagnetic transients, dynamic load models, and inverters. | |

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
