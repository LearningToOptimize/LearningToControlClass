# Class 2 — 08/29/2025

**Presenter:** Arnaud Deza

**Topic:** Numerical optimization for control (gradient/SQP/QP); ALM vs. interior-point vs. penalty methods

---

## Overview

This class covers the fundamental numerical optimization techniques essential for optimal control problems. We explore gradient-based methods, Sequential Quadratic Programming (SQP), and various approaches to handling constraints including Augmented Lagrangian Methods (ALM), interior-point methods, and penalty methods.

## Interactive Materials

The class is structured around 1 slide deck and four interactive Jupyter notebooks:

1. **[Part 1a: Root Finding & Backward Euler](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part1_root_finding.html)**
   - Root-finding algorithms for implicit integration
   - Fixed-point iteration vs. Newton's method
   - Application to pendulum dynamics


2. **[Part 1b: Minimization via Newton's Method](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part1_minimization.html)**
   - Unconstrained optimization fundamentals
   - Newton's method implementation
   - Globalization strategies: Hessian matrix and regularization

3. **[Part 2: Equality Constraints](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part2_eq_constraints.html)**
   - Lagrange multiplier theory
   - KKT conditions for equality constraints
   - Quadratic programming implementation

4. **[Part 3: Interior-Point Methods](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part3_ipm.html)**
   - Inequality constraint handling
   - Barrier methods and log-barrier functions
   - Comparison with penalty methods

## Additional Resources

- **[Lecture Slides (PDF)](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/ISYE_8803___Lecture_2___Slides.pdf)** - Complete slide deck
- **[LaTeX Source](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/main.tex)** - Source code for lecture slides

## Key Learning Outcomes

- Understand gradient-based optimization methods
- Implement Newton's method for minimization
- Apply root-finding techniques for implicit integration
- Solve equality-constrained optimization problems
- Compare different constraint handling methods
- Implement Sequential Quadratic Programming (SQP)

## Next Steps

This class provides the foundation for advanced topics in subsequent classes, including Pontryagin's Maximum Principle, nonlinear trajectory optimization, and stochastic optimal control.

