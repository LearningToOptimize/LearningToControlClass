# Class 2 — 08/29/2025

**Presenter:** Arnaud Deza

**Topic:** Numerical optimization for control (gradient/SQP/QP); ALM vs. interior-point vs. penalty methods

---

## Overview

This class covers the fundamental numerical optimization techniques essential for optimal control problems. We explore gradient-based methods, Sequential Quadratic Programming (SQP), and various approaches to handling constraints including Augmented Lagrangian Methods (ALM), interior-point methods, and penalty methods.

## Learning Objectives

By the end of this class, students will be able to:

- Understand the mathematical foundations of gradient-based optimization
- Implement Newton's method for unconstrained minimization
- Apply root-finding techniques for implicit integration schemes
- Solve equality-constrained optimization problems using Lagrange multipliers
- Compare and contrast different constraint handling methods (ALM, interior-point, penalty)
- Implement Sequential Quadratic Programming (SQP) for nonlinear optimization

## Prerequisites

- Solid understanding of linear algebra and calculus
- Familiarity with Julia programming
- Basic knowledge of differential equations
- Understanding of optimization concepts from Class 1

## Materials

### Interactive Notebooks

The class is structured around four interactive Jupyter notebooks that build upon each other:


1. **[Part 1a: Root Finding & Backward Euler](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part1_root_finding.html)**
   - Root-finding algorithms for implicit integration
   - Fixed-point iteration vs. Newton's method
   - Backward Euler implementation for ODEs
   - Convergence analysis and comparison
   - Application to pendulum dynamics

2. **[Part 1b: Minimization via Newton's Method](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part1_minimization.html)**
   - Unconstrained optimization fundamentals
   - Newton's method for minimization
   - Hessian matrix and positive definiteness
   - Regularization and line search techniques
   - Practical implementation with Julia

3. **[Part 2: Equality Constraints](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part2_eq_constraints.html)**
   - Lagrange multiplier theory
   - KKT conditions for equality constraints
   - Quadratic programming with equality constraints
   - Visualization of constrained optimization landscapes
   - Practical implementation examples

4. **[Part 3: Interior-Point Methods](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part3_ipm.html)**
   - Inequality constraint handling
   - Barrier methods and log-barrier functions
   - Interior-point algorithm implementation
   - Comparison with penalty methods
   - Convergence properties and practical considerations

### Additional Resources

- **[Lecture Slides (PDF)](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/ISYE_8803___Lecture_2___Slides.pdf)** - Complete slide deck from the presentation
- **[LaTeX Source Files](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/main.tex)** - Source code for the lecture slides
- **[Demo Script](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/penalty_barrier_demo.py)** - Python demonstration of penalty vs. barrier methods

## Key Concepts Covered

### Mathematical Foundations
- **Gradient and Hessian**: Understanding first and second derivatives in optimization
- **Newton's Method**: Quadratic convergence and implementation details
- **KKT Conditions**: Necessary and sufficient conditions for optimality
- **Duality Theory**: Lagrange multipliers and dual problems

### Numerical Methods
- **Root Finding**: Fixed-point iteration, Newton-Raphson method
- **Implicit Integration**: Backward Euler for stiff ODEs
- **Sequential Quadratic Programming**: Local quadratic approximations
- **Interior-Point Methods**: Barrier functions and path-following

### Constraint Handling
- **Equality Constraints**: Lagrange multipliers and null-space methods
- **Inequality Constraints**: Active set methods and interior-point approaches
- **Penalty Methods**: Quadratic and exact penalty functions
- **Augmented Lagrangian**: Combining penalty and multiplier methods

## Practical Applications

The methods covered in this class are fundamental to:
- **Optimal Control**: Trajectory optimization and feedback control design
- **Model Predictive Control**: Real-time optimization with constraints
- **Robotics**: Motion planning and control with obstacle avoidance
- **Engineering Design**: Constrained optimization in mechanical systems

## Further Reading

## Next Steps

This class provides the foundation for the advanced topics covered in subsequent classes, including:
- Pontryagin's Maximum Principle (Class 3)
- Nonlinear trajectory optimization (Class 5)
- Stochastic optimal control (Class 7)
- Physics-Informed Neural Networks (Class 10)

---

*For questions or clarifications, please reach out to Arnaud Deza at adeza3@gatech.edu*
