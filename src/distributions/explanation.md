# Understanding Bayesian Shrinkage Priors

In Bayesian shrinkage models, the shrinkage factor $\kappa_i$ defines how much a coefficient is pulled toward zero. It is defined as:

$$\kappa_i = \frac{1}{1 + \lambda_i^2}$$

* **$\kappa \approx 1$:** The model preserves the signal (little to no shrinkage).
* **$\kappa \approx 0$:** The model shrinks the coefficient toward zero (filtering out noise).

---

### Comparison of Prior Distributions



| Prior | Distribution of $\lambda_i$ | Shrinkage Behavior |
| :--- | :--- | :--- |
| **Gaussian** | Constant variance | **Global:** Applies uniform shrinkage; no feature selection. |
| **Laplace** | Exponential | **Fixed:** Good at shrinking small noise, but biases large signals. |
| **Student-t** | Inverse-Gamma | **Adaptive:** More robust than Gaussian, but less sparse than Horseshoe. |
| **Horseshoe** | Half-Cauchy | **Optimal:** Highly flexible; removes noise while preserving signal. |

---

### Key Mechanisms
* **Gaussian:** The "knob" is glued in place. It performs proportional, non-sparse shrinkage.
* **Laplace:** The "knob" is set to a fixed position. It is excellent at pushing small coefficients to zero but lacks the flexibility to avoid shrinking large signals.
* **Student-t:** A compromise. More robust to outliers than Gaussian, but lacks the extreme sparsity required for high-dimensional feature selection.
* **Horseshoe:** The "knob" is hyper-flexible. Due to the heavy tails of the Half-Cauchy distribution, it acts as an "all-or-nothing" operator, perfectly balancing noise removal and signal preservation.