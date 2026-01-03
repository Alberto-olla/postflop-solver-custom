# Technical Report: DCFR+ Algorithm Implementation & Optimization

## Abstract
This report describes the implementation of the Discounted Counterfactual Regret Minimization Plus (DCFR+) algorithm within the current solver framework. It focuses on the mathematical foundations, discounting mechanics, and the high-performance mixed-precision quantization scheme (16-bit strategy, 8-bit regrets, 4-bit compression for auxiliary values).

## 1. DCFR+ Mathematical Formulation

DCFR+ is an evolution of the Discounted CFR (DCFR) algorithm that integrates the advancements of CFR+ (regret clipping) with dynamic discounting of historical data to accelerate convergence.

### 1.1 Regret Discounting ($\alpha_t$)
Unlike standard CFR which treats all iterations equally, DCFR+ applies a uniform discount factor to accumulated regrets. The discount factor $\alpha_t$ increases over time, giving more weight to recent iterations as the strategy stabilizes.

$$R_i^T = (R_i^{T-1} \cdot \alpha_T + r_i^T)^+$$

Where:
- $R_i^T$ is the accumulated regret for action $i$ at iteration $T$.
- $r_i^T$ is the instantaneous regret.
- $\alpha_T = \frac{T^{1.5}}{T^{1.5} + 1}$ (implemented as $(T-1)$ based alignment with DCFR).

### 1.2 Strategy Averaging & Discounting ($\gamma_t$)
To prevent early stale strategy from polluting the final average strategy, DCFR+ employs a cubic discounting factor $\gamma_t$ and a periodic reset heuristic.

$$\bar{\sigma}_i^T = \bar{\sigma}_i^{T-1} \cdot \gamma_T + \sigma_i^T \cdot w_T$$

In our implementation, $w_T = \gamma_T$ is implicitly handled through frequency weighting, and $\gamma_T$ is defined as:

$$\gamma_T = \left( \frac{T_{reset}}{T_{reset} + 1} \right)^3$$

Where $T_{reset}$ is the number of iterations since the last **Strategy Reset**.

#### Strategy Reset Heuristic
The solver resets the cumulative strategy weights at specific intervals to "forget" the initial noisy results. Resets occur at iterations $t \in \{1, 4, 16, 64, 256, \dots\}$, following a power-of-4 sequence.

## 2. Mixed Precision Quantization (16/8/4/4)

To minimize memory footprint while maintaining numerical stability, the solver employs a heterogeneous quantization architecture.

### 2.1 Regret Storage (8-bit Unsigned)
Accumulated regrets in DCFR+ are always non-negative due to clipping $( \cdot )^+$. We exploit this by using **8-bit Unsigned Integers (`u8`)** instead of signed integers or floats.

- **Encoding**: Each regret $R$ is scaled to a `u8` range $[0, 255]$.
- **Stochastic Rounding**: To prevent quantization bias from accumulating, we use **Stochastic Rounding** instead of deterministic rounding.
  - $round_{stoch}(x) = \lfloor x \rfloor + 1$ with probability $x - \lfloor x \rfloor$.
  - This preserves the expected value of the regrets over many iterations.

### 2.2 Strategy Storage (16-bit Unsigned)
The cumulative strategy requires higher precision than regrets because fine-grained probabilities are critical for the final exploitability convergence. We use **16-bit Unsigned Integers (`u16`)** for strategy weights.

### 2.3 Auxiliary Values (4-bit Packed)
IP Counterfactual Values (CFVs) and Chance CFVs are compressed using **4-bit Signed Quantization**.
- Two values are packed into a single byte (`u8`).
- Uses a per-node scaling factor.
- Signed range: $[-7, 7]$.
- Decoded using: $V = \text{nibble} \times \frac{Scale}{7}$.

## 3. Implementation Details

### 3.1 Algorithm Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| $\alpha_{pow}$ | 1.5 | Exponent for regret discounting. |
| $\gamma_{pow}$ | 3.0 | Exponent for strategy averaging. |
| $\beta$ | 0.5 | (Not used in DCFR+, kept at 0.5 for DCFR parity). |
| Reset Base | 4 | Base for periodic strategy resets. |

### 3.2 Pruning (Branch Skipping)
The solver implements **Regret Pruning**. If an action's average regret falls below a dynamic threshold, the entire sub-tree is skipped for the current iteration.

$$Threshold = - (EffectiveStack \cdot \sqrt{T} \cdot K)$$
Where $K=10$ is a safety constant to prevent premature pruning.

## 4. Conclusion
The DCFR+ implementation combines the theoretical benefits of aggressive discounting with state-of-the-art memory optimization. By utilizing 8-bit stochastically rounded regrets and 4-bit auxiliary values, the solver can handle significantly larger game trees within the same RAM constraints as traditional solvers, without sacrificing convergence speed.
