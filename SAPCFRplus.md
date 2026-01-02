Technical Report: A Guide to Migrating from DCFR/DCFR+ to a SAPCFR+ Implementation

1.0 Introduction: The Evolution from Discounting to Robust Prediction

In the domain of solving large imperfect-information games, the continuous evolution of Counterfactual Regret Minimization (CFR) algorithms is of paramount strategic importance. Each new variant aims to achieve faster and more stable convergence towards a Nash Equilibrium. This document serves as a clear, mathematical, and implementation-focused guide for transitioning a codebase from a Discounted CFR (DCFR) or DCFR+ model to a Simple Asymmetric Predictive CFR+ (SAPCFR+) model. It is designed for developers and researchers who have an existing implementation of DCFR or its variants and are seeking to leverage more advanced techniques for enhanced performance. The migration path detailed herein involves replacing the foundational temporal discounting mechanism with an optimistic prediction model, which is further stabilized by a novel asymmetric step size to improve robustness against prediction errors.

The primary advantages of migrating from a discounting to a predictive framework, specifically to SAPCFR+, are substantial:

* From Discounting to Prediction: The transition represents a fundamental conceptual shift. Discounted models like DCFR operate by assigning less weight to regrets from earlier iterations, thereby mitigating the long-term impact of early, costly mistakes. In contrast, predictive models such as PCFR+ and its derivatives actively forecast the next iteration's regret, using this prediction to optimistically guide the strategy update.
* Addressing Prediction Inaccuracy: The primary weakness of standard predictive models like PCFR+ is their sensitivity to prediction accuracy. When the predicted regret (typically the previous iteration's regret) is a poor proxy for the actual regret, the algorithm's convergence can be significantly harmed, leading to performance instability.
* The SAPCFR+ Solution: SAPCFR+ emerges as a robust and elegant solution to this problem. It effectively mitigates the negative impact of prediction inaccuracy by introducing a simple, fixed asymmetry of step sizes. This modification dampens the influence of the prediction term in the update rule, thereby enhancing performance stability and robustness without sacrificing the core benefits of the predictive approach.

This report will first establish the baseline algorithms, DCFR and DCFR+, before deconstructing the predictive model they will be replaced with, and finally, outline the precise steps and code-level changes required for a successful migration.

2.0 Foundational Algorithms: A Review of DCFR and DCFR+

A clear understanding of the starting point—the architecture of DCFR and DCFR+—is essential before outlining the migration process. The core principle behind the DCFR family of algorithms is the discounting of cumulative regrets from earlier iterations. This mechanism is designed to address a key weakness in standard CFR, where high-regret actions taken in early, exploratory phases of the search can disproportionately influence the strategy for thousands of subsequent iterations. By systematically reducing the weight of past regrets, DCFR allows the algorithm to more rapidly adapt and recover from these initial costly mistakes.

Discounted CFR (DCFR)

The update mechanism of Discounted CFR (DCFR) applies a temporal discount factor to the cumulative regret before adding the instantaneous regret of the current iteration. It uses two distinct hyperparameters, α and β, to apply different discount rates to positive and non-positive cumulative regrets, respectively.

* Cumulative Regret Update Rule: R^t_I = R^{t-1}_I \odot d^{t-1}_I + r^t_I
* Discount Factor (d_t): d^t_I[a] = \begin{cases} \frac{t^\alpha}{t^\alpha+1} & \text{if } R^t_I[a] > 0 \\ \frac{t^\beta}{t^\beta+1} & \text{otherwise} \end{cases}

Here, t is the iteration number, r^t_I is the instantaneous regret vector at iteration t for infoset I, and ⊙ denotes element-wise multiplication.

DCFR+

The DCFR+ variant elegantly merges the discounting principle of DCFR with the "plus" mechanic from CFR+. After each update, any negative cumulative regrets are clipped to zero. This ensures that the algorithm does not accumulate negative regret, allowing actions that perform well to be revisited more promptly. This is achieved by applying the discount factor for positive regrets before the summation and then clipping the result to zero.

* Cumulative Regret Update Rule: R^t_I = [ R^{t-1}_I \cdot (\frac{(t-1)^\alpha}{(t-1)^\alpha+1}) + r^t_I ]_+

The [ ]+ notation signifies that each element of the resulting vector is set to max(0, value).

Pseudo Code: Core DCFR+ Regret Update

For each infoset I:

1. Calculate instantaneous regret: r_t

2. Calculate discount factor based on t and alpha

3. Update cumulative regret: R_t = R_t-1 * discount_factor + r_t

4. Clip negative regrets: R_t = max(0, R_t)

5. Calculate new strategy based on R_t

This discounting approach, while effective, is fundamentally reactive. The migration to SAPCFR+ involves moving from this reactive de-weighting of the past to a proactive, predictive approach.

3.0 The Predictive Leap: Deconstructing PCFR+ and its Limitations

The migration to SAPCFR+ begins with understanding its direct predecessor, Predictive CFR+ (PCFR+). This algorithm represents a paradigm shift from discounting past information to optimistically predicting future information. The core innovation of PCFR+ is the maintenance of two separate accumulated counterfactual regret vectors: an implicit vector based on observed regrets and an explicit vector based on predicted regrets. This dual-regret system allows the algorithm to use a forward-looking estimate to define its current strategy while accumulating actual, observed regret to ground its future predictions.

Core PCFR+ Mechanism

In PCFR+, the strategy for the current iteration t is derived from the explicit regret, R̂^t_I, which is calculated using a prediction of the current regret. The simplest and most common practice is to use the previous iteration's instantaneous regret, r^{t-1}_I, as this prediction. The implicit regret, R^{t+1}_I, is then updated using the actual observed regret, r^t_I, from playing the current strategy.

* Explicit (Predicted) Regret Update: \hat{R}^t_I = [R^t_I + r^{t-1}_I]_+
* Implicit (Observed) Regret Update: R^{t+1}_I = [R^t_I + r^t_I]_+
* Strategy Calculation: \sigma^t_i(I) = \frac{\hat{R}^t_I}{ ||\hat{R}^t_I||_1 }

The strategy for the current iteration, σ^t_i, is determined by normalizing the positive values of the explicit (predicted) regret vector.

The Vulnerability of Inaccurate Predictions

The primary vulnerability of the PCFR+ algorithm lies in its direct dependence on the accuracy of its prediction. The algorithm's performance is optimized when the prediction, r^{t-1}_I, is a close proxy for the actual observed regret, r^t_I. However, in many games, particularly those with volatile or non-stationary dynamics, this prediction can be highly inaccurate. When the prediction is poor, the optimistic step can guide the strategy in a suboptimal direction, harming the empirical convergence rate and making the algorithm's performance unstable across different game types.

This sensitivity to prediction error is precisely the problem that the asymmetric step size in SAPCFR+ is designed to solve.

4.0 The Migration Path: Modifying the Update Rule from DCFR+ to SAPCFR+

The migration from a DCFR+ architecture to SAPCFR+ is a fundamental replacement of the core regret update logic, not an incremental addition of features. The process involves completely removing the temporal discounting mechanism and implementing the two-regret system (implicit/explicit) characteristic of predictive models, but with a crucial modification to the explicit update rule that introduces a stabilizing asymmetric step size.

The conceptual changes can be broken down into three phases:

1. Phase 1: Deprecating the Discounting Mechanism. The first step is to remove the discounting logic from the existing DCFR/DCFR+ regret update loop. The term d^t_I or ((t-1)^α / ((t-1)^α+1)) is eliminated entirely. The core regret update will now be a simple summation, which forms the basis of the new implicit regret vector in SAPCFR+. The new rule for this vector is simply: R^{t+1}_I = [R^t_I + r^t_I]_+.
2. Phase 2: Introducing the Predictive (Explicit) Regret. Next, a second regret calculation must be introduced for the explicit regret, R̂^t_I. This vector will be used exclusively to determine the player's strategy for the current iteration. Implementing this requires storing the instantaneous regret from the previous iteration, r^{t-1}_I, so it can be used as the prediction in the current iteration t.
3. Phase 3: Implementing the Asymmetric Step Size. This is the key modification that defines SAPCFR+. Instead of the standard PCFR+ update (R̂^t_I = [R^t_I + r^{t-1}_I]_+), SAPCFR+ reduces the step size of the predictive term. This is based on the more general Asymmetric PCFR+ (APCFR+), which introduces an asymmetry parameter α. SAPCFR+ simplifies this by using a fixed value of α_tI = 2. This systematically dampens the influence of the (potentially inaccurate) prediction r^{t-1}_I. The update rule for the explicit regret thus becomes: \hat{R}^t_I = [R^t_I + \frac{1}{1 + \alpha_{tI}} r^{t-1}_I]_+ = [R^t_I + \frac{1}{3} r^{t-1}_I]_+

Final SAPCFR+ Update Rules

The complete set of update rules for the SAPCFR+ algorithm is as follows, replacing the single regret update of DCFR+.

* Explicit Regret (for strategy): \hat{R}^t_I = [R^t_I + \frac{1}{3} \cdot r^{t-1}_I]_+
* Implicit Regret (for next iteration): R^{t+1}_I = [R^t_I + r^t_I]_+
* Strategy Calculation: \sigma^t_i(I) = \frac{\hat{R}^t_I}{ ||\hat{R}^t_I||_1 }

The elegance of this migration lies in its simplicity. A profound improvement in robustness is achieved by replacing a complex discounting schedule with a simple, fixed coefficient in the predictive update.

5.0 Implementation Guide and Pseudo Code

This section provides a practical, side-by-side comparison to make the transition from a DCFR+ implementation to an SAPCFR+ one tangible for a developer. The core changes involve altering the data structures to support the two-regret system and modifying the main algorithm loop to perform the new update steps in the correct order.

Data Structure Comparison

Required Data Structures (DCFR+)	SAPCFR+ Data Structures
Cumulative_Regret[infoset]	Implicit_Regret[infoset]
Previous_Instantaneous_Regret[infoset]

In DCFR+, a single vector stores all historical regret information. In SAPCFR+, two vectors are needed: the Implicit_Regret is carried over between iterations, while Previous_Instantaneous_Regret stores r^{t-1} for the next iteration's prediction. The explicit regret (R̂^t_I) can be calculated on the fly at the beginning of each iteration and does not necessarily need to be stored as a persistent data structure between iterations.

Pseudo Code: Transitioning the Core Algorithm Loop

--- BEFORE: DCFR+ Update Logic ---

For each iteration t and infoset I:

1. traverse_tree(strategy_from(Cumulative_Regret)) -> returns instantaneous_regret_t
2. discount = ((t-1)^alpha) / ((t-1)^alpha + 1)
3. Cumulative_Regret = Cumulative_Regret * discount + instantaneous_regret_t
4. Cumulative_Regret = max(0, Cumulative_Regret) # The '+' part

--- AFTER: SAPCFR+ Update Logic ---

For each iteration t and infoset I:

1. Explicit_Regret = Implicit_Regret + (1/3) * Previous_Instantaneous_Regret
2. Explicit_Regret = max(0, Explicit_Regret)
3. traverse_tree(strategy_from(Explicit_Regret)) -> returns current_instantaneous_regret_t
4. Implicit_Regret = Implicit_Regret + current_instantaneous_regret_t
5. Implicit_Regret = max(0, Implicit_Regret)
6. Previous_Instantaneous_Regret = current_instantaneous_regret_t

The core implementation change involves restructuring the loop to first calculate the strategy based on a prediction, then execute the tree traversal to get the actual regret, and finally update the implicit regret and store the new instantaneous regret for the next cycle.

6.0 Conclusion and Key Takeaways

This report has detailed the complete migration path from a time-discounting regret minimization algorithm, DCFR or DCFR+, to a robust, predictive algorithm, SAPCFR+. The transition moves away from a reactive model that de-weights past errors towards a proactive one that optimistically predicts the future, while crucially including a mechanism to protect against the consequences of inaccurate predictions. This evolution offers a significant step forward in achieving faster and more stable convergence when solving complex imperfect-information games.

The most critical takeaways from this migration guide are:

1. Superior Robustness: SAPCFR+ directly addresses the primary vulnerability of predictive CFR methods. By systematically dampening the effect of inaccurate predictions with a fixed asymmetric step size, it enhances the stability and reliability of the convergence process, making it more effective across a wider range of games.
2. Implementation Simplicity: A significant boost in algorithmic robustness is achieved through a minimal and intuitive change. The migration replaces the entire discounting apparatus with a single-line modification to the explicit regret update rule: \hat{R}^t_I = [R^t_I + \frac{1}{3} \cdot r^{t-1}_I]_+. This simplicity lowers the barrier to adoption for researchers and developers.
3. Conceptual Shift: The migration embodies a powerful conceptual evolution in algorithm design for game solving. It marks a move beyond simply de-weighting the past to intelligently and, most importantly, cautiously predicting the future. This refined approach leads to more stable convergence, representing a state-of-the-art technique in the field.
