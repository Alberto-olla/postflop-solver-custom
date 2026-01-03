A Technical Report on the Predictive Discounted Counterfactual Regret Minimization (PDCFR+) Algorithm

1.0 Introduction to Equilibrium-Finding in Imperfect-Information Games

In the study of strategic interactions, Imperfect-Information Games (IIGs) present a formidable challenge, modeling scenarios where players must make decisions without complete knowledge of the game state. The primary objective in two-player zero-sum (2p0s) IIGs is to compute an approximate Nash Equilibrium (NE), a strategy profile where neither player can improve their outcome by unilaterally changing their strategy. The pursuit of such equilibria is not merely an academic exercise; it holds significant strategic importance in a variety of real-world domains, including automated strategic negotiations, cybersecurity protocols, and high-stakes games like poker.

For large-scale IIGs, where the state space is too vast for exact methods, iterative algorithms are the primary tool for approximating an NE. Among these, the Counterfactual Regret Minimization (CFR) family of algorithms has emerged as the most popular and empirically effective approach. The original "vanilla" CFR algorithm laid the foundational principles, but its convergence rate left room for significant improvement.

The development of CFR+ marked a key breakthrough, demonstrating an order-of-magnitude increase in convergence speed over its predecessor. From this new benchmark, two distinct pathways for improvement emerged. The first, Discounted CFR (DCFR), was designed to address a critical weakness in standard CFR: the tendency for high-regret mistakes made in early iterations to persistently skew strategy calculations for thousands of subsequent iterations. DCFR mitigates this by discounting the influence of past regrets. The second pathway led to Predictive CFR+ (PCFR+), an algorithm that leverages an "optimistic" or predictive mechanism, based on a theoretical connection to Online Mirror Descent, to anticipate future regrets and accelerate convergence.

This report addresses the central problem arising from the limitations of these two advanced variants. PCFR+'s performance degrades when confronted with dominated actions that generate high initial regret, as it weighs all iterations uniformly. Conversely, while DCFR effectively mitigates these early mistakes, it lacks a predictive mechanism to capitalize on the slowly-changing nature of game losses. This creates a clear need for an algorithm that synergizes the strengths of both approaches. This document provides a detailed technical and mathematical exposition of the Predictive Discounted Counterfactual Regret Minimization (PDCFR+) algorithm. PDCFR+ integrates the regret discounting of DCFR with the predictive updates of PCFR+ to achieve faster, more robust convergence in a wide range of game settings.

2.0 Mathematical Preliminaries and Notation

To rigorously analyze the mechanics of CFR algorithms, it is essential to first establish a clear and consistent mathematical framework. This section defines the core concepts of extensive-form games and regret minimization, which form the building blocks for the algorithms discussed in this report.

A two-player zero-sum imperfect-information extensive-form game can be described by the following components:

Term	Description
Information Set (Infoset), I	A set of game states that are indistinguishable to a player. For any two states h, h' in I, the acting player and available actions are identical.
Strategy, σi(I, a)	The probability that player i chooses action a in infoset I. σi(I) is the probability vector over all actions in I.
Strategy Profile, σ	A tuple of strategies, one for each player, defining the complete behavior for all players in the game.
Payoff Function, ui(σ)	The expected payoff for player i when all players follow the strategy profile σ.
Nash Equilibrium (NE)	A strategy profile σ* where no player has an incentive to unilaterally deviate: ui(σ*i, σ*−i) ≥ ui(σ'i, σ*−i) for any alternative strategy σ'i.
Exploitability, e(σ)	A measure of how far a strategy profile σ is from an NE. For a 2p0s game with sequence-form strategies (ẋ, ẏ), it is defined as e(ẋ, ẏ) = ∑ i∈{1,2} δi(ẋ, ẏ)/2, where δi is the incentive for player i to unilaterally deviate. An ε-NE is a profile with exploitability no greater than ε.

The CFR framework operates by minimizing a quantity known as counterfactual regret. This requires defining counterfactual values and regrets formally.

* Counterfactual Value (vσ(I, a)): This is the expected value for a player if they reach infoset I and choose action a, assuming all players follow strategy profile σ thereafter. Formally, it is the weighted average of the value of each state in the infoset after taking action a: vσ(I, a) = ∑h∈I (πσ−i(h|I)vσi (h · a)) where πσ−i(h|I) is the probability of being in state h given infoset I according to the opponent's strategy.
* Instantaneous Counterfactual Regret (rt(I, a)): This measures the immediate advantage of having chosen action a instead of following the current strategy σt at infoset I during iteration t. It is calculated as: rt(I, a) = vσt(I, a) − vσt(I) where vσt(I) is the expected value of following strategy σt(I).
* Cumulative Counterfactual Regret (RT(I, a)): This is the sum of instantaneous regrets for a given action over all iterations up to T: RT(I, a) = ∑t=1^T rt(I, a)

The local decision-making algorithm within CFR is Regret Matching (RM). It updates a player's strategy at each infoset based on the accumulated positive regrets. The strategy for the next iteration, T+1, is proportional to the positive cumulative regrets:

σT+1(I, a) = (RT+(I, a)) / (∑a'∈A(I) RT+(I, a'))

where RT+(I, a) = max{0, RT(I, a)}. If the sum of positive regrets is zero, a uniform random strategy is used. By minimizing these local counterfactual regrets at every infoset, the CFR algorithm minimizes the overall game regret, causing the average strategy to converge to a Nash Equilibrium.

3.0 The Evolution of Counterfactual Regret Minimization

PDCFR+ was not developed in a vacuum; it stands on the shoulders of a series of influential algorithms that progressively enhanced the performance of the original CFR framework. To understand the design principles of PDCFR+, it is crucial to analyze the key contributions and inherent limitations of its direct predecessors: CFR+, DCFR, and PCFR+. This section provides this context, building a clear rationale for the synthesis that PDCFR+ represents.

3.1 CFR+: The High-Performance Benchmark

CFR+ was a formidable benchmark that delivered convergence speeds often an order of magnitude faster than vanilla CFR. Its success is attributed to two simple yet powerful modifications:

1. Regret Matching+ (RM+): In contrast to standard RM, which allows cumulative regrets to become negative, RM+ resets any action's cumulative regret to zero if it becomes negative. This prevents actions that have performed poorly in the past from being suppressed for an unnecessarily long time. The update rule for the regret-like value Q at iteration T is: QT(I, a) = max{0, QT−1(I, a) + rt(I, a)}
2. Weighted Averaging: When computing the average strategy profile (which converges to the NE), CFR+ gives more weight to strategies from later iterations. As noted by Brown and Sandholm (2019a), "Empirically we observed that CFR+ converges faster when assigning iteration t a weight of t^2... We therefore use this weight for CFR+ and its variants throughout this paper". This reflects the assumption that later strategies are of higher quality.

3.2 Discounted CFR (DCFR): Mitigating High Initial Regrets

The primary motivation for DCFR was the observation that standard CFR variants perform poorly in games with very costly mistakes. A single catastrophic action in an early iteration can generate such high negative regret for that action (and high positive regret for others) that it dominates the cumulative regrets for thousands of subsequent iterations, effectively "locking in" a suboptimal strategy while the algorithm slowly corrects.

DCFR's core innovation is the discounting of regrets from earlier iterations. This ensures that the influence of old, potentially noisy regret information decays over time, allowing the algorithm to adapt more quickly. A key feature is the use of different decay rates for positive and negative regrets. The cumulative regret update rule for a decision node j is:

Rt j = Rt−1 j ⊙ dt−1 j + rtj

where ⊙ is element-wise multiplication and the discount factor dtj[a] is t^α / (t^α+1) if Rt j[a] > 0 and t^β / (t^β+1) otherwise. The parameters α and β control the decay rates. The average strategy is also updated using a discounted weighting scheme:

Xt = Xt−1((t−1)/t)^γ + ẋt

3.3 Predictive CFR+ (PCFR+): Leveraging Optimism

PCFR+ was developed by establishing a theoretical link between RM+ and Online Mirror Descent (OMD), a general framework for regret minimization. This connection enabled the creation of an "optimistic" or "predictive" variant that attempts to anticipate the next iteration's regret.

PCFR+ achieves this by maintaining two regret vectors at each decision node j for each iteration t:

1. Cumulative Regret (Rt j): This is updated using the standard RM+ rule: Rt j = [Rt−1 j + rtj]+ (where [·]+ denotes max(0, ·))
2. Predicted Regret (R̃t+1 j): This is an optimistic estimate of what the regret will be in the next iteration. It is calculated by adding a prediction of the next instantaneous regret, vt+1 j, to the current cumulative regret: R̃t+1 j = [Rt j + vt+1 j]+ A common practice is to use the most recent instantaneous regret as the prediction, i.e., vt+1 j ≈ rtj.
3. Strategy Update: Crucially, the strategy for the next iteration, xt+1 j, is derived from the predicted regret R̃t+1 j, not the cumulative regret.

Despite its rapid convergence in many games, PCFR+ has a critical limitation: its uniform weighting of regrets makes it highly susceptible to being derailed by high-regret dominated actions encountered in early iterations. This is the exact problem that DCFR was designed to solve, yet DCFR lacks the predictive mechanism of PCFR+. This sets the stage for a new algorithm that combines the benefits of both.

4.0 The PDCFR+ Algorithm: A Detailed Exposition

Predictive Discounted Counterfactual Regret Minimization (PDCFR+) is a principled synthesis of its predecessors, designed to overcome their individual limitations. Its core objective is to swiftly mitigate the negative effects of dominated actions and high initial regrets via discounting, while consistently leveraging optimistic predictions to accelerate convergence toward an equilibrium.

4.1 Mathematical Formulation

PDCFR+ achieves this synthesis by integrating the discounting factors of DCFR directly into the predictive update framework of PCFR+. At each decision node j on iteration t, the algorithm applies the following formal update rules:

1. Cumulative Regret Update (Rt j) This step updates the primary cumulative regret vector. It is identical to the DCFR+ update, applying a discount factor to the previous cumulative regret before adding the new instantaneous regret and clipping any negative values at zero. Rt j = [Rt−1 j * ((t−1)^α / ((t−1)^α+1)) + rtj ]+
2. Predictive Regret Update (R̃t+1 j) This is the core predictive step and the central innovation of PDCFR+. The newly updated cumulative regret Rt j is itself discounted before the prediction for the next iteration's regret, vt+1 j, is added. This prevents the optimistic prediction from being anchored to an undiscounted, potentially noisy history, thereby ensuring the "lookahead" is based on a more relevant, time-weighted foundation. R̃t+1 j = [Rt j * (t^α / (t^α+1)) + vt+1 j ]+
3. Next-Iteration Strategy (xt+1 j) Following the logic of PCFR+, the strategy for the next iteration is derived directly from the optimistic, predicted regret vector. xt+1 j = R̃t+1 j / ||R̃t+1 j||1
4. Average Strategy (Xt) The final average strategy, which converges to the NE, is computed using the same discounted weighting scheme as DCFR, giving greater importance to more recent iterations. Xt = Xt−1((t−1)/t)^γ + ẋt

4.2 Pseudocode Implementation

The following pseudocode outlines the core logic of the PDCFR+ algorithm from the perspective of a single player, emphasizing the correct handling of sequence-form strategies.

Algorithm PDCFR+ (Single Player)
Input: Game G, Total Iterations T, hyperparameters α, γ

// Initialize regrets for each local decision node j
for each decision node j of the current player do:
R_0[j] <- vector of zeros     // Cumulative regret
R_tilde_1[j] <- vector of zeros // Predicted regret
end for

// Initialize cumulative average strategy sum (sequence-form)
X_0 <- vector of zeros

for t = 1 to T do:
// 1. Construct current local strategies from previous predicted regrets
for each decision node j of the current player do:
if sum(R_tilde_t[j]) > 0 then
sigma_t[j] <- R_tilde_t[j] / sum(R_tilde_t[j])
else
sigma_t[j] <- uniform distribution over actions A(j)
end if
end for

// Convert local strategies to a full sequence-form strategy
x_t <- ConstructSequenceFormStrategy(sigma_t)

// 2. Compute instantaneous regrets for this iteration
r_t <- ComputeCounterfactualValues(G, sigma_t) // Returns local regrets
v_{t+1} <- r_t  // Use current regret as prediction for next

// 3. Update local regrets for each decision node
for each decision node j of the current player do:
// Cumulative Regret Update (with discounting)
discount_R <- ((t-1)^α) / ((t-1)^α + 1)
R_t[j] <- max(0, R_{t-1}[j] * discount_R + r_t[j])

    // Predictive Regret Update (with discounting)
    discount_R_tilde <- (t^α) / (t^α + 1)
    R_tilde_{t+1}[j] <- max(0, R_t[j] * discount_R_tilde + v_{t+1}[j])
end for

// 4. Update the cumulative average strategy sum (sequence-form)
avg_strategy_weight <- ((t-1)/t)^γ
X_t <- X_{t-1} * avg_strategy_weight + x_t
end for

// Return the final normalized average strategy
Return X_T / ||X_T||_1


5.0 Comparative Analysis and Performance

The theoretical design of an algorithm must ultimately be validated by its empirical performance. This section provides a comparative analysis of PDCFR+ against its key predecessors, CFR+ and DCFR+, examining both their core mechanisms and their observed convergence behavior in benchmark games. The results demonstrate that PDCFR+ successfully combines the strengths of prior methods, though its effectiveness varies with game structure.

The fundamental differences in the algorithms' update rules are summarized below. This comparison highlights how PDCFR+ uniquely integrates discounting into both the cumulative and predictive regret calculations.

Comparison of Core Update Mechanisms

Algorithm	Cumulative Regret Update Rule	Next-Iteration Strategy Source
CFR+	Rt j = [Rt−1 j + rtj]+	Current Cumulative Regret (Rt j)
DCFR+	Rt j = [Rt−1 j * ((t−1)^α/((t−1)^α+1)) + rtj]+	Current Cumulative Regret (Rt j)
PDCFR+	Rt j = [Rt−1 j * ((t−1)^α/((t−1)^α+1)) + rtj]+	Predicted Regret (R̃t+1 j)

A synthesis of experimental results reported in the literature reveals a clear performance pattern:

* Performance in Non-Poker and Small Poker Games: In non-poker games and small poker games such as Goofspiel, Battleship, and Kuhn Poker, PDCFR+ demonstrates exceptional performance. It has been shown to outperform other CFR variants by "4-8 orders of magnitude" in convergence speed (Xu et al., 2024b). This suggests that in games where strategies change more slowly and predictions are more reliable, the optimistic updates of PDCFR+ provide a substantial advantage.
* Performance in Large-Scale Poker Games: In contrast, for large-scale, complex environments like the Heads-Up No-Limit (HUNL) Texas Hold'em subgames, the simpler DCFR+ emerges as the fastest algorithm. The highly stochastic nature and vast state space of these games may reduce the accuracy of the simple predictive model (vt+1 j ≈ rtj), making the robust, non-predictive discounting of DCFR+ more effective.
* Performance in Games with Dominated Actions: A key validation of PDCFR+'s design comes from its performance in games with highly suboptimal, or "dominated," actions. In a test game like NFG (3), an early mistake can generate massive regret that hinders PCFR+ for thousands of iterations. Because PDCFR+ discounts past regrets, it is able to much more quickly learn stable cumulative regrets for the non-dominated actions, effectively ignoring the noise from the high initial mistake that continues to plague PCFR+.

In conclusion, the performance analysis shows that PDCFR+ is a powerful, general-purpose improvement over PCFR+, particularly for games where its predictive mechanism can be fully leveraged without being derailed by noisy initial iterations. At the same time, the simpler, non-predictive discounting of DCFR+ remains a more robust choice for the largest and most complex imperfect-information games tested to date.

6.0 Conclusion and Future Work

This report has provided a technical exposition of the Predictive Discounted Counterfactual Regret Minimization (PDCFR+) algorithm. The analysis demonstrates that PDCFR+ successfully integrates the regret discounting from DCFR and the optimistic predictive updates from PCFR+ into a single, principled CFR variant.

The primary contribution of PDCFR+ is its ability to mitigate the negative impact of dominated actions and high initial regrets, a known weakness of PCFR+, while still capitalizing on the accelerated convergence offered by optimistic predictions. It achieves this by applying discounting factors to both its cumulative and predictive regret update steps, ensuring that predictions are based on a forward-looking view that is not overly biased by early, potentially noisy, iterations. Empirical results confirm its state-of-the-art performance in many non-poker and small poker games, while also highlighting the continued robustness of simpler discounting methods in the largest game environments.

Several promising avenues for future research remain. One direction is to combine PDCFR+ with function approximation techniques, such as deep neural networks, to scale the algorithm to solve even larger games where explicitly storing regrets for every infoset is infeasible. Another promising area is the integration of PDCFR+ with dynamic discounting frameworks, which could allow the algorithm to adapt its discounting schedule automatically based on the game's dynamics, potentially leading to a more universally robust and high-performance equilibrium-finding algorithm.
