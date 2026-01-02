From Discounted CFR to DCFR+: A Mathematical and Algorithmic Evolution

The Counterfactual Regret Minimization (CFR) family of algorithms represents the state of the art for finding approximate Nash equilibria in large-scale imperfect-information games. Since its introduction, this iterative approach has spawned numerous variants, each building upon the last to achieve faster and more stable convergence. This report provides a detailed mathematical and algorithmic exposition of the evolution from Discounted CFR (DCFR) to its successor, DCFR+. We will explore how DCFR+ achieves its superior performance through the principled integration of core concepts from previous variants, namely the regret discounting of DCFR and the negative regret clipping of CFR+. This analysis begins with a foundational review of the algorithms that set the stage for these advancements: vanilla CFR and its key breakthrough, CFR+.

1. Foundational Algorithms: A Recap of CFR and CFR+

To fully appreciate the innovations of regret discounting, it is essential to first understand the foundational algorithms upon which they are built. Both vanilla CFR and its highly successful successor, CFR+, establish the core mechanics of regret-based strategy updates that are refined in later variants. A clear grasp of CFR's fundamental regret matching process and the powerful enhancements introduced by CFR+ provides the necessary baseline for comprehending the subsequent evolution toward DCFR and DCFR+.

1.1. Vanilla Counterfactual Regret Minimization (CFR)

The original Counterfactual Regret Minimization (CFR) algorithm, introduced by Zinkevich et al. (2007), operates by decomposing the total game regret into a series of local regrets at each decision point, or infoset. It iteratively refines a player's strategy by minimizing these local regrets over time.

The core of the algorithm relies on tracking the cumulative regret for not having chosen a specific action. The instantaneous regret for taking action a in infoset I during iteration t is defined as the difference between the value of that action and the overall value of the infoset under the current strategy profile σt:

rt(I, a) = vσt(I, a) − vσt(I)

This value is accumulated over iterations to calculate the cumulative regret for action a at the end of iteration T:

RT(I, a) = ∑t=1T rt(I, a)

CFR uses a simple yet effective regret-minimization algorithm known as Regret Matching (RM) to determine the strategy for the next iteration. In RM, the probability of selecting an action is proportional to its positive cumulative regret. The strategy σT+1 is defined as:

σT+1(I, a) = { (RT+(I,a)) / (∑a′∈A(I) RT+(I, a′)) if ∑a′ RT+(I, a′) > 0; 1 / |A(I)| otherwise }

where RT+(I, a) = max{0, RT(I, a)}.

Finally, the algorithm's output is not the strategy from the final iteration but an average strategy computed over all iterations. In vanilla CFR, this is a uniformly-weighted average of the strategies from each iteration t, weighted by the player i's reach probability to the infoset:

σ̄Ti(I) = (∑t=1T πσti(I)σti(I)) / (∑t=1T πσti(I))

This process guarantees that, as the number of iterations approaches infinity, the average strategy profile converges to a Nash equilibrium in two-player, zero-sum games.

1.2. The CFR+ Breakthrough

While theoretically sound, vanilla CFR's convergence could be slow in practice. The development of CFR+ introduced two concise yet impactful modifications that often led to an order of magnitude faster convergence.

1. Regret Clipping with Regret Matching+ (RM+): The first change addresses how regrets are accumulated. Instead of allowing cumulative regrets to become arbitrarily negative, CFR+ ensures that cumulative regrets are never negative. After each iteration, any action with a negative cumulative regret is "clipped" or reset to zero. This mechanism, known as Regret Matching+ (RM+), uses the following update rule:
2. QT(I, a) = max{0, QT−1(I, a) + rt(I, a)}
3. To maintain consistency with the source literature, Q is used here to denote the regret-like value in RM+; it is conceptually equivalent to the cumulative regret R used elsewhere, with the key difference being the clipping operation. This prevents the algorithm from having to overcome large negative regrets from past mistakes before it can explore a promising action again.
4. Weighted Strategy Averaging: The second change modifies how the average strategy is computed. Instead of weighting each iteration uniformly, "CFR+ uses a weighted average strategy where iteration T is weighted by T". Empirical results have shown that weighting by t² can lead to even faster convergence. This contrasts sharply with CFR's uniform weighting and allows the average strategy to more quickly reflect the improved strategies of later iterations.

CFR+ was a key breakthrough in solving large imperfect-information games. However, its approach of giving equal weight to each iteration's regret contribution revealed a weakness in certain game types, creating an opportunity for a new approach based on discounting.

2. The Motivation for Discounting: Introducing DCFR

A significant limitation of algorithms like CFR and CFR+ is that the regret from every iteration contributes equally to the cumulative regret total. This uniform weighting can dramatically slow convergence in games where a single action can be a very costly mistake. An agent making such a mistake early in the training process can accumulate a massive, negative regret for the optimal actions. This "regret debt" can take a vast number of subsequent iterations to overcome, disproportionately influencing the strategy long after the initial exploration phase. This vulnerability of CFR+ to high-regret early iterations necessitated a paradigm shift from uniform regret accumulation to a dynamic, weighted approach, leading directly to the development of Discounted CFR.

2.1. The Weakness of Uniform Weighting in Regret Accumulation

DCFR was designed to solve the problem of high-cost mistakes that can stall convergence in CFR+. Consider a simple motivating example presented by Brown and Sandholm involving an agent choosing between three actions with payoffs of 0, 1, and -1,000,000, respectively.

* Iteration 1: CFR+ begins with a uniform random strategy, assigning a 1/3 probability to each action. The expected payoff is (1/3)*0 + (1/3)*1 + (1/3)*(-1,000,000) ≈ -333,333. The instantaneous regret for an action is its value minus this expected value, resulting in regrets of 0 - (-333,333) = 333,333 for the first action, and 1 - (-333,333) = 333,334 for the second.
* Massive Regret: These initial regrets are accumulated. On the next iteration, the algorithm will play the first two actions with roughly 50% probability each, and the third action not at all.
* Slow Convergence: Because the regret for the optimal action (payoff 1) is only slightly higher than the regret for the suboptimal action (payoff 0), it will take an extremely long time for the strategy to converge. It was calculated that it would take 471,407 iterations for the agent to learn to play the best action with 100% probability.

This scenario highlights a critical flaw: the enormous regret generated in a single early iteration has an enduring and detrimental impact. To mitigate this, a mechanism is needed to assign less weight to earlier regrets, allowing the algorithm to "forget" its initial, high-cost explorations more quickly.

2.2. The DCFR Algorithm

Discounted CFR (DCFR) is a family of algorithms designed to address this weakness by discounting prior iterations when calculating both the cumulative regrets and the average strategy. This gives more weight to recent information, accelerating convergence. Here, we adopt the notation from Xu et al., where j represents a decision node (or infoset) for the subsequent formulas.

The algorithm is parameterized by α, β, and γ, which control the rate of discounting. The update rule for cumulative regret R at decision node j on iteration t is:

Rt j = Rt−1 j ⊙ dt−1 j + rtj

Here, rtj is the instantaneous regret, and dt−1 j is a vector of discount factors applied element-wise (⊙). The discount factor dt-1 j[a] for an action a at iteration t depends on the sign of the cumulative regret from the end of the previous iteration, Rt-1 j[a]:

dt-1 j[a] = { (t^α / (t^α+1)) if Rt-1 j[a] > 0; (t^β / (t^β+1)) otherwise }

This allows for different discounting rates for positive regrets (controlled by α) and negative regrets (controlled by β). Similarly, the update for the cumulative strategy X is also discounted, controlled by γ:

Xt = Xt−1( (t−1) / t )^γ + ẋt

DCFR represented a powerful new direction by introducing the principle of discounting. This set the stage for a further refinement that would combine its core idea with the proven regret-clipping technique from CFR+.

3. The Evolution to DCFR+: Combining Discounting and Regret Clipping

The development of DCFR+ represents a principled synthesis rather than a complete replacement of its predecessors. This evolutionary step integrates the novel regret discounting mechanism from DCFR with the highly effective negative regret clipping technique from CFR+. By doing so, DCFR+ aims to capture the benefits of both approaches—accelerated "forgetting" of early mistakes and the immediate reuse of promising actions—within a single, coherent algorithmic framework.

3.1. Mathematical Formulation and Core Principle

DCFR+ is explicitly designed to combine the core ideas of its two most influential predecessors. As stated by Xu et al., it "incorporates regret discounting similar to DCFR and clips negative regrets akin to CFR+". This fusion is most apparent in its regret update rule.

The mathematical formulation for the DCFR+ cumulative regret update is:

Rt j = [ Rt−1 j * ((t−1)^α / ((t−1)^α+1)) + rtj ]+

The key distinction from DCFR lies in the [...]+ operation, which is equivalent to max(0, ...) and is applied to the entire result. First, the previous iteration's regret Rt−1 is discounted. Then, the current instantaneous regret rtj is added. Finally, the entire sum is clipped, ensuring the resulting cumulative regret Rt j is never negative. This update is not just different, but simpler and more unified than its predecessor's. DCFR required a conditional check on the sign of each action's prior regret before applying one of two different discount factors (α or β). DCFR+ replaces this with a single discount factor (α) followed by a single clipping operation on the final sum—a more elegant and computationally streamlined approach.

The average strategy update in DCFR+, however, remains identical to the one used in DCFR, controlled by the discounting parameter γ:

Xt = Xt−1( (t−1) / t )^γ + ẋt

This design choice retains the benefit of giving more weight to recent strategies while surgically applying the regret-clipping logic from CFR+ to the regret accumulation step.

3.2. Algorithmic Comparison: DCFR vs. DCFR+

The evolutionary step from DCFR to DCFR+ can be clearly seen by comparing their core regret update pseudocode side-by-side. The fundamental change is the replacement of separate discounting factors for positive and negative regrets with a single discount factor followed by a global clipping operation.

DCFR	DCFR+
for each action a:<br>   if prev_regret[a] > 0:<br>     discount = (t-1)^α / ((t-1)^α + 1)<br>   else:<br>     discount = (t-1)^β / ((t-1)^β + 1)<br>   new_regret[a] = prev_regret[a] * discount + inst_regret[a]	discount = (t-1)^α / ((t-1)^α + 1)<br> sum_regret = prev_regret * discount + inst_regret<br> new_regret = max(0, sum_regret)

The critical difference is that DCFR applies logic before the update based on the sign of each action's previous regret, while DCFR+ applies a single clipping operation to the resultant regret vector for the infoset. This algorithmic change is not merely an ad-hoc combination but is validated by a solid theoretical framework.

4. Theoretical Justification and Performance Analysis

For any new algorithm variant to be accepted, its intuitive appeal must be backed by a solid theoretical foundation. This section explores the principled basis for DCFR+, demonstrating that its design is not an arbitrary fusion of ideas. DCFR+ can be formally derived from the Online Mirror Descent (OMD) framework, a general and powerful tool for regret minimization. This connection provides a rigorous justification for its structure and helps explain its strong empirical performance.

4.1. Derivation from Online Mirror Descent (OMD)

The structure of DCFR+ is not coincidental; it can be derived by directly employing the Online Mirror Descent (OMD) framework to minimize weighted counterfactual regret. This connection is crucial because it moves the algorithm beyond a collection of empirical "hacks" (discounting, clipping) and demonstrates that it is a coherent, theoretically-grounded method for minimizing weighted regret. This elevates the algorithm's status from a clever hybrid to a principled design.

Within the OMD framework, the use of increasing weights wt for the loss at each iteration can be interpreted in two ways:

1. As an increasing learning rate, which causes the algorithm to adapt more aggressively in later iterations.
2. As a decreasing regularization term, which imposes less restriction on the strategy updates as training progresses.

Intuitively, this aligns with the goal of controlling regret growth in the early stages of training. By starting with a smaller learning rate or stronger regularization, the algorithm is less susceptible to the large, noisy regret signals generated by early, uninformed exploration.

4.2. Empirical Performance Gains

The theoretical elegance of DCFR+ is matched by its practical effectiveness. Experimental results reported by Xu et al. demonstrate that DCFR+ achieves the fastest convergence on large-scale poker games when compared to other prominent CFR variants. Its superior performance validates the hypothesis that combining discounting with regret clipping offers a distinct advantage.

Further reinforcing this conclusion, a specific instance of DCFR+ with hyperparameters α=1.5 and γ=4 was independently discovered through an evolutionary search for high-performance CFR algorithms. This automatically discovered variant also showed faster convergence than DCFR, providing strong empirical evidence that the DCFR+ structure is highly effective in practice.

Ultimately, DCFR+ is not merely an incremental improvement but a powerful synthesis. It validates the hypothesis that the practical speed of CFR+'s regret clipping and the accelerated adaptation of DCFR's discounting are not mutually exclusive but can be unified within a single, theoretically sound framework derived from Online Mirror Descent. This combination, validated by both its theoretical underpinnings and state-of-the-art empirical results, marks a significant milestone in the evolution of equilibrium-finding algorithms for imperfect-information games.
