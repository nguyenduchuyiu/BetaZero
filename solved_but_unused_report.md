# Diagnostic Report: Solved But Unused Subgoals

This report documents concrete examples of **"Solved but Unused" (Benign)** subgoals extracted from your GammaZero search rollouts on `miniF2F-valid` and `miniF2F-test`. 

These subgoals were successfully verified and solved by child proof search branches, but they were not referenced or utilized in the final completed proof term of their parent state.

---

## 1. Key Solved-But-Unused Examples

Below is a curated table of specific instances found in your actual experiment rollouts.

| Theorem/Problem File | Subgoal ID & Label | Subgoal Goal (Lean Proposition) | Parent State ID & Label | Parent Goal (Lean Proposition) |
| :--- | :--- | :--- | :--- | :--- |
| **algebra_apbmpcneq0...** (Valid) | `state_15` (`h_eq_zero`) | `Polynomial.C x + Polynomial.C y * Polynomial.X + Polynomial.C z * Polynomial.X ^ 2 = 0` | `state_12` (`h_linear_independent`) | `∀ (x y z : ℚ), ↑x + ↑y * m + ↑z * m ^ 2 = 0 → x = 0 ∧ y = 0 ∧ z = 0` |
| **algebra_apbmpcneq0...** (Valid) | `state_16` (`h_x`) | `x = 0` | `state_12` (`h_linear_independent`) | `∀ (x y z : ℚ), ↑x + ↑y * m + ↑z * m ^ 2 = 0 → x = 0 ∧ y = 0 ∧ z = 0` |
| **algebra_apbmpcneq0...** (Valid) | `state_17` (`h_y`) | `y = 0` | `state_12` (`h_linear_independent`) | `∀ (x y z : ℚ), ↑x + ↑y * m + ↑z * m ^ 2 = 0 → x = 0 ∧ y = 0 ∧ z = 0` |
| **algebra_apbmpcneq0...** (Valid) | `state_18` (`h_z`) | `z = 0` | `state_12` (`h_linear_independent`) | `∀ (x y z : ℚ), ↑x + ↑y * m + ↑z * m ^ 2 = 0 → x = 0 ∧ y = 0 ∧ z = 0` |
| **algebra_ineq_nto1onlt2m1on** (Valid) | `state_38` (`h_f_n_le_f3`) | `Real.log ↑n / ↑n ≤ Real.log 3 / 3` | `state_32` (`h_case3`) | `n ≥ 3 → Real.log ↑n / ↑n ≤ Real.log 2 / 2` |
| **aime_1984_p5** (Test) | `state_34` (`h_sum_val`) | `logb 2 a + logb 2 b = 9` | `state_31` (`h_log_prod`) | `logb 2 (a * b) = 9` |
| **amc12_2001_p2** (Test) | `state_8` (`h_list`) | `∃ b a, Nat.digits 10 n = [b, a]` | `state_1` (`h_digits`) | `∃ a b, a < 10 ∧ b < 10 ∧ a ≠ 0 ∧ n = 10 * a + b` |
| **amc12a_2002_p21** (Test) | `state_8` (`h_period`) | `∀ (n : ℕ), u (n + 60) = u n` | `state_0` (`root_goal`) | `IsLeast Sn 1999` |
| **amc12a_2002_p21** (Test) | `state_23` (`h_decomp`) | `S 1998 = 33 * ∑ k ∈ range 60, u k + ∑ k ∈ range 18, u k` | `state_22` (`h_S_1998`) | `S 1998 ≤ 10000` |

---

## 2. Why Do Solved-But-Unused Subgoals Occur?

In an AND/OR search tree, a subgoal being **solved but unused** is a natural byproduct of search exploration and path redundancy. There are three primary reasons this happens in GammaZero:

### A. Search Path Diversification (OR-Node Competition)
A parent state (OR-node) $s$ can expand into **multiple alternative actions** (tactic actions or skeleton decompositions). 
* Suppose the parent state $s$ first expands a skeleton action $a_{\mathrm{skel}}$, which generates child subgoals $s'_1, s'_2$. 
* While child search processes are actively working to solve $s'_1$ and $s'_2$, the best-first search queue might allocate a budget to try a newly generated single-step **tactic action** $a_{\mathrm{tactic}}$ directly on $s$.
* If $a_{\mathrm{tactic}}$ successfully closes $s$ directly, then $s$ is marked as `SOLVED` via $a_{\mathrm{tactic}}$. 
* Even if the child search processes eventually succeed in solving $s'_1$ and $s'_2$, the final stitched proof for $s$ will use the direct tactic proof from $a_{\mathrm{tactic}}$, making the solved subgoals $s'_1$ and $s'_2$ redundant (unused) in the active proof term.

### B. Pruning / Patching of Redundant Scaffolding
Sometimes the skeleton generator (LLM) proposes a decomposition that introduces three subgoals: `have h1 := sorry`, `have h2 := sorry`, and `have h3 := sorry`.
* During elaboration and stitching, the final synthesis block might only require `h1` and `h2` to close the main goal (e.g., `exact my_lemma h1 h2`, completely ignoring `h3`).
* Even if a tactic branch successfully solves `h3` (making `h3` `SOLVED`), the De Bruijn expression-tree analyzer traverses the elaborated parent term and finds that the variable `h3` is never referenced.
* Consequently, `h3` is classified as **solved but unused** (benign redundancy).

### C. Over-Decomposition of Known Lemmas
In the case of **amc12a_2002_p21** (`state_8` `h_period` with goal `∀ n, u (n + 60) = u n`), the model generated a subgoal to prove that the sequence modulo 10 has a period of 60.
* Although this subgoal was solved by a child branch, the parent proof was ultimately closed using a direct cyclic tactic or standard Mathlib modular arithmetic simplification that bypassed the explicit period-60 declaration.
* This represents a classic "over-decomposition" where the model generates a valid intermediate claim, but the compiler succeeds via a shorter, more direct path.

---

## 3. Methodological Importance for Your Thesis/Paper
When discussing this in your evaluation or methodology sections, you can highlight this as follows:
> "The presence of **solved but unused (benign)** subgoals ($n_{\mathrm{benign}}$) demonstrates the exploratory resilience of the GammaZero search. By exploring multiple parallel proof paths (diversifying via both single-step tactic actions and skeleton decompositions), the search frequently finds direct shortcuts that bypass complex subgoal scaffolding. Our dependency-aware reward hyperparameter $w_{\mathrm{b}} = 0.5$ successfully penalizes the computational waste of these redundant subgoals without penalizing the overall correctness of the proof."
