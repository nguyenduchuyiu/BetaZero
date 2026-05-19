# GammaZero System Description

## Abstract

GammaZero is a system for proving Lean 4 theorems with language models. It represents proof search as an AND/OR graph. Each proof state can be solved by one successful action. An action can be a direct Lean tactic, or a proof skeleton that creates smaller proof states. The search uses a priority queue over open proof states. The priority score is a custom heuristic inspired by proof-number search. It gives high priority to states that are likely to complete an active proof decomposition, for example the last unsolved child of a committed skeleton.

This document explains the system at the level of concepts and data flow. It assumes the reader has not read the code.

## 1. Basic Terms

Lean 4 is the theorem prover used to check every proof. A generated proof is accepted only if Lean accepts it.

A tactic is a short Lean proof script for the current goal. If it closes the goal, the proof state is solved.

A skeleton is a larger Lean proof outline. It may contain `sorry` placeholders. Each placeholder becomes a child proof state. The system then tries to solve those child states separately.

A proof state is a pair of local hypotheses and a goal. In the code this is `ProofState`.

An action is one attempt to solve a proof state. In the code this is `Action`. There are two action kinds:

- `tactic`: direct proof attempt for the current state.
- `skeleton`: decomposition into child proof states.

The main idea is simple: try direct tactics first. If direct tactics do not solve the state, ask the model for a skeleton. Then solve the children of the skeleton. If all children are solved, the skeleton solves its parent.

## 2. AND/OR Proof Graph

The search graph is stored in `gammazero/search/graph/and_or_graph.py`.

A proof state is an OR node. It is solved when at least one of its actions is solved. This matches proof search: one valid proof is enough.

An action is an AND node when it is a skeleton. A skeleton is solved only when all of its child proof states are solved. This matches decomposition: every subgoal introduced by the skeleton must be proved.

A tactic action has no child proof states. It is solved if Lean verifies that the tactic closes the goal.

The graph stores:

- all actions attached to each proof state,
- the parent proof state of each action,
- the current status of each proof state and action,
- the environment reward `r_env`,
- the dependency reward `r_dep`,
- the depth of each proof state.

Statuses are propagated upward. If a child state is solved, its parent skeleton may become solved. If a skeleton becomes solved, its parent proof state becomes solved. If all useful actions for a state fail, the state can be marked failed.

## 3. Lean Verification

Lean verification is the ground truth. The language model can propose code, but Lean decides whether the code is valid.

The verifier runs Lean through a persistent REPL process. The worker first imports Mathlib and stores the resulting Lean environment. Later verification calls reuse this environment. This avoids importing Mathlib again for every candidate proof.

Each verification call returns structured information:

- whether Lean accepted the code,
- whether the theorem is complete,
- the list of remaining `sorry` placeholders,
- Lean errors and warnings.

The worker also protects the search from stuck Lean processes. If a verification call takes too long, the worker kills the process and starts a new one.

## 4. Model Outputs

The policy model is asked for two kinds of outputs.

For a tactic request, the model receives the current proof state and returns Lean code intended to close that state directly.

For a skeleton request, the model receives the current proof state and returns a Lean proof outline. The outline may contain `sorry`. Each `sorry` describes a child goal that the search can solve later.

The system extracts Lean code from model text before sending it to Lean. If the model output violates the expected format or policy rules, the action is rejected or repaired.

## 5. Repairing Generated Lean Code

Many generated proofs are close to useful but do not compile. GammaZero uses a repair component called `Sorrifier`, implemented in `gammazero/search/sorrifier`.

The repair process is conservative. It does not try to invent a new proof. It removes or replaces invalid parts so that Lean can still expose useful structure.

Common repair actions include:

- replace a failing proof block with `sorry`,
- remove a malformed line,
- close an unfinished block with `sorry`,
- remove a larger block when local repair does not make progress.

The repaired code is scored by how much original code survived. This score is `r_env`.

## 6. Dependency Analysis

Skeletons can contain child lemmas that are not actually used by the final proof. Some of those child lemmas may still contain `sorry`. Lean rejects the whole theorem if any `sorry` remains, even when the remaining placeholder is in an unused lemma.

GammaZero handles this by analyzing dependencies in the Lean expression tree. The analyzer checks which local declarations are used by the final proof term.

Each child lemma is classified into one of four groups:

- `core_solved`: solved and used by the final proof,
- `core_failed`: unsolved and used by the final proof,
- `benign`: solved but not used,
- `malignant`: unsolved and not used.

The names are implementation labels. The important distinction is whether a lemma is used by the final proof. If an unused lemma is present, the proof stitcher can remove it from the final proof text.

This lets the system keep a valid proof even when the model generated extra unused structure.

## 7. Search Procedure

GammaZero uses a best-first search over the AND/OR graph. The search keeps a priority queue of open proof states. At each step, it selects the open states with the highest heuristic scores.

For each selected state, the search performs the following operations:

1. Try tactic actions for the state.
2. Verify each tactic in Lean.
3. Record the best tactic reward seen for the state.
4. If tactic attempts are not enough, ask for skeleton actions.
5. Score the skeleton actions.
6. Choose one skeleton to focus on.
7. Add the skeleton children as new proof states.
8. Recompute graph statuses.
9. Reinsert still-open states into the priority queue.

The search has finite limits from configuration. Examples are the maximum number of actions, maximum depth, maximum tactic attempts per state, and maximum skeleton attempts per state. These limits keep a hard problem from consuming the whole run.

The search also keeps only the best open states in the queue. This is not a separate algorithmic idea. It is a practical limit on how many open states are kept for later work.

## 8. Tactic Attempts Before Skeletons

For a new proof state, GammaZero tries direct tactics before asking for a skeleton. This is useful because many goals can be solved without decomposition.

A skeleton is considered only after the state has received enough tactic attempts. The default configuration uses `min_tactic_before_skeleton`.

If a tactic has a strong `r_env` score but does not yet solve the state, the system keeps trying tactics for longer. The reason is that a good partial tactic may be close to a complete proof, and a skeleton may be unnecessary.

This rule avoids decomposing every goal too early.

## 9. Skeleton Commitment

When several skeletons are available for one proof state, the system chooses one skeleton as the active decomposition. This is called commitment.

The committed skeleton receives focused search effort. Its children are inserted into the queue. The parent state does not keep asking for new skeletons while the committed skeleton is active.

Other good skeletons can be stored as reserved skeletons. If the committed skeleton fails, a reserved skeleton can be activated later.

A committed skeleton fails when one of its required child states fails. It may also become stale if repeated search rounds do not solve more of its children. In that case, the system can move to a reserved skeleton.

This design prevents the search from spreading across many incompatible decompositions at the same parent state.

## 10. State Heuristic

The state heuristic is implemented in `gammazero/search/rollout/heuristic.py`.

The score of a proof state is:

```text
score(state)
  = incoming_skeleton_weight * incoming_skeleton_score
  + best_tactic_weight * best_tactic_r_env
  + committed_skeleton_progress_bonus
  - depth_penalty * depth
  - tactic_retry_penalty * tactic_tries
  - skeleton_retry_penalty * skeleton_tries
  - bad_skeleton_round_penalty * bad_skeleton_rounds
```

The terms have the following meanings.

`incoming_skeleton_score` is the score of the skeleton that created the state. A child from a promising skeleton inherits some of that promise.

`best_tactic_r_env` is the best environment reward among tactics already tried on the state. A state with a good partial tactic remains interesting.

`committed_skeleton_progress_bonus` is added when the state is a child of the currently committed skeleton of its parent. The bonus is larger when the state is the last open child of that skeleton. This is the main mechanism that pushes the search to finish an active decomposition.

The depth penalty prefers shallower states when other signals are similar.

The retry penalties reduce the score of states that have already consumed many tactic or skeleton attempts.

The bad skeleton penalty reduces the score of a state when recent skeleton attempts did not create useful child states.

## 11. Skeleton Heuristic

The skeleton score is also implemented in `heuristic.py`.

The score of a skeleton action is:

```text
score(skeleton)
  = skeleton_r_env_weight * r_env
  + skeleton_parent_score_weight * parent_last_score
  + child_count_score(number_of_children)
  - skeleton_depth_penalty * parent_depth
  - skeleton_sorrified_penalty, if the skeleton was repaired
```

`r_env` measures how much of the generated skeleton survived verification and repair.

`parent_last_score` lets a skeleton inherit part of the priority of the parent state.

`child_count_score` prefers a small, useful number of children. A skeleton with zero children is penalized. One or two children are treated as good. More children receive a mild linear penalty.

The depth penalty discourages decompositions that start too deep in the graph.

The repair penalty mildly discourages skeletons that needed repair before they became usable.

## 12. Rewards

GammaZero records two rewards for actions.

The first reward is `r_env`. It measures how much generated code survived after repair. The system compares the original generated code with the repaired code. A proof that survives mostly intact receives a higher score. A proof that is almost entirely replaced by `sorry` receives a low score.

The second reward is `r_dep`. It measures whether the skeleton introduced useful lemmas. The reward is high when solved child lemmas are used by the final proof. It is lower when the skeleton contains unused solved lemmas or unused failed lemmas.

The dependency reward has the form:

```text
r_dep = n_core / (n_core + 0.5 * n_benign + 2.0 * n_unused_failed)
```

Here `n_core` is the number of solved child lemmas used by the final proof. `n_benign` is the number of solved but unused child lemmas. `n_unused_failed` is the number of unsolved and unused child lemmas.

These rewards serve two purposes. They guide search during a rollout, and they provide training signals after the rollout.

## 13. Backup Values

After a rollout, the graph computes values for actions. A tactic value is based on its direct rewards. A skeleton value is based on its rewards and the values of its children.

For a skeleton, the child contribution uses the weakest child value. This reflects the AND structure: a skeleton is only as strong as its hardest required child.

Failed skeletons keep their own recorded reward for analysis, but they do not contribute value upward to solve the parent state.

## 14. Final Proof Construction

When child proof states are solved, the system can insert their proofs back into the parent skeleton. This is handled by the proof stitcher.

The stitcher replaces target `sorry` placeholders with child proofs. It then verifies the full theorem again in Lean.

After verification, dependency analysis may remove unused local declarations. The final proof is accepted only if Lean accepts the resulting theorem without unresolved placeholders.

## 15. Configuration

The main runtime configuration is in `configs/api.yaml`. The `Config` class loads this file and validates the `heuristic` section against the current `SimpleHeuristicScorer` constructor. This matters because stale heuristic keys should fail early instead of silently changing behavior.

Important configuration groups are:

- search limits: `max_depth`, `max_nodes`,
- tactic sampling: `initial_tactic_k`, `retry_tactic_k`, `max_tactic_per_state`,
- skeleton sampling: `initial_skeleton_k`, `retry_skeleton_k`, `max_skeleton_per_state`,
- state queue limits: `state_beam_width`, `state_beam_per_depth`,
- skeleton commitment: `skeleton_commitment`, `max_reserved_skeletons_per_state`, `commit_stale_rounds_before_fallback`,
- heuristic weights: the keys under `heuristic`.

The configuration does not define the algorithm by itself. It only sets limits and weights for the components described above.

## 16. Summary

GammaZero combines direct proof attempts and structured decomposition.

The AND/OR graph records how proof states, tactics, and skeletons depend on one another.

Lean verification is used for every generated proof fragment.

The search uses a priority queue and a simple heuristic. The heuristic gives high priority to states that are likely to finish an active skeleton, especially the last unsolved child.

Skeleton commitment keeps the search focused on one decomposition at a time while still allowing fallback to reserved skeletons.

Repair and dependency analysis make generated Lean code more useful without trusting invalid code. The system can preserve useful proof structure, remove unused declarations, and verify the final theorem again in Lean.
