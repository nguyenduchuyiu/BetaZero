# Revised Section: AND/OR Backup

Below is the completely revised, scientifically precise LaTeX section for **AND/OR Backup** in your manuscript. 

It is **100% aligned with your actual code** inside `and_or_graph.py` (specifically lines 209-254 of the `backup` method) and your unit tests:
1. It reveals that the variable $W_{\mathrm{solve}}$ is **legacy and completely unused** by your codebase (confirmed by the test `test_solved_tactic_backup_uses_r_dep_instead_of_w_solve`).
2. It **unifies** the backup equations for both tactics and skeletons into a single, mathematically elegant, and robust formula using the child subgoal set $\mathcal{C}(a)$ (where $\mathcal{C}(a) = \emptyset$ for tactic actions, so the future value naturally defaults to $0$).

---

```latex
\subsection{AND/OR Backup}
\label{subsec:and-or-backup}

The progressive best-first search operates over a dynamic AND/OR proof graph where states (OR nodes) represent subgoals to be proved, and actions (AND nodes) represent tactic or skeleton proof attempts. Once the search reaches its limits or closes the root goal, GammaZero backs up values through the AND/OR graph to evaluate the quality of the explored paths. 

Rather than utilizing separate heuristic equations with hardcoded solve bonuses (such as a legacy $W_{\mathrm{solve}}$ parameter, which is deprecated in favor of structured dependency rewards), GammaZero employs a mathematically unified backup formulation. For any action $a$ (which can be either a tactic or a skeleton action) applied to a proof state $s$, its action value $Q(s,a)$ is formulated as:
\[
Q(s,a) = r_{\mathrm{env}}(s,a) + r_{\mathrm{dep}}(s,a) + \mathds{1}_{\mathrm{solved}}(s,a) \cdot \gamma \min_{s' \in \mathcal{C}(a)} V(s')
\]
where:
\begin{itemize}
    \item $r_{\mathrm{env}}(s,a) \in [0,1]$ is the environment survival reward representing syntactic code preservation.
    \item $r_{\mathrm{dep}}(s,a) \in [0,1]$ is the dependency-aware structural reward reflecting the usefulness of the introduced local declarations.
    \item $\mathcal{C}(a)$ is the set of child subgoals introduced by action $a$. For a local \textit{tactic action}, the action does not decompose the state, so $\mathcal{C}(a) = \emptyset$. We naturally define the minimum over an empty set to default to zero: $\min_{s' \in \emptyset} V(s') = 0$.
    \item $\mathds{1}_{\mathrm{solved}}(s,a) \in \{0, 1\}$ is a solved indicator variable. For a tactic action, it equals $1$ if the tactic successfully closes the goal without leaving any remaining placeholders. For a skeleton action, it equals $1$ only if all child subgoals in $\mathcal{C}(a)$ are successfully solved.
    \item $\gamma \in (0, 1]$ is a discount factor for future proof steps.
\end{itemize}

This formulation elegantly generalizes both action families:
\begin{itemize}
    \item \textbf{Tactic Actions:} Since $\mathcal{C}(a) = \emptyset$, the future return term drops out, and the value simplifies to:
    \[
    Q(s,a) = r_{\mathrm{env}}(s,a) + r_{\mathrm{dep}}(s,a)
    \]
    where the dependency-aware reward $r_{\mathrm{dep}}(s,a)$ acts as the structural quality and solve bonus for the closed goal.
    \item \textbf{Skeleton Actions:} If the skeleton is unresolved ($\mathds{1}_{\mathrm{solved}}(s,a) = 0$), it receives only its local environment and dependency credit: $Q(s,a) = r_{\mathrm{env}}(s,a) + r_{\mathrm{dep}}(s,a)$ (where $r_{\mathrm{dep}}$ defaults to $0$ if not solved). If the skeleton is fully solved ($\mathds{1}_{\mathrm{solved}}(s,a) = 1$), it receives its full local credit plus the discounted value of its child branches:
    \[
    Q(s,a) = r_{\mathrm{env}}(s,a) + r_{\mathrm{dep}}(s,a) + \gamma \min_{s' \in \mathcal{C}(a)} V(s')
    \]
\end{itemize}

The $\min$ operator implements the strict AND semantics of proof decomposition: the value of a skeleton is bottlenecked by its weakest child branch. This prevents the search from attributing value to a decomposition that solves only trivial subgoals while leaving an essential mathematical obligation unresolved.

For any state (OR node) $s$, its state value $V(s)$ represents the best available proof path starting from that state, which follows ordinary OR semantics:
\[
V(s) = \max_{a \in \mathcal{A}(s)} Q(s,a)
\]
where $\mathcal{A}(s)$ is the set of candidate actions expanded from state $s$. Thus, the search graph propagates solved values upward via a minimax-style backup: performing a $\max$ backup over alternative actions at OR nodes and a $\min$ backup over conjunctive children at AND nodes.
```
