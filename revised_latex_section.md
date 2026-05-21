# Revised Section: Dependency-aware Structural Reward

Below is the completely revised, scientifically accurate LaTeX section for your manuscript. 

It correctly generalizes the dependency reward mechanism to **both skeleton and tactic actions** (whenever they introduce local `have` or `let` declarations), presents the **reward formulation in a parameterized mathematical form** (using weights $w_{\mathrm{b}}$ and $w_{\mathrm{m}}$), and **correctly replaces all references to MCTS with Best-First Search**, perfectly aligning the text with the project's actual search framework.

---

```latex
\subsection{Dependency-aware Structural Reward}
\label{subsec:dep-reward}

For both skeleton and tactic actions, syntactic validity alone is an insufficient indicator of quality. An action $a$ applied to state $s$ may introduce a set of local declarations (binders) via \texttt{have} or \texttt{let} statements. In a \textit{skeleton action}, these binders correspond to child subgoals that are eventually solved by child proof search branches and stitched back into the parent template. In a \textit{tactic action}, these binders represent intermediate auxiliary claims or lemmas introduced locally within a single-step proof script. 

In either case, the language model might introduce redundant local claims that compile successfully but are completely irrelevant to the final proof, or it may leave unresolved declarations containing placeholders (\texttt{sorry}) that render the resulting proof incomplete. To evaluate whether these generated declarations actually contribute to the final proof, GammaZero performs a De Bruijn-aware expression-tree dependency analysis over the elaborated Lean proof term.

\subsubsection{Greedy Garbage Pruning for Scoring Refinement}
Lean's elaborated expression tree may sometimes contain references to local binders that appear superficially in the surface syntax but are logically redundant to the core proof term. To distinguish genuinely utilized dependencies from redundant artifacts, the environment performs a greedy pruning pass prior to scoring. 

Let $\mathcal{V}$ be the set of candidate local variables introduced by the action $a$. The system attempts to remove each candidate variable $v \in \mathcal{V}$ from the proof script and re-runs Lean verification. If the proof remains complete without $v$, the variable is confirmed as redundant and is pruned. This greedy verification pass ensures that the subsequent dependency analysis is performed on a minimized, logically tight proof representation.

\subsubsection{Expression-Tree Dependency Traversal}
Following the greedy pruning pass, GammaZero traverses the elaborated Lean 4 expression tree (represented as a JSON-serializable syntax structure). The analyzer recursively traverses the tree nodes, including \texttt{bvar}, \texttt{fvar}, \texttt{app}, \texttt{lam}, \texttt{forallE}, and \texttt{letE} structures, to determine if each remaining variable in $\mathcal{V}$ is referenced by the final proof expression.

To handle variable shadowing and name changes inside the Lean compiler, the traversal is strictly \textit{De Bruijn-aware}. When the analyzer enters a binder node (such as \texttt{lam}, \texttt{forallE}, or \texttt{letE}), it increments and shifts the target De Bruijn index:
\[
\text{index}_{\text{target}} \leftarrow \text{index}_{\text{target}} + 1
\]
so that bound variable references (\texttt{bvar}) in the inner body are attributed to the correct local declaration. The analyzer also detects the presence of the placeholder axiom \texttt{sorryAx} in the expression tree. 

Combining usage detection with placeholder verification yields four structural cases for each candidate declaration:
\begin{itemize}
    \item \textbf{Solved and used (Core):} The local declaration has a fully solved proof body and is actively utilized by the elaborated proof expression.
    \item \textbf{Unresolved and used:} The local declaration still contains a placeholder (\texttt{sorryAx}) but is utilized by the proof term, meaning the proof remains incomplete. Such cases are classified as structural failures.
    \item \textbf{Solved but unused (Benign):} The local declaration is complete but redundant (e.g., it was successfully pruned or is not referenced by the elaborated proof term).
    \item \textbf{Unresolved and unused (Malignant):} The local declaration is unfinished and completely unused by the final proof term.
\end{itemize}

\subsubsection{Reward Formulation}
Let $n_{\mathrm{core}}$, $n_{\mathrm{benign}}$, and $n_{\mathrm{malignant}}$ denote the counts of core (solved-and-used), benign (solved-but-unused), and malignant (unresolved-and-unused) local declarations, respectively. We formulate the dependency-aware structural reward as a normalized usefulness ratio with parameterized penalty weights:
\[
r_{\mathrm{dep}}(s,a) = \frac{n_{\mathrm{core}}}{n_{\mathrm{core}} + w_{\mathrm{b}}\,n_{\mathrm{benign}} + w_{\mathrm{m}}\,n_{\mathrm{malignant}}}
\]
where $w_{\mathrm{b}} \ge 0$ is the penalty hyperparameter for benign (solved-but-unused) work, and $w_{\mathrm{m}} > w_{\mathrm{b}}$ is the penalty hyperparameter for malignant (unresolved-and-unused) declarations. If the denominator is zero (i.e., the action introduces no local declarations and does not fail), the reward defaults to $1.0$.

This parameterized formulation gracefully reflects the logical trade-offs of structural waste:
\begin{itemize}
    \item A solved-but-unused declaration ($n_{\mathrm{benign}}$) is mildly penalized via a small weight $w_{\mathrm{b}}$ because, although it does not block the final compilation, it represents redundant search and API generation effort.
    \item An unresolved-and-unused declaration ($n_{\mathrm{malignant}}$) is heavily penalized via a larger weight $w_{\mathrm{m}}$ because it introduces non-functional, incomplete scaffolding into the workspace without any logical utility.
\end{itemize}
In our active search implementation, we configure these penalty weights as $w_{\mathrm{b}} = 0.5$ and $w_{\mathrm{m}} = 2.0$. This dependency reward provides the best-first search algorithm with a fine-grained, dependency-aware guiding signal that distinguishes high-quality structured proofs from redundant or bloated reasoning chains.
```
