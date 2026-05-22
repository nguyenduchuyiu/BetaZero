# New Section: Lean Feedback & Critique-Correction Loop

Below is a complete, academically elegant LaTeX section for your manuscript describing **GammaZero's Lean Feedback and Critique-Correction Loop**. 

It is **100% faithful to the implementation details** found in `gammazero/policy/prompt.py` (including the precise formatting blocks, instructions, and sliding window parameters):

---

```latex
\subsection{Lean Feedback and Critique-Correction Loop}
\label{subsec:lean-feedback}

Syntactic validation alone does not prevent the language model from generating proofs that fail to elaborate due to type mismatches, unresolved identifiers, or logical gaps. Rather than treating failures as terminal search states, GammaZero implements a closed-loop \textit{compiler-in-the-loop critique-correction mechanism}. When a generated tactic or skeleton action fails to verify, the Lean 4 compiler's rich diagnostic error trace is captured, structured, and injected back into the language model's prompt context during search retries.

\subsubsection{Structured Diagnostic Feedback Blocks}
When a tactic or skeleton action fails verification, the environment captures the failing code snippet alongside Lean's precise stdout/stderr compiler diagnostics. The system formats this diagnostic telemetry into a structured feedback block:
\begin{align*}
\mathcal{F}(a) = \Big[ &\texttt{"FAILED CHECKED CODE:"} \mathbin{\Vert} \text{code}_{\text{failed}} \\
&\mathbin{\Vert} \texttt{"LEAN ERROR FEEDBACK:"} \mathbin{\Vert} \text{feedback}_{\text{Lean}} \Big]
\end{align*}
where $\Vert$ represents string concatenation. In our implementation, this formatting block is structured as:
\begin{verbatim}
FAILED CHECKED CODE:
```lean4
[Failing proof or skeleton script]
```

LEAN ERROR FEEDBACK:
[Lean compiler error messages, e.g., "unknown identifier 'bad'"]
\end{verbatim}

\subsubsection{Multi-Turn Context-Preserving Retries}
During search expansion, if a state requires additional exploration budget (e.g., up to $k_{\mathrm{retry}}$ attempts), the prompt builder compiles these failure blocks into the model's history. The system appends a steering directive instructing the agent to learn from the compiler error and avoid repeating its mistake:
\begin{verbatim}
PREVIOUS TACTIC ATTEMPTS FAILED.
Use the Lean feedback below to produce a NEW tactic proof for the same goal.
Do not repeat the failed tactic. Preserve the exact theorem signature.
\end{verbatim}

To prevent the context window from bloating and to keep the model focused on the most recent and relevant mistakes, GammaZero employs a sliding window of historical failures. Let $\mathcal{H}_t = \{\mathcal{F}(a_1), \mathcal{F}(a_2), \ldots, \mathcal{F}(a_t)\}$ be the set of historical failure blocks accumulated on state $s$. The prompt builder restricts the context to the $N$ most recent failures:
\[
\text{Prompt}_{\text{retry}}(s) = \text{Prompt}_{\text{base}}(s) \cup \bigcup_{i = \max(1, t - N + 1)}^{t} \mathcal{F}(a_i)
\]
In our deployment, we set $N = 3$. This context-bounded feedback loop provides the language model with an active, real-time debugging signal, transforming static proof generation into an adaptive, self-correcting dialogue with the Lean 4 compiler.
```
