# Trajectory Case Study: aime_1991_p9

Below is the highly polished, academically elevated LaTeX section for **Section 8.8 Trajectory Case Study** of the GammaZero manuscript. It integrates all the quantitative and qualitative findings extracted directly from the successful `aime_1991_p9` rollout search graph.

```latex
\subsection{Trajectory Case Study: \texttt{aime\_1991\_p9}}
\label{subsec:case-study}

A representative deep trajectory that highlights the structural power of GammaZero is the complete solution of the theorem \texttt{aime\_1991\_p9}.
This complex trigonometry and algebra problem requires proving:
\[
\forall (x : \mathbb{R}) (m : \mathbb{Q}), \left( \frac{1}{\cos x} + \tan x = \frac{22}{7} \right) \land \left( \frac{1}{\sin x} + \cot x = m \right) \implies m.\mathrm{den} + m.\mathrm{num} = 44
\]
GammaZero successfully solved this theorem at search depth 7, using 506 generated actions and 916 Lean verification calls. This case study demonstrates how GammaZero's graph-guided search successfully navigates deep nested skeleton decompositions, coordinates local tactic proofs, and handles dependency classes to reconstruct a fully verified proof.

\subsubsection{Search Tree Complexity and Exploration Space}
The global search graph for this problem contained 545 nodes: 39 OR nodes (subgoal states) and 506 AND nodes (candidate actions). The search space exhibits heavy exploration at critical mathematical junctions:
\begin{itemize}
    \item \textbf{Root Subgoal (\texttt{state\_0}):} Expanded 36 candidate actions before settling on the successful skeleton decomposition (\texttt{action\_32}).
    \item \textbf{Half-Angle Equation (\texttt{state\_1}):} Proving $\tan(x/2) = 15/29$ expanded 36 candidate actions before identifying the successful rational parameterization skeleton (\texttt{action\_65}).
    \item \textbf{Quotient Simplification (\texttt{state\_28}):} Proving $\frac{1 + \cos x}{\sin x} = \frac{1}{\tan(x/2)}$ expanded 50 candidate actions before discovering the double-angle expansion skeleton (\texttt{action\_476}).
\end{itemize}
The priority score successfully prioritized promising subgoals, allowing the search to go up to depth 7 (e.g., proving the non-degeneracy condition $\cos^2(x/2) \neq 0$ at \texttt{state\_8}), while leaving 14 invalid or highly complex alternative subgoals (e.g., \texttt{state\_15} to \texttt{state\_20} and \texttt{state\_35} to \texttt{state\_38}) completely abandoned.

\subsubsection{Committed Skeleton Path and Subgoal Decomposition}
The committed proof path showcases three core categories of skeleton decompositions, which are structurally and logically organized into dependency classes:

\begin{enumerate}
    \item \textbf{Algebraic and Rational Parameterization (Dependency Class 1):} 
    At depth 0 (\texttt{state\_0}), \texttt{action\_32} decomposes the goal by introducing the half-angle tangent value $\tan(x/2) = 15/29$ as an intermediate landmark, reducing the root goal to four subgoals:
    \[
    \text{subgoals} = \{ \tan(x/2) = 15/29, \, m = 29/15, \, m.\mathrm{num} = 29, \, m.\mathrm{den} = 15 \}
    \]
    Similarly, at depth 1 (\texttt{state\_1}), \texttt{action\_65} rationalizes the trigonometric equation by introducing:
    \[
    \frac{1}{\cos x} + \tan x = \frac{1 + \tan(x/2)}{1 - \tan(x/2)}
    \]
    reducing the task to proving this trigonometric identity (\texttt{state\_2}) and solving the pure algebraic equation $\frac{1+t}{1-t} = \frac{22}{7} \implies t = \frac{15}{29}$ (\texttt{state\_21}).
    
    \item \textbf{Rational Parameterization Identities (Dependency Class 2):}
    At depth 2 (\texttt{state\_2}), \texttt{action\_98} introduces the formal parameter $t = \tan(x/2)$ and establishes the standard rational parameterizations for $\cos x$ and $\tan x$:
    \[
    \cos x = \frac{1-t^2}{1+t^2} \quad (\texttt{state\_3}), \qquad \tan x = \frac{2t}{1-t^2} \quad (\texttt{state\_13})
    \]
    and the algebraic identity:
    \[
    \frac{1}{\frac{1-t^2}{1+t^2}} + \frac{2t}{1-t^2} = \frac{1+t}{1-t} \quad (\texttt{state\_14})
    \]
    
    \item \textbf{Double-Angle Expansion (Dependency Class 3):}
    To prove $\cos x = \frac{1-t^2}{1+t^2}$ (\texttt{state\_3}), \texttt{action\_133} decomposes the relation into double-angle components at depth 4:
    \[
    \cos x = \cos^2(x/2) - \sin^2(x/2) \quad (\texttt{state\_4})
    \]
    \[
    \cos^2(x/2) = \frac{1}{1+t^2} \quad (\texttt{state\_5}), \qquad \sin^2(x/2) = \frac{t^2}{1+t^2} \quad (\texttt{state\_12})
    \]
    
    \item \textbf{Non-Degeneracy and Division Verification (Dependency Class 4):}
    To establish $\cos^2(x/2) = \frac{1}{1+t^2}$ (\texttt{state\_5}), \texttt{action\_170} requires proving the non-degeneracy condition $\cos(x/2) \neq 0$ (\texttt{state\_11}). Furthermore, at depth 6 (\texttt{state\_7}), proving the standard secant identity $1 + \tan^2(x/2) = \sec^2(x/2)$ (\texttt{state\_7}) creates a sub-subgoal $\cos^2(x/2) \neq 0$ (\texttt{state\_8}). 
\end{enumerate}

\subsubsection{Coordination of Local Tactic Proofs}
While skeletons established the macro-logical structure of the proof, local tactic actions proved the leaf subgoals:
\begin{itemize}
    \item \textbf{Algebraic Solvers:} \texttt{state\_21} was closed by \texttt{action\_344} using a combination of \texttt{field\_simp} and \texttt{linarith} in 6 lines of code. \texttt{state\_14} was proved by \texttt{action\_339} using \texttt{by\_cases h : 1 + t = 0} to handle potential division-by-zero singularities.
    \item \textbf{Trigonometric Identities:} The double-angle identity for $\sin x$ (\texttt{state\_30}) was solved by \texttt{action\_485} using \texttt{rw [← Real.sin\_two\_mul], mul\_div\_cancel₀}. The fundamental Pythagorean identity at depth 7 (\texttt{state\_9}) was proved by \texttt{action\_289} using \texttt{field\_simp} and \texttt{rw [Real.sin\_sq\_add\_cos\_sq]}.
    \item \textbf{Non-Degeneracy Proofs by Contradiction:} The critical division-by-zero prevention subgoals (\texttt{state\_8} and \texttt{state\_11}) were solved by \texttt{action\_265} and \texttt{action\_301} using contradiction. For instance, \texttt{action\_301} introduced the hypothesis $h_{\mathrm{cos\_half}} : \cos(x/2) = 0$, derived $\cos x = -1$ and $\tan x = 0$ via double-angle identities, and showed that this contradicts the main hypothesis $1/\cos x + \tan x = 22/7$ (since $-1 + 0 = -1 \neq 22/7$).
\end{itemize}

\subsubsection{Final Stitching and Lean Verification}
Once all subgoals on the active path were solved, the And-Or graph propagated the solved status upward. The final reconstructed proof is a nested block of 95 lines of fully verified, formal Lean 4 code. The proof was verified by the Lean 4 compiler without a single placeholder or \texttt{sorry}, confirming the mathematical soundness of the entire trajectory.
```
