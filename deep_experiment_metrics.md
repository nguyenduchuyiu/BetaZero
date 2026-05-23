# GammaZero Deep Experimental Metrics (Consolidated Report)

This report presents the consolidated statistics for both **miniF2F-valid** and **miniF2F-test** Splits, providing the five requested metrics to strengthen the experimental section of the manuscript.

---

## 1. Consolidated Deep Experimental Metrics Table

| Metric | miniF2F-valid (33 problems) | miniF2F-test (244 problems) |
| :--- | :---: | :---: |
| **Total Solved Theorems** | 27 / 33 (81.8%) | 188 / 244 (77.0%) |
| **1. Final Stitching Success Count** | 27 / 27 (**100.0%**) | 188 / 188 (**100.0%**) |
| **2. Number of Inserted Skeletons** | 568 (Avg: 17.21 / run) | 3362 (Avg: 13.78 / run) |
| **3. Solved Style Breakdown** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;*Solved by Direct Local Proof only* | 13 (48.1%) | 146 (77.7%) |
| &nbsp;&nbsp;&nbsp;&nbsp;*Solved with Skeleton Decomposition* | 14 (51.9%) | 42 (22.3%) |
| **4. Subgoal Extraction Failure Rate** | 45 / 582 (**7.73%**) | 233 / 3384 (**6.89%**) |
| **5. Subgoal Dependency Classes** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;*Solved-and-Used* | 142 | 437 |
| &nbsp;&nbsp;&nbsp;&nbsp;*Solved-but-Unused* | 9 | 77 |
| &nbsp;&nbsp;&nbsp;&nbsp;*Unresolved-Unused* | 145 | 894 |
| &nbsp;&nbsp;&nbsp;&nbsp;*Unresolved-Used* | 28 | 343 |

---

## 2. LaTeX Formatted Version for the Paper

```latex
\begin{table}[t]
\centering
\caption{Consolidated deep experimental metrics of GammaZero on \texttt{miniF2F-valid} and \texttt{miniF2F-test} splits.}
\label{tab:deep-metrics}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{\texttt{miniF2F-valid} (N=33)} & \textbf{\texttt{miniF2F-test} (N=244)} \\
\midrule
Solved / Total & 27 / 33 (81.8\%) & 188 / 244 (77.0\%) \\
\midrule
\textbf{1. Final Stitching Success} & 27 / 27 (100.0\%) & 188 / 188 (100.0\%) \\
\textbf{2. Inserted Skeletons (Total / Avg)} & 568 / 17.21 & 3362 / 13.78 \\
\textbf{3. Proving Style Breakdown} & & \\
\quad Solved by Direct Local Proof & 13 (48.1\%) & 146 (77.7\%) \\
\quad Solved with Skeleton Decomposition & 14 (51.9\%) & 42 (22.3\%) \\
\textbf{4. Subgoal Extraction Failure Rate} & 45 / 582 (7.73\%) & 233 / 3384 (6.89\%) \\
\textbf{5. Subgoal Dependency Classes} & & \\
\quad Solved-and-Used & 142 & 437 \\
\quad Solved-but-Unused & 9 & 77 \\
\quad Unresolved-Unused & 145 & 894 \\
\quad Unresolved-Used & 28 & 343 \\
\bottomrule
\end{tabular}
\end{table}
```
