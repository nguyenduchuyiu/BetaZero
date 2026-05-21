# GammaZero Deep Experimental Metrics (Consolidated Report)

This report presents the consolidated statistics for both **miniF2F-valid** and **miniF2F-test** Splits, providing the five requested metrics to strengthen the experimental section of the manuscript.

---

## 1. Consolidated Deep Experimental Metrics Table

| Metric | miniF2F-valid (33 problems) | miniF2F-test (36 problems) |
| :--- | :---: | :---: |
| **Total Solved Theorems** | 27 / 33 (81.8%) | 28 / 36 (77.8%) |
| **1. Final Stitching Success Count** | 27 / 27 (**100.0%**) | 28 / 28 (**100.0%**) |
| **2. Number of Inserted Skeletons** | 568 (Avg: 17.21 / run) | 595 (Avg: 16.53 / run) |
| **3. Solved Style Breakdown** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;*Solved by Direct Local Proof only* | 13 (48.1%) | 21 (75.0%) |
| &nbsp;&nbsp;&nbsp;&nbsp;*Solved with Skeleton Decomposition* | 14 (51.9%) | 7 (25.0%) |
| **4. Subgoal Extraction Failure Rate** | 45 / 582 (**7.73%**) | 54 / 598 (**9.03%**) |
| **5. Subgoal Dependency Classes** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;*Solved-and-Used* | 142 | 89 |
| &nbsp;&nbsp;&nbsp;&nbsp;*Solved-but-Unused* | 9 | 5 |
| &nbsp;&nbsp;&nbsp;&nbsp;*Unresolved-Unused* | 145 | 118 |
| &nbsp;&nbsp;&nbsp;&nbsp;*Unresolved-Used* | 28 | 49 |

---

## 2. LaTeX Formatted Version for the Paper

```latex
\begin{table}[t]
\centering
\caption{Consolidated deep experimental metrics of GammaZero on \texttt{miniF2F-valid} and \texttt{miniF2F-test} splits.}
\label{tab:deep-metrics}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{\texttt{miniF2F-valid} (N=33)} & \textbf{\texttt{miniF2F-test} (N=36)} \\
\midrule
Solved / Total & 27 / 33 (81.8\%) & 28 / 36 (77.8\%) \\
\midrule
\textbf{1. Final Stitching Success} & 27 / 27 (100.0\%) & 28 / 28 (100.0\%) \\
\textbf{2. Inserted Skeletons (Total / Avg)} & 568 / 17.21 & 595 / 16.53 \\
\textbf{3. Proving Style Breakdown} & & \\
\quad Solved by Direct Local Proof & 13 (48.1\%) & 21 (75.0\%) \\
\quad Solved with Skeleton Decomposition & 14 (51.9\%) & 7 (25.0\%) \\
\textbf{4. Subgoal Extraction Failure Rate} & 45 / 582 (7.73\%) & 54 / 598 (9.03\%) \\
\textbf{5. Subgoal Dependency Classes} & & \\
\quad Solved-and-Used & 142 & 89 \\
\quad Solved-but-Unused & 9 & 5 \\
\quad Unresolved-Unused & 145 & 118 \\
\quad Unresolved-Used & 28 & 49 \\
\bottomrule
\end{tabular}
\end{table}
```
