# GammaZero Budget Usage Plot Report (miniF2F-test)

This artifact presents a high-resolution visualization of the search budget consumed by GammaZero across all **244 problems** of the completed `miniF2F-test` split.

## 1. Budget Usage Distribution Plot

The plot below displays the number of expanded action nodes (AND nodes) for all 244 theorems, sorted in ascending order of resource usage. Solved problems are colored in emerald green, while failed problems are colored in crimson rose.

![GammaZero Budget Usage Distribution](/home/npthai/.gemini/antigravity/brain/5a1fc07e-dc22-4f49-bac2-0c1f69b4f581/budget_usage_distribution.png)

---

## 2. In-Depth Graph Insights

1. **Bimodal Resource Profile (Early-Stopping vs. Exhaustion):**
   * **Emerald Green (Solved):** The vast majority of solved problems are clustered on the left side of the chart, utilizing very few nodes (average of **36.2 nodes**). This shows that if a path to a correct proof is discoverable, the BFS heuristic identifies it extremely rapidly, leading to highly efficient early-termination.
   * **Crimson Rose (Failed):** The unsolved problems on the far right consume the full search budget (maxing out at **512 nodes**). This is the expected behavior for unsuccessful search-guided theorem proving splits, where the BFS algorithm exhausts the node allocation limits.

2. **Practical Computational Footprint:**
   * Because of the extremely low node footprint of solved cases, the overall search is incredibly cost-efficient, averaging only **102.8 total expanded nodes per theorem across all 244 problems**.
