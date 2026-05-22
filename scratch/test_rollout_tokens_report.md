# Gemini 3 Flash miniF2F-test Rollout Performance & Node Stats

**Total Problems:** 87
**Total Solved:** 59 / 87 (**67.82%**)
- **Direct Solved at Root (Depth 0):** 41 / 87 (**47.13%**)
- **Hierarchical Solved (Skeleton Depth >= 1):** 18 / 87 (**20.69%**)

## 1. Summary Averages

| Metrics | Solved Problems | Failed Problems | All Combined |
| :--- | :---: | :---: | :---: |
| **Average Total Nodes** | 63.6 | 406.1 | 173.8 |
| **Average Tactic Nodes** | 52.1 | 312.0 | 135.7 |
| **Average Skeleton Nodes** | 5.2 | 67.9 | 25.4 |
| **Average Est. Output Tokens** | 24,626 | 200,324 | 81,172 |
| **Total Output Tokens Generated** | - | - | **7,061,990** |

## 2. Direct Root Solves vs Hierarchical Solves Breakdown

### Direct Solves at Root (Depth 0) — 41 Problems
These problems were solved by directly applying tactic steps on the root goal, without requiring skeleton decomposition:

1. `aimeII_2001_p3` (11 nodes, 7,204 output tokens)
2. `aimeII_2020_p6` (17 nodes, 13,886 output tokens)
3. `aimeI_2000_p7` (7 nodes, 2,571 output tokens)
4. `aime_1991_p1` (21 nodes, 15,595 output tokens)
5. `algebra_2complexrootspoly_xsqp49eqxp7itxpn7i` (7 nodes, 264 output tokens)
6. `algebra_2rootsintpoly_am10tap11eqasqpam110` (5 nodes, 7 output tokens)
7. `algebra_2rootspoly_apatapbeq2asqp2ab` (5 nodes, 7 output tokens)
8. `algebra_2varlineareq_xpeeq7_2xpeeq3_eeq11_xeqn4` (5 nodes, 217 output tokens)
9. `algebra_3rootspoly_amdtamctambeqnasqmbpctapcbtdpasqmbpctapcbta` (5 nodes, 7 output tokens)
10. `algebra_amgm_sumasqdivbsqgeqsumbdiva` (23 nodes, 8,938 output tokens)
11. `algebra_apb4leq8ta4pb4` (13 nodes, 3,733 output tokens)
12. `algebra_binomnegdiscrineq_10alt28asqp1` (7 nodes, 284 output tokens)
13. `algebra_manipexpr_2erprsqpesqeqnrpnesq` (5 nodes, 7 output tokens)
14. `algebra_manipexpr_apbeq2cceqiacpbceqm2` (5 nodes, 234 output tokens)
15. `algebra_sqineq_2at2pclta2c2p41pc` (5 nodes, 95 output tokens)
16. `algebra_sqineq_2unitcircatblt1` (5 nodes, 240 output tokens)
17. `algebra_sqineq_36azm9asqle36zsq` (5 nodes, 237 output tokens)
18. `algebra_sqineq_4bap1lt4bsqpap1sq` (5 nodes, 194 output tokens)
19. `algebra_xmysqpymzsqpzmxsqeqxyz_xpypzp6dvdx3y3z3` (7 nodes, 2,163 output tokens)
20. `amc12_2000_p11` (7 nodes, 541 output tokens)
21. `amc12_2000_p5` (5 nodes, 119 output tokens)
22. `amc12_2001_p9` (5 nodes, 568 output tokens)
23. `amc12a_2002_p1` (19 nodes, 5,465 output tokens)
24. `amc12a_2002_p12` (23 nodes, 16,483 output tokens)
25. `amc12a_2003_p1` (9 nodes, 877 output tokens)
26. `amc12a_2008_p2` (13 nodes, 884 output tokens)
27. `amc12a_2009_p2` (5 nodes, 11 output tokens)
28. `amc12a_2009_p5` (5 nodes, 266 output tokens)
29. `amc12a_2009_p9` (5 nodes, 206 output tokens)
30. `amc12a_2010_p10` (15 nodes, 8,455 output tokens)
31. `amc12a_2013_p7` (5 nodes, 636 output tokens)
32. `amc12a_2013_p8` (13 nodes, 2,121 output tokens)
33. `amc12a_2016_p2` (23 nodes, 8,559 output tokens)
34. `amc12a_2016_p3` (5 nodes, 36 output tokens)
35. `amc12a_2017_p2` (7 nodes, 339 output tokens)
36. `amc12a_2021_p7` (5 nodes, 292 output tokens)
37. `amc12b_2003_p6` (7 nodes, 2,883 output tokens)
38. `amc12b_2003_p9` (5 nodes, 336 output tokens)
39. `amc12b_2020_p5` (9 nodes, 2,815 output tokens)
40. `imo_1966_p5` (9 nodes, 9,631 output tokens)
41. `imo_1974_p5` (7 nodes, 3,817 output tokens)

### Hierarchical Solves (Depth >= 1) — 18 Problems
These problems were solved using GammaZero's signature nested skeleton proof decomposition search:

1. `aime_1983_p9` (65 nodes, 4 skeletons, 22,393 output tokens)
2. `aime_1988_p3` (272 nodes, 18 skeletons, 86,489 output tokens)
3. `aime_1990_p2` (158 nodes, 8 skeletons, 35,680 output tokens)
4. `aime_1996_p5` (427 nodes, 50 skeletons, 285,986 output tokens)
5. `algebra_amgm_faxinrrp2msqrt2geq2mxm1div2x` (62 nodes, 4 skeletons, 12,366 output tokens)
6. `algebra_amgm_prod1toneq1_sum1tongeqn` (108 nodes, 8 skeletons, 31,797 output tokens)
7. `amc12_2000_p15` (58 nodes, 4 skeletons, 16,232 output tokens)
8. `amc12a_2003_p24` (236 nodes, 20 skeletons, 72,798 output tokens)
9. `amc12a_2008_p15` (316 nodes, 37 skeletons, 81,946 output tokens)
10. `amc12a_2008_p8` (64 nodes, 4 skeletons, 11,011 output tokens)
11. `amc12a_2010_p11` (85 nodes, 8 skeletons, 28,078 output tokens)
12. `amc12a_2011_p18` (126 nodes, 8 skeletons, 47,663 output tokens)
13. `amc12b_2004_p3` (55 nodes, 4 skeletons, 15,947 output tokens)
14. `imo_1961_p1` (76 nodes, 4 skeletons, 24,878 output tokens)
15. `imo_1964_p1_1` (194 nodes, 14 skeletons, 50,416 output tokens)
16. `imo_1964_p1_2` (70 nodes, 6 skeletons, 17,511 output tokens)
17. `imo_1965_p1` (470 nodes, 60 skeletons, 272,345 output tokens)
18. `imo_1966_p4` (541 nodes, 46 skeletons, 218,159 output tokens)

## 3. Detailed Problem Statistics

| No. | Problem Name | Status | Total Nodes | Tactic Nodes | Skeleton Nodes | Est. Output Tokens |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `aimeII_2001_p3` | **SOLVED (Direct)** | 11 | 10 | 0 | 7,204 |
| 2 | `aimeII_2020_p6` | **SOLVED (Direct)** | 17 | 16 | 0 | 13,886 |
| 3 | `aimeI_2000_p7` | **SOLVED (Direct)** | 7 | 6 | 0 | 2,571 |
| 4 | `aime_1983_p9` | **SOLVED (Hierarchical)** | 65 | 50 | 4 | 22,393 |
| 5 | `aime_1984_p15` | FAILED | 468 | 318 | 120 | 341,890 |
| 6 | `aime_1984_p5` | FAILED | 550 | 432 | 78 | 213,859 |
| 7 | `aime_1987_p8` | FAILED | 536 | 462 | 45 | 351,783 |
| 8 | `aime_1988_p3` | **SOLVED (Hierarchical)** | 272 | 222 | 18 | 86,489 |
| 9 | `aime_1988_p4` | FAILED | 541 | 453 | 56 | 155,503 |
| 10 | `aime_1990_p2` | **SOLVED (Hierarchical)** | 158 | 136 | 8 | 35,680 |
| 11 | `aime_1991_p1` | **SOLVED (Direct)** | 21 | 20 | 0 | 15,595 |
| 12 | `aime_1991_p6` | FAILED | 530 | 402 | 104 | 242,368 |
| 13 | `aime_1994_p4` | FAILED | 65 | 32 | 32 | 36,597 |
| 14 | `aime_1996_p5` | **SOLVED (Hierarchical)** | 427 | 338 | 50 | 285,986 |
| 15 | `aime_1997_p11` | FAILED | 65 | 32 | 32 | 26,028 |
| 16 | `algebra_2complexrootspoly_xsqp49eqxp7itxpn7i` | **SOLVED (Direct)** | 7 | 6 | 0 | 264 |
| 17 | `algebra_2rootsintpoly_am10tap11eqasqpam110` | **SOLVED (Direct)** | 5 | 4 | 0 | 7 |
| 18 | `algebra_2rootspoly_apatapbeq2asqp2ab` | **SOLVED (Direct)** | 5 | 4 | 0 | 7 |
| 19 | `algebra_2varlineareq_xpeeq7_2xpeeq3_eeq11_xeqn4` | **SOLVED (Direct)** | 5 | 4 | 0 | 217 |
| 20 | `algebra_3rootspoly_amdtamctambeqnasqmbpctapcbtdpasqmbpctapcbta` | **SOLVED (Direct)** | 5 | 4 | 0 | 7 |
| 21 | `algebra_amgm_faxinrrp2msqrt2geq2mxm1div2x` | **SOLVED (Hierarchical)** | 62 | 46 | 4 | 12,366 |
| 22 | `algebra_amgm_prod1toneq1_sum1tongeqn` | **SOLVED (Hierarchical)** | 108 | 96 | 8 | 31,797 |
| 23 | `algebra_amgm_sqrtxymulxmyeqxpy_xpygeq4` | FAILED | 65 | 32 | 32 | 35,404 |
| 24 | `algebra_amgm_sumasqdivbsqgeqsumbdiva` | **SOLVED (Direct)** | 23 | 22 | 0 | 8,938 |
| 25 | `algebra_apb4leq8ta4pb4` | **SOLVED (Direct)** | 13 | 12 | 0 | 3,733 |
| 26 | `algebra_binomnegdiscrineq_10alt28asqp1` | **SOLVED (Direct)** | 7 | 6 | 0 | 284 |
| 27 | `algebra_manipexpr_2erprsqpesqeqnrpnesq` | **SOLVED (Direct)** | 5 | 4 | 0 | 7 |
| 28 | `algebra_manipexpr_apbeq2cceqiacpbceqm2` | **SOLVED (Direct)** | 5 | 4 | 0 | 234 |
| 29 | `algebra_sqineq_2at2pclta2c2p41pc` | **SOLVED (Direct)** | 5 | 4 | 0 | 95 |
| 30 | `algebra_sqineq_2unitcircatblt1` | **SOLVED (Direct)** | 5 | 4 | 0 | 240 |
| 31 | `algebra_sqineq_36azm9asqle36zsq` | **SOLVED (Direct)** | 5 | 4 | 0 | 237 |
| 32 | `algebra_sqineq_4bap1lt4bsqpap1sq` | **SOLVED (Direct)** | 5 | 4 | 0 | 194 |
| 33 | `algebra_xmysqpymzsqpzmxsqeqxyz_xpypzp6dvdx3y3z3` | **SOLVED (Direct)** | 7 | 6 | 0 | 2,163 |
| 34 | `amc12_2000_p11` | **SOLVED (Direct)** | 7 | 6 | 0 | 541 |
| 35 | `amc12_2000_p15` | **SOLVED (Hierarchical)** | 58 | 48 | 4 | 16,232 |
| 36 | `amc12_2000_p5` | **SOLVED (Direct)** | 5 | 4 | 0 | 119 |
| 37 | `amc12_2001_p2` | FAILED | 563 | 464 | 48 | 241,178 |
| 38 | `amc12_2001_p9` | **SOLVED (Direct)** | 5 | 4 | 0 | 568 |
| 39 | `amc12a_2002_p1` | **SOLVED (Direct)** | 19 | 18 | 0 | 5,465 |
| 40 | `amc12a_2002_p12` | **SOLVED (Direct)** | 23 | 22 | 0 | 16,483 |
| 41 | `amc12a_2002_p21` | FAILED | 562 | 441 | 68 | 284,490 |
| 42 | `amc12a_2003_p1` | **SOLVED (Direct)** | 9 | 8 | 0 | 877 |
| 43 | `amc12a_2003_p24` | **SOLVED (Hierarchical)** | 236 | 187 | 20 | 72,798 |
| 44 | `amc12a_2003_p25` | FAILED | 534 | 382 | 130 | 324,529 |
| 45 | `amc12a_2008_p15` | **SOLVED (Hierarchical)** | 316 | 260 | 37 | 81,946 |
| 46 | `amc12a_2008_p2` | **SOLVED (Direct)** | 13 | 12 | 0 | 884 |
| 47 | `amc12a_2008_p4` | FAILED | 169 | 122 | 40 | 41,257 |
| 48 | `amc12a_2008_p8` | **SOLVED (Hierarchical)** | 64 | 48 | 4 | 11,011 |
| 49 | `amc12a_2009_p15` | FAILED | 529 | 406 | 100 | 388,278 |
| 50 | `amc12a_2009_p2` | **SOLVED (Direct)** | 5 | 4 | 0 | 11 |
| 51 | `amc12a_2009_p25` | FAILED | 544 | 376 | 133 | 290,699 |
| 52 | `amc12a_2009_p5` | **SOLVED (Direct)** | 5 | 4 | 0 | 266 |
| 53 | `amc12a_2009_p9` | **SOLVED (Direct)** | 5 | 4 | 0 | 206 |
| 54 | `amc12a_2010_p10` | **SOLVED (Direct)** | 15 | 14 | 0 | 8,455 |
| 55 | `amc12a_2010_p11` | **SOLVED (Hierarchical)** | 85 | 70 | 8 | 28,078 |
| 56 | `amc12a_2010_p22` | FAILED | 538 | 420 | 92 | 221,651 |
| 57 | `amc12a_2011_p18` | **SOLVED (Hierarchical)** | 126 | 100 | 8 | 47,663 |
| 58 | `amc12a_2013_p7` | **SOLVED (Direct)** | 5 | 4 | 0 | 636 |
| 59 | `amc12a_2013_p8` | **SOLVED (Direct)** | 13 | 12 | 0 | 2,121 |
| 60 | `amc12a_2015_p10` | FAILED | 65 | 32 | 32 | 23,107 |
| 61 | `amc12a_2016_p2` | **SOLVED (Direct)** | 23 | 22 | 0 | 8,559 |
| 62 | `amc12a_2016_p3` | **SOLVED (Direct)** | 5 | 4 | 0 | 36 |
| 63 | `amc12a_2017_p2` | **SOLVED (Direct)** | 7 | 6 | 0 | 339 |
| 64 | `amc12a_2017_p7` | FAILED | 258 | 172 | 78 | 104,994 |
| 65 | `amc12a_2019_p21` | FAILED | 554 | 464 | 48 | 280,269 |
| 66 | `amc12a_2019_p9` | FAILED | 323 | 204 | 100 | 175,270 |
| 67 | `amc12a_2020_p13` | FAILED | 197 | 140 | 48 | 106,462 |
| 68 | `amc12a_2020_p21` | FAILED | 554 | 452 | 60 | 440,409 |
| 69 | `amc12a_2021_p7` | **SOLVED (Direct)** | 5 | 4 | 0 | 292 |
| 70 | `amc12b_2002_p11` | FAILED | 551 | 454 | 58 | 229,618 |
| 71 | `amc12b_2002_p3` | FAILED | 568 | 426 | 82 | 113,876 |
| 72 | `amc12b_2002_p6` | FAILED | 562 | 448 | 61 | 203,898 |
| 73 | `amc12b_2003_p17` | FAILED | 563 | 461 | 50 | 168,147 |
| 74 | `amc12b_2003_p6` | **SOLVED (Direct)** | 7 | 6 | 0 | 2,883 |
| 75 | `amc12b_2003_p9` | **SOLVED (Direct)** | 5 | 4 | 0 | 336 |
| 76 | `amc12b_2004_p3` | **SOLVED (Hierarchical)** | 55 | 48 | 4 | 15,947 |
| 77 | `amc12b_2020_p5` | **SOLVED (Direct)** | 9 | 8 | 0 | 2,815 |
| 78 | `amc12b_2021_p21` | FAILED | 533 | 456 | 56 | 360,128 |
| 79 | `imo_1961_p1` | **SOLVED (Hierarchical)** | 76 | 56 | 4 | 24,878 |
| 80 | `imo_1962_p4` | FAILED | 65 | 32 | 32 | 27,696 |
| 81 | `imo_1964_p1_1` | **SOLVED (Hierarchical)** | 194 | 164 | 14 | 50,416 |
| 82 | `imo_1964_p1_2` | **SOLVED (Hierarchical)** | 70 | 60 | 6 | 17,511 |
| 83 | `imo_1965_p1` | **SOLVED (Hierarchical)** | 470 | 363 | 60 | 272,345 |
| 84 | `imo_1966_p4` | **SOLVED (Hierarchical)** | 541 | 451 | 46 | 218,159 |
| 85 | `imo_1966_p5` | **SOLVED (Direct)** | 9 | 8 | 0 | 9,631 |
| 86 | `imo_1973_p3` | FAILED | 318 | 222 | 86 | 179,684 |
| 87 | `imo_1974_p5` | **SOLVED (Direct)** | 7 | 6 | 0 | 3,817 |
