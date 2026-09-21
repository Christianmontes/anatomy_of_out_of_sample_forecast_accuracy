| Model | h=1 IS-Shapley-VI | h=1 IS-GPBSV | h=3 IS-Shapley-VI | h=3 IS-GPBSV | h=6 IS-Shapley-VI | h=6 IS-GPBSV |
|---|---|---|---|---|---|---|
| PCA | 0.558*** | 0.681*** | 0.546*** | 0.654*** | 0.455* | 0.549** |
| ENet | 0.546** | 0.592** | 0.324 | 0.330 | 0.559** | 0.784*** |
| RF | 0.860*** | 0.868*** | 0.608*** | 0.624*** | 0.719*** | 0.748*** |
| XGBoost | 0.499** | 0.601*** | 0.266 | 0.398 | 0.290 | 0.463 |
| Neural net | 0.570*** | 0.758*** | 0.553*** | 0.737*** | 0.310 | 0.572*** |
| Linear comb. | 0.649*** | 0.754*** | 0.546** | 0.659*** | 0.553** | 0.686*** |
| Nonlinear comb. | 0.667*** | 0.785*** | 0.583*** | 0.675*** | 0.416 | 0.600*** |
| All-models comb. | 0.709*** | 0.799*** | 0.604*** | 0.689*** | 0.484** | 0.641*** |
| **Mean** | **0.632** | **0.730** | **0.504** | **0.596** | **0.473** | **0.630** |

*Model Agreement Score (MAS) of each in-sample importance ranking with the out-of-sample GPBSV ranking (higher = closer agreement; ***/**/* = p<=0.01/0.05/0.10). IS-Shapley-VI reproduces the paper's Table 1 MAS; IS-GPBSV is the symmetric in-sample-vs-out-of-sample GPBSV robustness check.*
