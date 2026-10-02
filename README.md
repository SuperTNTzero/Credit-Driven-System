# Credit-Driven Organization Dynamics

This repository contains the paper-facing implementation and evidence for a minimal organization loop in adaptive computational systems.

The central hypothesis is that effective behavior depends on two separable dynamical dimensions:

- `W`: the available capability support;
- `Phi`: a persistent, feedback-updated organization state that changes responsibility, routing, consolidation, and recovery.

The flagship system follows the loop

```text
W -> responsibility -> behavior -> credit -> Phi -> effective units -> W'
```

It is evaluated in a long-horizon formal-arithmetic environment, a complexity-scaling study, a probe-lifecycle study, and two cross-domain boundary experiments.

## Start Here

- [Main manuscript](期刊论文草稿_最小组织闭环与可验证算术发现.md)
- [Supplementary information](补充材料_组织动力学完整数学与预测.md)
- [EXP243-long report](EXP243-long_单体长时程组织闭环实验报告.md)
- [EXP244 report](EXP244_复杂度上升与非线性自动发现边界实验报告.md)
- [EXP245 report](EXP245_探针生命周期与组织过拟合实验报告.md)

## Main Evidence

| Experiment | Question | Entry point |
|---|---|---|
| EXP243-long | Can credit, persistent organization, consolidation, damage, and recovery operate in one uninterrupted state lineage? | `run_exp243_long.ps1` |
| EXP244 | When does recursive organization outperform static search or direct memory as compositional complexity rises? | `run_exp244.ps1` |
| EXP245 | How do birth, persistence, retirement, and reactivation affect organization cost and overfitting? | `run_exp245.ps1` |
| EXP120 | Where does direct content memory remain preferable in a small enumerable environment? | `exp120_minigrid_phi_persistent_organization.py` |
| EXP166 | Can responsibility repair transfer across tasks after a local capability is damaged? | `robot_phi_exp166/run_docker.ps1` |

Formal result JSON files and the four manuscript figures are retained under `results/`. Debug runs, downloaded datasets, checkpoints, caches, and the wider exploratory archive are intentionally excluded from the publication repository.

## Reproduce the Flagship Experiments

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\run_exp243_long.ps1
powershell -ExecutionPolicy Bypass -File .\run_exp244.ps1
powershell -ExecutionPolicy Bypass -File .\run_exp245.ps1
python .\plot_science_figures.py
```

EXP243-long is the primary confirmation run. It uses 20 paired seeds, 29 sequential tasks, seven methods, and one persistent state lineage per method and seed. The manuscript and supplementary information define the statistical unit, budgets, interventions, and interpretation limits.

EXP166 requires a local MuJoCo/ContinualWorld Docker image and a separately supplied MuJoCo license. The license is never tracked by Git.

## Repository Scope

This is a curated publication snapshot rather than the complete exploratory workspace. The inclusion list is maintained in `.gitignore`; add new files there deliberately when they become part of the auditable evidence chain.
