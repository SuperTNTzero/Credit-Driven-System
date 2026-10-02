# EXP166 Docker 使用说明

本实验复用 `credit-continualworld:latest` 镜像中的 MuJoCo 2.0、MetaWorld 和 ContinualWorld。宿主机只保存代码与结果。

## 一键运行

在 `信用神经网络` 目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\robot_phi_exp166\run_docker.ps1
```

默认运行 5 个种子。快速单种子检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\robot_phi_exp166\run_docker.ps1 -Seeds "0"
```

## 直接命令

```powershell
$root = (Resolve-Path .\robot_phi_exp166).Path
$key = (Resolve-Path .\continualworld_phi\mjkey.txt).Path
docker run --rm `
  -v "${root}:/workspace/project/robot_phi_exp166" `
  -v "${key}:/opt/.mujoco/mjkey.txt:ro" `
  credit-continualworld `
  python3 robot_phi_exp166/run_exp166.py --seeds 0,1,2,3,4
```

结果写入：

- `robot_phi_exp166/results/EXP166_robot_skill_reuse.json`
- `robot_phi_exp166/results/EXP166_robot_skill_reuse.png`

正式 5 种子结果与机制边界见 `../EXP166_机器人技能组合损坏与跨任务复用实验报告.md`。当前结果支持责任修复的跨任务关系复用，不支持 Phi 在当前任务修复速度上优于强 Task-KV。

## 环境检查

若镜像不存在，先在 `信用神经网络` 目录构建：

```powershell
$key = (Resolve-Path .\continualworld_phi\mjkey.txt).Path
docker build --secret "id=mjkey,src=$key" -f continualworld_phi/Dockerfile -t credit-continualworld .
```

实验需要本地 MuJoCo 2.0 license。license 只在构建和运行时挂载，不写入结果或派生镜像。

## 结果解释限制

该实验使用真实 MuJoCo 动力学，但运动角色、冗余原语和逐轴后果接口由实验者给定。结果只能验证给定能力接口后的组织、责任重分配和结构复用，不能解释为完整 VLA 或真实人形机器人已经自主发现技能。
