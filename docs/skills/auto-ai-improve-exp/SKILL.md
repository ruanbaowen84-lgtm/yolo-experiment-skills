---
name: auto-ai-improve-exp
description: Run reproducible Ultralytics YOLO and YOLO-Master detection experiments from a baseline, candidate model YAMLs, and a local dataset. Use for baseline training, ablation and candidate selection, checkpoint resume, and reports. 适用于 YOLO 检测自动实验、对照消融和训练续跑。
---

# YOLO 检测自动实验

## 准备实验

- 使用已准备好的 YOLO 检测数据 YAML、一份基线模型 YAML 和至少一份候选模型 YAML。读取项目里已有的路径、训练设置和预算。用户要求从本地配置库挑候选时，先核对候选与基线的框架版本、任务头、自定义模块和模型尺寸，再把选中的候选及改动写进实验配置。
- 从 [实验配置模板](assets/experiment.example.yaml) 复制一份配置。路径相对于配置文件解析。使用安装的 Ultralytics 时省略 `framework_root`。模型需要 YOLO-Master 自定义模块时，提供包含 `ultralytics/` 的本地源码根目录，并使用该源码所需的依赖。保留决定模型尺寸的原始 YAML 文件名。
- 第一版支持 RGB detect、单个训练设备和显式候选列表。所有模型使用同一数据划分、训练参数、种子及可选初始化权重。需要确认的信息限于无法从项目和请求确定的必填项，或会改变实验设计的选择。
- 显式设置 `max_candidates` 和 `epochs` 来限制运行预算。正式实验按研究目标设置轮数和筛选阈值。短训示例用来检查流程。

## 启动与恢复

用当前环境的 Python 运行本技能目录下的脚本。以下命令在技能目录执行。先预检，再用同一份配置启动实验。

```text
python scripts/run_experiment.py --config path/to/experiment.yaml --dry-run
python scripts/run_experiment.py --config path/to/experiment.yaml
```

预检打印解析后的路径和配置，不启动训练。训练依次执行基线和预算内的候选，独立验证每个 `best.pt`。候选按相对基线的 mAP50 与 mAP50–95 之和增益和融合后 GFLOPs 增幅筛选。在符合阈值的候选中选验证分数最高的一组，同分优先低 GFLOPs。候选都未达标时保留基线。需要逐阶段消融时，用各阶段独立的输出目录，显式指定该阶段的基线和组合候选。

候选筛选使用 `val`。配置了 `final_test: true` 并且数据有独立 `test` 划分时，入选结果确定后评估基线和入选模型。检查 `state.json`、阶段日志和结果文件来判断实际进度。中断后重新执行原命令，已完成阶段会复用，训练从有效的 `last.pt` 恢复。配置、模型 YAML 或初始化权重有变动时使用新的输出目录。失败根据阶段日志定位，修复环境后继续同一次实验。

## 交付结果

提供输出目录的 `report.md`、`metrics.json`、模型权重以及关键日志。报告写明基线、候选、验证集增益、参数量、融合后 GFLOPs、筛选结果和适用的独立测试结果。缺少输出或训练失败时引用实际状态和日志。
