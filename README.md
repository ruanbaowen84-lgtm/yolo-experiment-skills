# YOLO Experiment Skills

从一份 YAML 配置自动完成 YOLO 检测对照实验，产出可恢复的训练记录和可对比的验证报告。框架通过 Ultralytics Python API 调用，也支持指定本地 YOLO-Master 源码树。

## 安装与快速验证

使用 Python 3.11 或更新版本。先在虚拟环境里安装依赖，有 GPU 的机器可先根据 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 选择合适的 PyTorch 构建。下面的命令在当前项目根目录执行，Windows PowerShell 和使用 Python 的 Linux/macOS 终端均可使用。

```text
python -m pip install -r docs/skills/auto-ai-improve-exp/requirements.txt
python docs/skills/auto-ai-improve-exp/scripts/make_smoke_demo.py --output experiments/smoke
python docs/skills/auto-ai-improve-exp/scripts/run_experiment.py --config experiments/smoke/experiment.yaml --dry-run
python docs/skills/auto-ai-improve-exp/scripts/run_experiment.py --config experiments/smoke/experiment.yaml
```

生成器创建 20 张人工图像、对应的 YOLO 边界框标签和两份小型模型配置。默认在 CPU 上各训练一轮，用于检查执行流程。真实实验请复制 [配置模板](docs/skills/auto-ai-improve-exp/assets/experiment.example.yaml)，换成自己的数据和模型。首次运行框架可能下载渲染所需的字体。示例数据和权重初始化无需下载。

## 配置和结果

- 支持 RGB YOLO 检测。目录型划分可填单个或多个图片目录，列表文件使用绝对图片路径，或使用相对列表所在目录的 `./` 路径。数据 YAML 的相对 `path` 以数据 YAML 所在目录为基准。标签使用常规的 `images/` 与 `labels/` 平行结构。
- 默认比较同一基线和配置中的前 `max_candidates` 个候选。训练参数、数据和种子一致。候选按 `val` 的 mAP50 与 mAP50–95 增益及融合后 GFLOPs 阈值筛选。开启 `final_test` 后，筛选结束才评估基线和入选模型的独立测试集。
- 输出目录含 `report.md`、`metrics.json`、`state.json`、解析后的配置、模型 YAML 快照、日志和权重。中断后运行同一命令恢复。配置、模型或数据改动后使用新的 `output`。预检用图片及标签的路径、大小和修改时间识别常见数据改动。
- 私有模型和权重应来自可信来源。使用自定义 YOLO-Master 模块时，在配置中填写 `framework_root` 并安装相应版本所需的依赖。公开 Ultralytics 只加载它支持的模型 YAML。

## Skill 入口与验证

把 [auto-ai-improve-exp](docs/skills/auto-ai-improve-exp) 安装到所用助手的技能目录，或让助手读取其中的 `SKILL.md`。CLI 也可以直接使用。

```text
python -m unittest discover -s docs/skills/auto-ai-improve-exp/tests -v
```

框架项目和授权信息见 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 和 [Tencent YOLO-Master](https://github.com/Tencent/YOLO-Master)。
