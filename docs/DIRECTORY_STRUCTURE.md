# SonarGNN 项目目录结构说明

为了确保本 CDS 525 深度学习课程期末项目的代码可读性、可维护性以及团队协作的顺畅，我们设计了以下遵循机器学习/深度学习行业最佳实践的目录结构。

## 📁 整体结构预览

```text
SonarGNN/
├── configs/                # 配置文件目录
│   ├── baseline.yaml       # 基线模型配置 (如学习率、Batch Size)
│   └── experiment_loss.yaml# 损失函数消融实验配置
├── data/                   # 数据目录 (注意：大文件不应提交到 Git)
│   ├── raw/                # 原始数据集 (直接从 Kaggle 下载的压缩包或 CSV)
│   └── processed/          # 预处理后的数据 (如 PyG 生成的 .pt 缓存文件)
├── docs/                   # 项目文档与管理规划
│   ├── CDS525 - Group Project Instruction_2026.docx # 课程作业指导书
│   ├── DIRECTORY_STRUCTURE.md  # 也就是本文档
│   └── Task_Allocation_Plan.md # 团队分工与协作流程
├── notebooks/              # Jupyter Notebook 探索性分析
│   └── EDA.ipynb           # 数据的探索性数据分析 (建议组员 A 使用)
├── scripts/                # 自动化运维脚本
│   ├── run_experiments.sh  # 批量运行所有消融实验的 Bash 脚本
│   └── train_pipeline.sh   # 完整的从头训练管道脚本
├── src/                    # 核心业务逻辑代码 (Python 包)
│   ├── __init__.py
│   ├── data/               # 数据加载与图构建模块
│   │   ├── __init__.py
│   │   └── dataset.py      # EllipticDataset 类的定义
│   ├── evaluation/         # 评估与指标计算模块
│   │   ├── __init__.py
│   │   └── metrics.py      # F1-Score, PR-AUC 等不平衡指标计算
│   ├── models/             # 深度学习模型定义
│   │   ├── __init__.py
│   │   └── gcn.py          # GCN 架构的 PyTorch 实现
│   ├── training/           # 训练循环与优化模块
│   │   ├── __init__.py
│   │   ├── loss.py         # 自定义损失函数 (如 Weighted CE)
│   │   └── train.py        # 标准的 Training Loop
│   └── utils/              # 通用工具函数
│       ├── __init__.py
│       ├── logger.py       # 统一的控制台与文件日志输出
│       └── plot.py         # 统一的图表绘制脚本 (供组员统一调用)
├── tests/                  # 单元测试代码 (确保模型前向传播不崩溃)
├── .gitignore              # Git 忽略文件配置
├── pyproject.toml          # 现代 Python 项目管理文件 (替代 setup.py)
├── README.md               # 项目主页说明
└── requirements.txt        # Python 依赖清单
```

## 🛠️ 关键模块详细说明

### 1. `src/` (源代码主目录)
这是整个项目的核心大脑，按照功能进行了严格的解耦：
*   **`data/`**: 专门负责“喂数据”。把原始的 CSV 转变成 PyTorch Geometric 认识的拓扑图格式。
*   **`models/`**: 专门负责“长什么样”。这里只定义神经网络的架构，例如包含几个 GraphConv 层、是否有 Dropout，**绝对不包含**训练逻辑。
*   **`training/`**: 专门负责“怎么学”。它从 `models` 拿架构，从 `data` 拿数据，通过 `loss.py` 计算梯度并反向传播。
*   **`evaluation/`**: 专门负责“考几分”。独立于训练模块，用来处理验证集/测试集，输出符合 CDS 525 课程要求的硬性指标。
*   **`utils/plot.py`**: 专门负责“画好图”。为了让最终的 Word 报告和 PPT 看起来专业且风格统一，所有成员必须调用此文件里的画图函数，严禁自己随便 `plt.plot()`。

### 2. `configs/` (配置驱动开发)
为了避免每次改学习率都要去代码里找硬编码（Hardcode）的变量，我们将所有可变的超参数抽离到 YAML 文件中。
例如，你想测试不同的学习率，只需要在命令行运行：
`python src/training/train.py --config configs/baseline.yaml`

### 3. `data/` 与 `.gitignore`
*   **注意**: 深度学习的数据集通常极大，**绝对不要**把 `data/raw` 和 `data/processed` 里的内容提交到 GitHub。
*   我们已经在 `.gitignore` 中配置了对 `data/raw/*` 和 `data/processed/*` 的忽略，但保留了 `.gitkeep` 使得空文件夹结构能被 Git 追踪。

### 4. `pyproject.toml` vs `requirements.txt`
*   `requirements.txt`: 提供给组员一键快速安装环境 (`pip install -r requirements.txt`)。
*   `pyproject.toml`: 遵循 PEP 621 的现代 Python 打包标准，定义了项目的元数据、依赖项以及格式化工具（如 ruff）的配置，是未来工业界的标准。
