# 老年人视频识别评测系统

> 基于 Qwen2.5-Omni 的**老年人视频字幕转写评测框架**，支持 BLEU、CER 等多指标评估

## 🎯 这是什么？

这是一个用于**评估多模态大模型视频字幕转写能力**的系统：

**输入**：老年人视频 + 原始字幕（SRT）
**输出**：模型转写质量评分（BLEU/CER 指标）

### 典型应用场景

- 评估模型在不同视频条件下的表现（光线、背景噪音、口音等）
- 对比不同模型或不同参数的性能差异
- 找出模型幻觉（hallucination）问题
- 分析低分样本的根本原因

## ✨ 核心功能

- 📹 **视频切片**：将长视频按字幕时间戳切分为短片段
- 🤖 **批量推理**：支持多种多模态模型（Qwen2.5-Omni、Qwen2-VL 等）
- 📊 **多指标评测**：BLEU-1/2/3/4、CER 等
- 🔧 **灵活配置**：通过命令行参数或 config.py 配置
- 📁 **文件夹模式**：视频和字幕按文件名自动匹配

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/你的用户名/elderly_video_evaluation.git
cd elderly_video_evaluation

# 创建虚拟环境（推荐）
python -m venv .venv

# Windows 激活
.venv\Scripts\activate

# Linux/Mac 激活
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 📁 数据准备

### 1. 创建目录结构

```text
data/
├── videos/              # 视频文件（mp4, avi, mov 等）
│   ├── 001.mp4
│   └── 002.mp4
└── references/          # 参考文本（Ground Truth）
    ├── 001.srt          # SRT 字幕格式
    └── 002.txt          # 或纯文本格式
```

### 2. 文件匹配规则

视频和参考文本通过**文件名**自动匹配（扩展名不同没关系）：

| 视频文件 | 参考文本 | 匹配结果 |
|---------|---------|---------|
| `001.mp4` | `001.srt` | ✅ 匹配 |
| `001.mp4` | `001.txt` | ✅ 匹配 |
| `video_a.mp4` | `video_a.txt` | ✅ 匹配 |

### 3. 参考文本格式

**纯文本格式 (.txt)**：
```text
老年人说的话的完整转录内容
```

**SRT 字幕格式 (.srt)**：
```text
1
00:00:01,000 --> 00:00:05,000
老年人说的第一句话

2
00:00:06,000 --> 00:00:10,000
老年人说的第二句话
```

## 🚀 快速开始

### 第一步：视频切片

```bash
python clips_tools.py --clip_duration 30
```

这会将 `videos/` 中的长视频按 30 秒切分为短片段，输出到 `clips_30s/` 目录。

### 第二步：运行评测

```bash
python run_evaluation.py
```

批量处理所有视频片段，生成评测结果。

### 第三步：分析结果

```bash
python analyze_results.py --threshold 0.5
```

分离低分和高分样本，生成统计信息。

## 💡 完整工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入数据                                  │
│  ┌──────────────────┐    ┌──────────────────┐                │
│  │   视频文件        │    │   字幕文件        │                │
│  │   videos/        │    │   videos/        │                │
│  └──────────────────┘    └──────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  步骤1: clips_tools.py - 视频切片                               │
│  - 按字幕时间戳切割                                             │
│  - 视频 + 字幕对齐                                              │
│  - 输出: clips_30s/                                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  步骤2: run_evaluation.py - 批量推理 + 评测                    │
│  - 加载模型 (my_model.py)                                      │
│  - 批量处理视频片段                                             │
│  - 计算 BLEU/CER 指标                                           │
│  - 输出: results/results.csv                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  步骤3: analyze_results.py - 结果分析                          │
│  - 分离低分/高分样本                                            │
│  - 统计信息分析                                                 │
│  - 输出: results/analysis/                                      │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 进阶用法

### 自定义切片时长

```bash
# 60 秒切片
python clips_tools.py --clip_duration 60

# 45 秒切片
python clips_tools.py --clip_duration 45
```

### 分析不同指标

```bash
# 分析 BLEU-1（默认）
python analyze_results.py --threshold 0.5

# 分析 BLEU-2
python analyze_results.py --metric 'BLEU-2' --threshold 0.4

# 分析 CER（字符错误率，越低越好）
python analyze_results.py --metric 'CER' --threshold 0.15
```

### 重新评测（多指标组合）

```bash
# 仅 BLEU-1（快速）
python re_evaluate.py --profile bleu1_only

# 完整 BLEU 家族
python re_evaluate.py --profile bleu_family

# BLEU + CER（综合）
python re_evaluate.py --profile bleu_cer

# 所有指标（详细）
python re_evaluate.py --profile comprehensive
```

## 🔌 接入自定义模型

### 步骤 1：复制模板

```bash
cp model_adapter/custom_model_example.py model_adapter/my_model.py
```

### 步骤 2：修改模型代码

打开 `my_model.py`，修改推理逻辑：

```python
class OmniModelAdapter(BaseVideoModelAdapter):
    
    def _load_model(self):
        """加载你的多模态模型"""
        from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self.config["model_path"])
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.config["model_path"],
            device_map="auto"
        )
    
    def infer(self, video_path: str) -> InferenceResult:
        """推理单个视频"""
        # 构建消息...
        # 推理...
        predicted_text = "模型预测的文本"
        
        return InferenceResult(
            video_path=video_path,
            predicted_text=predicted_text
        )
```

### 步骤 3：注册模型

在 `run_evaluation.py` 的 `create_model()` 函数中添加分支。

## 📊 评测指标

| 指标 | 说明 | 范围 | 推荐场景 |
|------|------|-------|---------|
| **BLEU-1** | 一元语法相似度 | 0-1（越高越好） | 通用 |
| **BLEU-2** | 二元语法相似度 | 0-1（越高越好） | 短语准确度 |
| **BLEU-4** | 四元语法相似度 | 0-1（越高越好） | 长句流畅度 |
| **CER** | 字符错误率 | 0-1（越低越好） | 中文语音识别 |

## 📂 项目结构

```
elderly_video_evaluation/
├── config.py                 # 【配置】所有参数集中管理
├── clips_tools.py            # 【步骤1】视频切片工具
├── run_evaluation.py         # 【步骤2】主评测脚本
├── analyze_results.py        # 【步骤3】结果分析工具 ⭐
├── re_evaluate.py           # 【可选】重新评测
├── simple_evaluator.py     # 简化版评测器
├── my_model.py              # 【修改】模型适配文件
│
├── model_adapter/           # 模型适配器
│   ├── base_adapter.py     # 基类
│   └── custom_model_example.py  # 模板
│
├── data_loader/             # 数据加载
├── metrics/                 # 评测指标
│
└── data/                    # 示例数据
```

## ⚙️ 配置说明

### config.py

主要配置项：

```python
# 文件路径
RESULTS_CSV = "results/results.csv"
ANALYSIS_DIR = "results/analysis"

# 视频切片
CLIP_DURATION = 30  # 切片时长（秒）

# 评测指标
BLEU1_LOW_SCORE_THRESHOLD = 0.5
```

### 参数优先级

```
命令行参数 > config.py > 代码默认值
```

例如：`python analyze_results.py --threshold 0.3` 会覆盖 config.py 中的 0.5。

## 📚 其他文档

- [快速参考](QUICK_REFERENCE.txt) - 命令行参数速查
- [详细指南](USAGE_GUIDE.txt) - 完整使用说明
- [功能演示](FEATURE_DEMO.txt) - 进阶功能示例

## 🚀 快速测试

```bash
# 使用虚拟模型测试流程
python run_evaluation.py --model_type dummy --video_dir data/videos --ref_dir data/references
```

## 📄 许可证

MIT License

---

**祝你评测顺利！** 🎉
