# 老年人视频识别评测系统

> 一个简单、解耦的多模态模型 (Omni) 视频识别评测框架，支持 BLEU、CER 等指标

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ 特性

- 🔌 **模型可插拔**：只需实现推理接口，即可接入 Qwen2.5-Omni、Qwen2-VL、InternVL 等多模态模型
- 📁 **文件夹模式**：视频和参考文本按文件名自动匹配
- 📊 **多种输出**：自动生成 JSON、CSV 和文本报告

---

## 📦 安装

```bash
git clone https://github.com/yourusername/elderly_video_evaluation.git
cd elderly_video_evaluation
pip install -r requirements.txt
```

---

## 📁 数据准备

### 文件夹结构

```
data/
├── videos/              # 视频文件
│   ├── 001.mp4
│   ├── 002.mp4
│   └── 003.mp4
└── references/          # 参考文本（Ground Truth）
    ├── 001.srt          # SRT 字幕格式
    ├── 002.txt          # 或纯文本格式
    └── 003.txt
```

### 匹配规则

**视频和参考文本通过文件名匹配**（扩展名不同没关系）：

| 视频文件 | 参考文本 | 匹配结果 |
|---------|---------|---------|
| `001.mp4` | `001.srt` | ✅ 匹配 |
| `001.mp4` | `001.txt` | ✅ 匹配 |
| `video_a.mp4` | `video_a.txt` | ✅ 匹配 |
| `001.mp4` | `002.txt` | ❌ 不匹配 |

### 参考文本格式

**格式 1：纯文本 (.txt)**
```
老年人说的话的完整转录内容
```

**格式 2：SRT 字幕 (.srt)**
```
1
00:00:01,000 --> 00:00:05,000
老年人说的第一句话

2
00:00:06,000 --> 00:00:10,000
老年人说的第二句话
```
> 系统会自动提取 SRT 中的所有文本并合并

---

## 🔧 接入你的模型（3步）

### Step 1: 复制模板

```bash
cp model_adapter/custom_model_example.py model_adapter/my_model.py
```

### Step 2: 实现推理逻辑

打开 `model_adapter/my_model.py`，修改两个方法：

```python
from model_adapter.base_adapter import BaseVideoModelAdapter, InferenceResult

class OmniModelAdapter(BaseVideoModelAdapter):
    
    def _load_model(self):
        """加载你的多模态模型"""
        from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self.config["model_path"])
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.config["model_path"],
            device_map="auto"
        )
        
        # 可选：加载 LoRA 微调权重
        if self.config.get("use_lora"):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.config["lora_path"])
    
    def infer(self, video_path: str) -> InferenceResult:
        """推理单个视频，返回预测文本"""
        # 构建消息
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": self.config["prompt"]}
            ]
        }]
        
        # 推理
        # ... 你的推理代码 ...
        predicted_text = "模型预测的文本"
        
        return InferenceResult(
            video_path=video_path,
            predicted_text=predicted_text  # 这个字段会与 reference 对比计算指标
        )
```

### Step 3: 注册并运行

打开 `run_evaluation.py`，在 `create_model()` 函数中添加分支：

```python
elif model_type == "qwen_omni":
    from model_adapter.my_model import OmniModelAdapter
    return OmniModelAdapter(
        model_name=model_name,
        config={
            "model_path": "Qwen/Qwen2.5-Omni-7B",
            "device": "cuda",
            "prompt": "请识别视频中老年人说的话",
            "use_lora": False,
            "lora_path": None
        }
    )
```

运行：

```bash
python run_evaluation.py --model_type qwen_omni --video_dir data/videos --ref_dir data/references
```

---

## 📊 评测流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   视频文件   │────▶│   模型推理   │────▶│  预测文本   │
│  video.mp4  │     │  your model │     │ predicted   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐                                │  对比
│  参考文本   │────────────────────────────────┼───────▶ 指标分数
│ reference   │                                │        (BLEU, CER)
└─────────────┘                                │
```

---

## 📊 评测指标

| 指标 | 说明 | 值范围 | 推荐场景 |
|------|------|-------|---------|
| **BLEU-1** | 一元语法相似度 | 0~1（越高越好） | 通用 |
| **CER** | 字符错误率 | 0~∞（越低越好） | 中文语音识别 |

---

## 📁 项目结构

```
elderly_video_evaluation/
├── run_evaluation.py          # 主入口
├── simple_evaluator.py        # 评测器
│
├── model_adapter/             # 【核心】模型适配
│   ├── base_adapter.py        # 基类（必须继承）
│   └── custom_model_example.py # 模板（复制这个开始）⭐
│
├── data_loader/               # 数据加载
│   └── folder_loader.py       # 自动匹配视频和参考文本
│
├── metrics/                   # 评测指标
│   └── bleu_metrics.py        # BLEU、CER
│
└── data/                      # 示例数据
    └── evaluation_data.json
```

---

## 🚀 快速测试（使用虚拟模型）

```bash
# 使用 dummy 模型测试流程是否正常
python run_evaluation.py --model_type dummy --video_dir data/videos --ref_dir data/references
```

---

## 📄 许可证

MIT License
