# 快速开始指南

## 5 分钟上手评测系统

### 步骤 1：安装依赖

```bash
cd elderly_video_evaluation
pip install -r requirements.txt
```

### 步骤 2：准备数据

```
data/
├── videos/              # 放入你的视频文件
│   ├── video1.mp4
│   └── video2.mp4
└── references/          # 放入对应的 SRT/TXT 文件
    ├── video1.srt       # 文件名必须与视频一致
    └── video2.txt
```

### 步骤 3：运行评测

```bash
python run_evaluation.py --video_dir data/videos --ref_dir data/references
```

### 步骤 4：查看结果

```bash
# 查看文本报告
cat results/*_report.txt

# 或用 Excel 打开 CSV
start results/*.csv
```

---

## 常用命令

```bash
# 使用 CER 指标（推荐中文）
python run_evaluation.py --video_dir data/videos --ref_dir data/references --use_cer

# 指定保存目录
python run_evaluation.py --video_dir data/videos --ref_dir data/references --save_dir my_results

# 查看所有参数
python run_evaluation.py --help
```

---

## 替换成你的模型

编辑 `model_adapter/custom_model_example.py`：

```python
class MyModelAdapter(BaseVideoModelAdapter):
    def infer(self, video_path: str):
        # 在这里写你的推理代码
        result = your_model.predict(video_path)
        return InferenceResult(video_path, result)
```

然后运行：

```bash
python run_evaluation.py --model_type my_model --video_dir videos --ref_dir references
```

---

更多文档请查看 [README.md](README.md)
