# -*- coding: utf-8 -*-
"""
项目全局配置文件
集中管理所有硬编码参数，其他脚本通过导入此模块获取配置
"""

# ========== 📁 文件路径配置 ==========
RESULTS_CSV = "results/results.csv"
ANALYSIS_DIR = "results/analysis"
PREDICTIONS_JSON = "results/predictions.json"
REPORT_FILE = "results/report.txt"

# ========== 🎬 视频切片配置 ==========
CLIP_DURATION = 30  # 秒
VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]

# 源视频和 SRT 目录
ORIGINAL_VIDEO_DIR = "videos"  # 修改为实际路径
ORIGINAL_SRT_DIR = "videos"    # 修改为实际路径

# 输出切片目录
CLIP_VIDEO_DIR = "clips_30s"      # 修改为实际路径
CLIP_SRT_DIR = "clips_30s"        # 修改为实际路径

# ========== 📊 评测指标配置 ==========
BLEU1_LOW_SCORE_THRESHOLD = 0.5   # BLEU-1 低分阈值
BLEU1_FILTER_THRESHOLD = 0.1      # BLEU-1 过滤阈值（低于此值视为异常）

# 使用的评测指标
USE_BLEU1 = True
USE_BLEU2 = False
USE_BLEU3 = False
USE_BLEU4 = False
USE_CER = False

# 低分视频显示数量
LOW_SCORE_DISPLAY_COUNT = 10

# ========== 🔤 文本预处理配置 ==========
# 数字转汉字映射表
DIGIT_TO_CHINESE = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
}

# 模型幻觉文本过滤规则（正则表达式）
HALLUCINATION_PATTERNS = [
    r"请根据上下文.*?翻译成中文[:：]?",
    r"以下是.*?的翻译",
    r"翻译结果[:：]",
]

# ========== 🤖 模型配置 ==========
# Qwen2.5-Omni 模型配置
QWEN_OMNI_CONFIG = {
    "model_path": "Qwen/Qwen2.5-Omni-7B",
    "temperature": 0.0,
    "top_k": 1,
    "do_sample": False,
    "max_new_tokens": 512,
}

# 视频处理配置
VIDEO_FPS = 2              # 每秒提取帧数
VIDEO_MAX_FRAMES = 16      # 最多提取帧数

# ========== 📝 CSV 编码配置 ==========
CSV_ENCODING = "utf-8-sig"  # 处理 Excel 生成的 UTF-8 带 BOM 文件

# ========== 📈 评测指标组合配置 ==========
METRIC_PROFILES = {
    "bleu1_only": {
        "description": "仅 BLEU-1（快速评测）",
        "use_bleu1": True,
        "use_bleu2": False,
        "use_bleu3": False,
        "use_bleu4": False,
        "use_cer": False,
    },
    "bleu_family": {
        "description": "BLEU-1/2/3/4（完整 BLEU）",
        "use_bleu1": True,
        "use_bleu2": True,
        "use_bleu3": True,
        "use_bleu4": True,
        "use_cer": False,
    },
    "bleu_cer": {
        "description": "BLEU-1 + CER（综合评测）",
        "use_bleu1": True,
        "use_bleu2": False,
        "use_bleu3": False,
        "use_bleu4": False,
        "use_cer": True,
    },
    "comprehensive": {
        "description": "所有指标（详细分析）",
        "use_bleu1": True,
        "use_bleu2": True,
        "use_bleu3": True,
        "use_bleu4": True,
        "use_cer": True,
    },
}

# ========== 🗂️ 命令行参数默认值 ==========
# 这些参数可以通过命令行覆盖
DEFAULT_CLIP_DURATION = CLIP_DURATION
DEFAULT_THRESHOLD = BLEU1_LOW_SCORE_THRESHOLD
DEFAULT_PROFILE = "bleu1_only"  # re_evaluate.py 默认评测策略
