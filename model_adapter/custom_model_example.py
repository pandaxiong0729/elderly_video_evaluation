"""
多模态模型 (Omni) 适配器模板

支持的模型类型：
- Qwen2.5-Omni (Qwen/Qwen2.5-Omni-7B)
- Qwen2-VL
- InternVL
- LLaVA
- MiniCPM-V

============= 使用步骤 =============

Step 1: 复制这个文件
    cp custom_model_example.py my_model.py

Step 2: 修改推理逻辑
    - _load_model() → 加载你的模型
    - infer() → 实现视频推理

Step 3: 注册到 run_evaluation.py

=========================================
"""

from model_adapter.base_adapter import BaseVideoModelAdapter, InferenceResult
from typing import Dict, Any


class OmniModelAdapter(BaseVideoModelAdapter):
    """
    多模态模型适配器模板
    
    适用于 Qwen2.5-Omni, Qwen2-VL, InternVL 等视频理解模型
    
    你需要实现两个方法：
    1. _load_model() - 加载模型权重
    2. infer() - 对视频进行推理，返回预测文本
    """
    
    def __init__(self, model_name: str = "omni_model", config: Dict[str, Any] = None):
        """
        初始化
        
        Args:
            model_name: 模型名称（用于结果记录）
            config: 配置参数：
                {
                    "model_path": "Qwen/Qwen2.5-Omni-7B",  # HuggingFace 模型路径
                    "device": "cuda",
                    "prompt": "请识别视频中老年人说的话",
                    "max_frames": 16,      # 最大帧数
                    "use_lora": False,     # 是否使用 LoRA 微调权重
                    "lora_path": None,     # LoRA 权重路径
                }
        """
        super().__init__(model_name, config)
    
    def _load_model(self):
        """
        加载多模态模型
        """
        model_path = self.config.get("model_path", "Qwen/Qwen2.5-Omni-7B")
        device = self.config.get("device", "cuda")
        use_lora = self.config.get("use_lora", False)
        lora_path = self.config.get("lora_path", None)
        
        print(f"[{self.model_name}] 加载模型...")
        print(f"  模型路径: {model_path}")
        print(f"  设备: {device}")
        
        # ============ 示例：Qwen2.5-Omni 模型 ============
        # from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
        # 
        # self.processor = AutoProcessor.from_pretrained(model_path)
        # self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        #     model_path,
        #     device_map=device,
        #     torch_dtype="auto"
        # )
        # 
        # # 如果使用 LoRA 微调权重
        # if use_lora and lora_path:
        #     from peft import PeftModel
        #     self.model = PeftModel.from_pretrained(self.model, lora_path)
        #     print(f"  已加载 LoRA: {lora_path}")
        
        # ============ 示例：Qwen2-VL 模型 ============
        # from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        # 
        # self.processor = AutoProcessor.from_pretrained(model_path)
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model_path,
        #     device_map=device,
        #     torch_dtype="auto"
        # )
        
        # ============ 示例：InternVL 模型 ============
        # from transformers import AutoModel, AutoTokenizer
        # 
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # self.model = AutoModel.from_pretrained(
        #     model_path,
        #     device_map=device,
        #     trust_remote_code=True
        # )
        
        # TODO: 替换成你的模型加载代码
        self.model = None
        self.processor = None
    
    def infer(self, video_path: str) -> InferenceResult:
        """
        对单个视频进行推理
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            InferenceResult: 必须包含 predicted_text 字段
        """
        prompt = self.config.get("prompt", "请仔细听这段视频中老年人说的话，并转录成文字")
        max_frames = self.config.get("max_frames", 16)
        
        # ============ 示例：Qwen2.5-Omni 推理 ============
        # from qwen_omni_utils import process_mm_info
        # 
        # messages = [{
        #     "role": "user",
        #     "content": [
        #         {"type": "video", "video": video_path},
        #         {"type": "text", "text": prompt}
        #     ]
        # }]
        # 
        # text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        # inputs = self.processor(
        #     text=text, audios=audios, images=images, videos=videos,
        #     return_tensors="pt", padding=True
        # ).to(self.model.device)
        # 
        # outputs = self.model.generate(**inputs, max_new_tokens=256)
        # predicted_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # ============ 示例：Qwen2-VL 推理 ============
        # from qwen_vl_utils import process_vision_info
        # 
        # messages = [{
        #     "role": "user",
        #     "content": [
        #         {"type": "video", "video": video_path, "max_pixels": 360*420, "fps": 1.0},
        #         {"type": "text", "text": prompt}
        #     ]
        # }]
        # 
        # text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = self.processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     return_tensors="pt"
        # ).to(self.model.device)
        # 
        # outputs = self.model.generate(**inputs, max_new_tokens=256)
        # predicted_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # ============ 示例：InternVL 推理 ============
        # pixel_values = load_video(video_path, max_num=max_frames)
        # response = self.model.chat(self.tokenizer, pixel_values, prompt)
        # predicted_text = response
        
        # TODO: 替换成你的推理代码
        predicted_text = "这里是模型预测的文本"
        
        return InferenceResult(
            video_path=video_path,
            predicted_text=predicted_text,
            confidence=1.0,
            metadata={"model": self.model_name}
        )


# ============================================================
# 注册到评测系统
# ============================================================
# 
# 方法 1：在 run_evaluation.py 中添加分支
# 
#   打开 run_evaluation.py，找到 create_model 函数，添加：
#
#   elif model_type == "qwen_omni":
#       from model_adapter.my_model import OmniModelAdapter
#       return OmniModelAdapter(
#           model_name=model_name,
#           config={
#               "model_path": "Qwen/Qwen2.5-Omni-7B",
#               "device": "cuda",
#               "prompt": "请识别视频中老年人说的话",
#               "use_lora": True,           # 可选：是否使用微调权重
#               "lora_path": "path/to/lora"  # 可选：LoRA 权重路径
#           }
#       )
#
#   然后运行：
#   python run_evaluation.py --model_type qwen_omni --video_dir videos --ref_dir refs
#
# ============================================================
# 
# 方法 2：直接写脚本（更灵活）
# 
#   见下方 __main__ 部分
#
# ============================================================


if __name__ == "__main__":
    """
    直接运行示例 - 不使用 run_evaluation.py
    
    python model_adapter/custom_model_example.py
    """
    from pathlib import Path
    import sys
    
    # 添加项目根目录到路径
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data_loader.folder_loader import FolderDataLoader
    from metrics.bleu_metrics import BLEU1, CER
    from simple_evaluator import SimpleEvaluator
    
    # ============ 配置 ============
    VIDEO_DIR = "data/videos"        # 视频文件夹
    REF_DIR = "data/references"      # 参考文本文件夹（文件名需与视频对应）
    MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"
    SAVE_DIR = "results"
    # ==============================
    
    # 1. 创建你的模型
    model = OmniModelAdapter(
        model_name="qwen_omni_7b",
        config={
            "model_path": MODEL_PATH,
            "device": "cuda",
            "prompt": "请仔细听这段视频中老年人说的话，并转录成文字",
            "max_frames": 16,
            # "use_lora": True,
            # "lora_path": "path/to/lora_weights"
        }
    )
    
    # 2. 加载数据
    # reference 文件夹中的 .srt/.txt 文件会自动与视频匹配
    # 例如：video1.mp4 对应 video1.srt 或 video1.txt
    data_loader = FolderDataLoader(
        video_dir=VIDEO_DIR,
        reference_dir=REF_DIR
    )
    print(f"加载了 {len(data_loader)} 个样本")
    
    # 3. 创建评测指标
    # BLEU-1: 一元语法相似度
    # CER: 字符错误率（中文评测推荐）
    metrics = [BLEU1(), CER()]
    
    # 4. 运行评测
    evaluator = SimpleEvaluator(model=model, metrics=metrics, verbose=True)
    report = evaluator.evaluate(
        data_loader=data_loader,
        save_dir=SAVE_DIR,
        save_json=True,
        save_csv=True
    )
    
    # 5. 打印结果
    print("\n" + "="*50)
    print("评测结果")
    print("="*50)
    for metric_name, score in report.metrics_summary.items():
        print(f"  {metric_name}: {score:.4f}")
