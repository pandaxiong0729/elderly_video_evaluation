from model_adapter.base_adapter import BaseVideoModelAdapter, InferenceResult
from typing import Dict, Any
# 使用 Thinker 模型（只生成文本，不生成音频）支持批量推理
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
import torch
import re

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
        加载多模态模型 - 使用 Thinker 模型（只生成文本，支持批量推理）
        """
        model_path = self.config.get("model_path", "Qwen/Qwen2.5-Omni-7B")
        # 使用 Thinker 模型，不加载音频生成模块，节省显存且支持批量
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True 
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    def infer(self, video_path: str) -> InferenceResult:
        """
        对单个视频进行推理
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            InferenceResult: 必须包含 predicted_text 字段
        """       
        # 科研级零幻觉 Prompt（使用 XML 标签强制格式约束）
        system_prompt = """你是一个纯语音转写机器。

规则：
1. 只输出视频中人类说的原话
2. 不添加任何文字、解释、说明、前缀、后缀、引导
3. 不总结、不推理、不补充、不猜测
4. 不输出任何与语音无关的内容
5. 仅输出语音内容，一字不差

输出格式：
<transcript>这里只放语音转写内容，无其他任何文字</transcript>"""

        # Few-shot 示例：真实的老年人对话场景
        # 关键：包含实际可能出现的语气词、停顿、不完整句子
        few_shot_examples = """示例1：
音频中的话语：不不不 我那时候就是这样 有什么办法呢 哈哈
输出：<transcript>不不不我那时候就是这样有什么办法呢哈哈</transcript>

示例2：
音频中的话语：嗯 对 就是这么回事儿 我当时啊 就在那儿
输出：<transcript>嗯对就是这么回事儿我当时啊就在那儿</transcript>

示例3：
音频中的话语：解放以后呢 就不在那住了 是吧 对对对
输出：<transcript>解放以后呢就不在那住了是吧对对对</transcript>"""

        # 构建对话格式
        conversations = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"下面是一些转写示例，请学习格式：\n\n{few_shot_examples}"}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "我理解了。我会严格按照示例格式，只转写音频中的语音内容，放在 <transcript></transcript> 标签中，不添加任何额外信息。"}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "请转写这个视频中的所有语音内容，使用示例中的格式。"}
                ],
            },
        ]
        
        # 准备输入
        inputs = self.processor.apply_chat_template(
            conversations,
            load_audio_from_video=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            fps=1,
            padding=True,
            use_audio_in_video=True,
        ).to(self.model.device)
        
        # ========== 科研级零幻觉解码参数（完全确定性） ==========
        # 参考：所有客观生成评测论文的标准配置
        input_len = inputs['input_ids'].shape[1]
        
        text_ids = self.model.generate(
            **inputs,
            use_audio_in_video=True,
            # ===== 关键参数：完全消除幻觉 =====
            temperature=0.0,              # ✅ 完全确定性，禁止自由发挥（NO.1消幻觉手段）
            top_p=1.0,                    # 与 temperature=0 配套
            top_k=1,                      # ✅ 贪心解码，最保守选择
            do_sample=False,              # ✅ 确定性解码
            repetition_penalty=1.0,       # 不惩罚重复（语音转写可能有重复词）
            no_repeat_ngram_size=0,       # 允许重复 n-gram（转写内容可能有重复）
            # ===== 长度控制 =====
            max_new_tokens=150,           # 根据平均视频长度设置
            min_length=1,
            # ===== 其他稳定性参数 =====
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        # =============================================
        
        # 只取生成的新 token，跳过输入部分
        generated_ids = text_ids[:, input_len:]
        predicted_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 从 <transcript> 标签中提取纯文本（强制格式）
        match = re.search(r'<transcript>(.*?)</transcript>', predicted_text, re.DOTALL)
        if match:
            predicted_text = match.group(1).strip()
        
        # 清理
        del inputs, text_ids
        torch.cuda.empty_cache()
        
        return InferenceResult(
            video_path=video_path,
            predicted_text=predicted_text,
        )
