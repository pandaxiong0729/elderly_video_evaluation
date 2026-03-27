"""
模型适配器接口定义
所有模型适配器必须继承此基类，实现统一的推理接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """推理结果数据结构"""
    video_path: str
    predicted_text: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None


class BaseVideoModelAdapter(ABC):
    """视频识别模型适配器基类"""
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        """
        初始化模型适配器
        
        Args:
            model_name: 模型名称
            config: 模型配置参数
        """
        self.model_name = model_name
        self.config = config or {}
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """加载模型（由子类实现）"""
        pass
    
    @abstractmethod
    def infer(self, video_path: str) -> InferenceResult:
        """
        对单个视频进行推理
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            InferenceResult: 推理结果
        """
        pass
    
    def infer_batch(self, video_paths: List[str]) -> List[InferenceResult]:
        """
        批量推理（默认实现，子类可重写优化）
        
        Args:
            video_paths: 视频文件路径列表
            
        Returns:
            List[InferenceResult]: 推理结果列表
        """
        results = []
        for path in video_paths:
            result = self.infer(path)
            results.append(result)
        return results
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"
