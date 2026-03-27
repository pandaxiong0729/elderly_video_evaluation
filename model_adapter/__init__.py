"""
模型适配器模块
提供模型推理接口和示例实现
"""

from .base_adapter import BaseVideoModelAdapter, InferenceResult
from .model_examples import DummyModelAdapter, QwenVLAdapter, InternVLAdapter

__all__ = [
    'BaseVideoModelAdapter',
    'InferenceResult',
    'DummyModelAdapter',
    'QwenVLAdapter',
    'InternVLAdapter',
]
