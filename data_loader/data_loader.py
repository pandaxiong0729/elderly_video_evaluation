"""
数据加载器
负责加载评测数据集，提供统一的接口
"""

import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationSample:
    """评测样本数据结构"""
    video_path: str
    reference_text: str  # ground truth
    metadata: Dict = None


class BaseDataLoader:
    """数据加载器基类"""
    
    def __init__(self, data_dir: str):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.samples: List[EvaluationSample] = []
        self._load_data()
    
    def _load_data(self):
        """加载数据（由子类实现）"""
        pass
    
    def get_samples(self) -> List[EvaluationSample]:
        """获取所有样本"""
        return self.samples
    
    def get_video_paths(self) -> List[str]:
        """获取所有视频路径"""
        return [sample.video_path for sample in self.samples]
    
    def get_references(self) -> List[str]:
        """获取所有参考文本"""
        return [sample.reference_text for sample in self.samples]
    
    def __len__(self):
        return len(self.samples)


class JsonDataLoader(BaseDataLoader):
    """
    JSON 格式数据加载器
    
    数据格式示例：
    [
        {
            "video_path": "path/to/video1.mp4",
            "reference_text": "参考文本内容",
            "metadata": {"speaker": "老人 A", "age": 75}
        },
        ...
    ]
    """
    
    def __init__(self, json_path: str, video_dir: str = None):
        """
        初始化 JSON 数据加载器
        
        Args:
            json_path: JSON 文件路径
            video_dir: 视频文件目录（可选，用于拼接完整路径）
        """
        self.json_path = Path(json_path)
        self.video_dir = Path(video_dir) if video_dir else None
        super().__init__(str(self.json_path.parent))
    
    def _load_data(self):
        """从 JSON 文件加载数据"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件不存在：{self.json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 格式错误：{e}")
        
        self.samples = []
        for i, item in enumerate(data):
            if 'video_path' not in item:
                raise ValueError(f"第{i+1}条数据缺少 video_path 字段")
            if 'reference_text' not in item:
                raise ValueError(f"第{i+1}条数据缺少 reference_text 字段")
            
            video_path = item.get('video_path', '')
            
            # 如果指定了视频目录，拼接完整路径
            if self.video_dir and not os.path.isabs(video_path):
                video_path = str(self.video_dir / video_path)
            
            sample = EvaluationSample(
                video_path=video_path,
                reference_text=item.get('reference_text', ''),
                metadata=item.get('metadata', {})
            )
            self.samples.append(sample)
        
        print(f"✓ 加载了 {len(self.samples)} 个评测样本")


class CSVDataloader(BaseDataLoader):
    """
    CSV 格式数据加载器
    
    CSV 格式：
    video_path,reference_text,metadata
    path/to/video1.mp4，参考文本 1，{"speaker": "老人 A"}
    """
    
    def __init__(self, csv_path: str, video_dir: str = None):
        """
        初始化 CSV 数据加载器
        
        Args:
            csv_path: CSV 文件路径
            video_dir: 视频文件目录
        """
        import csv
        self.csv_path = Path(csv_path)
        self.video_dir = Path(video_dir) if video_dir else None
        super().__init__(str(self.csv_path.parent))
    
    def _load_data(self):
        """从 CSV 文件加载数据"""
        import csv
        
        self.samples = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = row.get('video_path', '')
                
                if self.video_dir and not os.path.isabs(video_path):
                    video_path = str(self.video_dir / video_path)
                
                # 解析 metadata（JSON 字符串）
                metadata_str = row.get('metadata', '{}')
                try:
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}
                
                sample = EvaluationSample(
                    video_path=video_path,
                    reference_text=row.get('reference_text', ''),
                    metadata=metadata
                )
                self.samples.append(sample)
        
        print(f"✓ 加载了 {len(self.samples)} 个评测样本")
