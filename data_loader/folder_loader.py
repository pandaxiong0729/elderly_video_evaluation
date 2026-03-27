"""
文件夹数据加载器
从文件夹中加载视频和对应的参考文本（srt/txt）

文件夹结构：
data/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── references/
    ├── video1.srt (或 video1.txt)
    ├── video2.srt (或 video2.txt)
    └── ...
"""

import os
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class EvaluationSample:
    """评测样本数据结构"""
    video_path: str
    reference_text: str  # ground truth
    metadata: Dict = None


class FolderDataLoader:
    """
    文件夹数据加载器
    
    自动匹配视频文件和参考文本文件
    支持 .srt 和 .txt 格式的参考文本
    """
    
    def __init__(self, 
                 video_dir: str, 
                 reference_dir: str,
                 video_extensions: List[str] = None,
                 ref_extensions: List[str] = None):
        """
        初始化文件夹数据加载器
        
        Args:
            video_dir: 视频文件目录
            reference_dir: 参考文本目录（包含 srt/txt 文件）
            video_extensions: 视频文件扩展名列表（默认：['.mp4', '.avi', '.mov', '.mkv']）
            ref_extensions: 参考文件扩展名列表（默认：['.srt', '.txt']）
        """
        self.video_dir = Path(video_dir)
        self.reference_dir = Path(reference_dir)
        self.video_extensions = video_extensions or ['.mp4', '.avi', '.mov', '.mkv']
        self.ref_extensions = ref_extensions or ['.srt', '.txt']
        
        self.samples: List[EvaluationSample] = []
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        print(f"\n📂 加载数据...")
        print(f"  视频目录：{self.video_dir}")
        print(f"  参考目录：{self.reference_dir}")
        
        # 获取所有视频文件
        video_files = self._get_files(self.video_dir, self.video_extensions)
        print(f"  找到 {len(video_files)} 个视频文件")
        
        # 获取所有参考文件
        ref_files = self._get_files(self.reference_dir, self.ref_extensions)
        print(f"  找到 {len(ref_files)} 个参考文件")
        
        # 创建文件名到参考文件的映射
        ref_map = {}
        for ref_path in ref_files:
            # 获取文件名（不含扩展名）
            base_name = ref_path.stem
            ref_map[base_name] = ref_path
        
        # 匹配视频和参考文本
        matched_count = 0
        unmatched_videos = []
        
        for video_path in video_files:
            base_name = video_path.stem
            
            if base_name in ref_map:
                # 找到匹配的参考文件
                ref_path = ref_map[base_name]
                ref_text = self._read_reference(ref_path)
                
                sample = EvaluationSample(
                    video_path=str(video_path),
                    reference_text=ref_text,
                    metadata={'reference_file': str(ref_path)}
                )
                self.samples.append(sample)
                matched_count += 1
            else:
                unmatched_videos.append(video_path.name)
        
        print(f"  ✓ 成功匹配 {matched_count} 对视频 - 参考文本")
        
        if unmatched_videos:
            print(f"  ⚠️  {len(unmatched_videos)} 个视频未找到参考文本:")
            for name in unmatched_videos[:5]:  # 只显示前 5 个
                print(f"     - {name}")
            if len(unmatched_videos) > 5:
                print(f"     ... 还有 {len(unmatched_videos) - 5} 个")
        
        if len(self.samples) == 0:
            print("\n❌ 错误：没有匹配到任何视频 - 参考文本对！")
            print("请确保视频文件和参考文件的文件名一致（扩展名可以不同）")
            print("例如：video1.mp4 对应 video1.srt 或 video1.txt")
    
    def _get_files(self, directory: Path, extensions: List[str]) -> List[Path]:
        """获取目录下指定扩展名的所有文件"""
        files = []
        if not directory.exists():
            print(f"  ⚠️  目录不存在：{directory}")
            return files
        
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))  # 支持大写扩展名
        
        return sorted(files)
    
    def _read_reference(self, ref_path: Path) -> str:
        """
        读取参考文本文件
        
        Args:
            ref_path: 参考文件路径
            
        Returns:
            str: 纯文本内容（SRT 会自动提取文本）
        """
        with open(ref_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 如果是 SRT 文件，提取纯文本
        if ref_path.suffix.lower() == '.srt':
            text = self._parse_srt(content)
        else:
            text = content.strip()
        
        return text
    
    def _parse_srt(self, srt_content: str) -> str:
        """
        解析 SRT 字幕文件，提取纯文本
        
        SRT 格式：
        1
        00:00:01,000 --> 00:00:04,000
        这是第一句字幕
        
        2
        00:00:05,000 --> 00:00:08,000
        这是第二句字幕
        """
        # 移除时间戳和序号
        lines = srt_content.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # 跳过空行、序号行、时间戳行
            if not line:
                continue
            if re.match(r'^\d+$', line):  # 序号
                continue
            if re.match(r'^\d{2}:\d{2}:\d{2}', line):  # 时间戳
                continue
            # 保留文本行
            text_lines.append(line)
        
        return ' '.join(text_lines)
    
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


# 使用示例
if __name__ == "__main__":
    # 示例用法
    loader = FolderDataLoader(
        video_dir="data/videos",
        reference_dir="data/references"
    )
    
    print(f"\n加载了 {len(loader)} 个样本")
    for sample in loader.get_samples()[:3]:  # 显示前 3 个
        print(f"\n视频：{sample.video_path}")
        print(f"参考文本：{sample.reference_text[:50]}...")
