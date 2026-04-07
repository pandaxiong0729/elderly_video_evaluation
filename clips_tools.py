import subprocess
import re
from pathlib import Path
from typing import List, Tuple
from config import ORIGINAL_VIDEO_DIR, ORIGINAL_SRT_DIR, CLIP_VIDEO_DIR, CLIP_SRT_DIR, CLIP_DURATION, VIDEO_FORMATS


def parse_srt_time(time_str: str) -> float:
    """解析 SRT 时间格式 -> 秒"""
    time_str = time_str.replace(',', '.')
    match = re.match(r'(\d+):(\d+):(\d+)\.(\d+)', time_str)
    if not match:
        return 0.0
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def seconds_to_srt_time(seconds: float) -> str:
    """秒 -> SRT 时间格式"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_srt_segments(srt_path: str) -> List[Tuple[float, float, str]]:
    """
    从 SRT 文件提取所有字幕块
    返回: [(start_time, end_time, text), ...]
    """
    srt_path = Path(srt_path)
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(srt_path, 'r', encoding='gbk') as f:
            content = f.read()
    
    segments = []
    blocks = re.split(r'\n\n+', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # 找到时间行
        time_line = None
        text_start = 0
        for i, line in enumerate(lines):
            if '-->' in line:
                time_line = line
                text_start = i + 1
                break
        
        if not time_line:
            continue
        
        # 解析时间
        parts = time_line.split('-->')
        if len(parts) != 2:
            continue
        
        start_str = parts[0].strip()
        end_str = parts[1].strip()
        start_sec = parse_srt_time(start_str)
        end_sec = parse_srt_time(end_str)
        
        # 提取文本
        text = ' '.join(lines[text_start:]).strip()
        
        segments.append((start_sec, end_sec, text))
    
    return segments


def group_segments_by_clips(segments: List[Tuple[float, float, str]], 
                             clip_duration: float = None) -> List[List[Tuple[float, float, str]]]:
    """
    将字幕块按时间分组，每组对应一个~30s的视频片段
    返回: [[(t1, t2, txt), ...], [(t3, t4, txt), ...], ...]
    """
    if clip_duration is None:
        clip_duration = CLIP_DURATION
    
    clips = {}
    
    for start_time, end_time, text in segments:
        # 判断这个字幕属于哪个片段
        # 使用字幕的开始时间来判断
        clip_idx = int(start_time // clip_duration)
        
        if clip_idx not in clips:
            clips[clip_idx] = []
        
        clips[clip_idx].append((start_time, end_time, text))
    
    return [clips[i] for i in sorted(clips.keys())]


def create_srt_file(segments: List[Tuple[float, float, str]], output_path: str):
    """
    从字幕块生成 SRT 文件
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    srt_lines = []
    for idx, (start_time, end_time, text) in enumerate(segments, 1):
        start_srt = seconds_to_srt_time(start_time)
        end_srt = seconds_to_srt_time(end_time)
        
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_srt} --> {end_srt}")
        srt_lines.append(text)
        srt_lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_lines))


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1:noprint_wrappers=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"  ⚠️  无法获取视频时长: {e}")
        return 0.0


def cut_video_segment(input_video: str, start_time: float, end_time: float, 
                      output_path: str):
    """切割视频的一个片段"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c:v", "copy",
        "-c:a", "aac",
        str(output_path)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_all(video_dir: str = None, srt_dir: str = None, 
                 output_video_dir: str = None, output_srt_dir: str = None,
                 clip_duration: float = None):
    """
    批量处理所有视频和 SRT
    
    Args:
        video_dir: 原始视频目录（默认使用配置）
        srt_dir: 原始字幕目录（默认使用配置）
        output_video_dir: 切割后视频输出目录（默认使用配置）
        output_srt_dir: 切割后字幕输出目录（默认使用配置）
        clip_duration: 切片时长（秒，默认使用配置）
    """
    # 使用传入参数或默认值
    video_dir = video_dir or ORIGINAL_VIDEO_DIR
    srt_dir = srt_dir or ORIGINAL_SRT_DIR
    output_video_dir = output_video_dir or CLIP_VIDEO_DIR
    output_srt_dir = output_srt_dir or CLIP_SRT_DIR
    clip_duration = clip_duration or CLIP_DURATION
    
    # 转换字符串路径为 Path 对象
    clip_video_dir = Path(output_video_dir)
    clip_srt_dir = Path(output_srt_dir)
    
    print("\n" + "="*80)
    print("📹 视频和字幕完整切片工具（视频和SRT完全对齐）")
    print("="*80)
    print(f"配置：切片时长 = {clip_duration} 秒")
    print("原理：")
    print("  1. 从 original/*.srt 提取字幕块和时间戳")
    print(f"  2. 按 ~{clip_duration}s 时间窗口分组")
    print("  3. 精确切割视频")
    print("  4. 生成对应的 SRT（保留完整格式）")
    print()
    
    # 1. 扫描 SRT 目录
    srt_dir_path = Path(srt_dir)
    if not srt_dir_path.exists():
        print(f"❌ SRT 目录不存在: {srt_dir}")
        return
    
    srt_files = sorted(srt_dir_path.glob("*.srt"))
    if not srt_files:
        print(f"❌ SRT 目录中没有找到 .srt 文件: {srt_dir}")
        return
    
    print(f"📋 找到 {len(srt_files)} 个 SRT 文件\n")
    
    # 2. 处理每个 SRT 对应的视频
    video_dir_path = Path(video_dir)
    if not video_dir_path.exists():
        print(f"❌ 视频目录不存在: {video_dir}")
        return
    
    print("-"*80)
    print("正在处理...")
    print("-"*80)
    
    processed_count = 0
    
    for srt_path in srt_files:
        srt_name = srt_path.stem  # 不含后缀的名称，如 test10
        
        # 查找对应的视频
        video_found = False
        video_path = None
        
        for ext in VIDEO_FORMATS:
            candidates = list(video_dir_path.glob(f"{srt_name}.{ext}"))
            if candidates:
                video_path = candidates[0]
                video_found = True
                break
        
        if not video_found:
            print(f"\n⚠️  {srt_path.name} 找不到对应的视频，跳过")
            continue
        
        print(f"\n▶️  处理: {srt_name}")
        print(f"   视频: {video_path.name}")
        print(f"   SRT:  {srt_path.name}")
        
        # 3. 提取字幕块
        segments = extract_srt_segments(str(srt_path))
        if not segments:
            print(f"   ⚠️  无法从 SRT 提取字幕块，跳过")
            continue
        
        print(f"   📊 提取到 {len(segments)} 个字幕块")
        
        # 4. 按时间分组
        clip_groups = group_segments_by_clips(segments, clip_duration=clip_duration)
        print(f"   🔪 分组为 {len(clip_groups)} 个视频片段（~{clip_duration}s/个）")
        
        # 5. 验证视频时长
        video_duration = get_video_duration(str(video_path))
        if video_duration == 0:
            print(f"   ❌ 无法获取视频时长，跳过")
            continue
        
        # 6. 逐个切割
        for clip_idx, clip_segments in enumerate(clip_groups):
            # 计算切割时间点（使用该分组中所有字幕的最小和最大时间）
            start_time = min(s[0] for s in clip_segments)
            end_time = max(s[1] for s in clip_segments)
            
            # 确保不超过视频时长
            end_time = min(end_time, video_duration)
            duration = end_time - start_time
            
            # 输出文件名（使用 Path 对象）
            output_video = clip_video_dir / f"{srt_name}_{clip_idx:03d}.mp4"
            output_srt = clip_srt_dir / f"{srt_name}_{clip_idx:03d}.srt"
            
            print(f"   📹 [{clip_idx+1}/{len(clip_groups)}] {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
            
            # 切割视频
            cut_video_segment(str(video_path), start_time, end_time, str(output_video))
            
            # 生成 SRT（调整时间为相对于片段的开始时间）
            adjusted_segments = [
                (s[0] - start_time, s[1] - start_time, s[2])
                for s in clip_segments
            ]
            create_srt_file(adjusted_segments, str(output_srt))
        
        print(f"   ✅ 完成: {len(clip_groups)} 个片段")
        processed_count += 1
    
    print("\n" + "="*80)
    if processed_count > 0:
        print(f"✅ 完成！共处理 {processed_count} 个视频")
        print("="*80)
        print(f"\n输出目录：")
        print(f"  视频: {output_video_dir}")
        print(f"  SRT:  {output_srt_dir}")
        print(f"\n视频和 SRT 已完全对齐")
        print(f"\n运行评测:")
        print(f"  python run_evaluation.py --video_dir {output_video_dir} --ref_dir {output_srt_dir}")
    else:
        print(f"❌ 未处理任何视频")
        print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="视频和字幕切片工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用默认配置（30秒）
  python clips_tools.py
  
  # 自定义切片时长（60秒）
  python clips_tools.py --clip_duration 60
  
  # 自定义路径和时长
  python clips_tools.py --clip_duration 45 --video_dir /path/to/videos
        """
    )
    
    # 路径参数
    parser.add_argument("--video_dir", type=str, default=None,
                       help="原始视频目录路径")
    parser.add_argument("--srt_dir", type=str, default=None,
                       help="原始字幕目录路径")
    parser.add_argument("--output_video_dir", type=str, default=None,
                       help="切割后视频输出目录")
    parser.add_argument("--output_srt_dir", type=str, default=None,
                       help="切割后字幕输出目录")
    
    # 切片配置
    parser.add_argument("--clip_duration", type=float, default=None,
                       help=f"单个视频片段时长（秒），默认: {CLIP_DURATION}")
    
    args = parser.parse_args()
    
    # 使用命令行参数覆盖配置
    if args.clip_duration:
        clip_duration_override = args.clip_duration
    else:
        clip_duration_override = CLIP_DURATION
    
    if args.video_dir:
        video_dir_override = args.video_dir
    else:
        video_dir_override = ORIGINAL_VIDEO_DIR
    
    if args.srt_dir:
        srt_dir_override = args.srt_dir
    else:
        srt_dir_override = ORIGINAL_SRT_DIR
    
    if args.output_video_dir:
        output_video_override = args.output_video_dir
    else:
        output_video_override = CLIP_VIDEO_DIR
    
    if args.output_srt_dir:
        output_srt_override = args.output_srt_dir
    else:
        output_srt_override = CLIP_SRT_DIR
    
    process_all(
        video_dir=video_dir_override,
        srt_dir=srt_dir_override,
        output_video_dir=output_video_override,
        output_srt_dir=output_srt_override,
        clip_duration=clip_duration_override
    )
