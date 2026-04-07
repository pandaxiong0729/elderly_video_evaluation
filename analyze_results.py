import pandas as pd
from pathlib import Path
from config import RESULTS_CSV, ANALYSIS_DIR, BLEU1_LOW_SCORE_THRESHOLD, LOW_SCORE_DISPLAY_COUNT, CSV_ENCODING


def analyze_results(csv_path: str = None, output_dir: str = None, threshold: float = None, metric: str = None):
    """
    分析评测结果，分离低分和高分视频
    
    Args:
        csv_path: 输入 CSV 文件路径（默认使用配置文件）
        output_dir: 输出目录（默认使用配置文件）
        threshold: 分数阈值（默认使用配置文件）
        metric: 要分析的指标列名（如 'BLEU-1', 'BLEU-2', 'BLEU-4', 'CER' 等）
               如果不指定，会自动查找 BLEU-1 列
    """
    # 使用配置文件中的默认值
    csv_path = csv_path or RESULTS_CSV
    output_dir = output_dir or ANALYSIS_DIR
    threshold = threshold or BLEU1_LOW_SCORE_THRESHOLD
    
    # 读取 CSV
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return
    
    print("\n" + "="*80)
    print("📊 评测结果分析工具")
    print("="*80)
    print(f"📁 输入文件: {csv_path}")
    print(f"📈 分析阈值: {threshold}\n")
    
    # 读取数据
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return
    
    print(f"📋 总行数: {len(df)}\n")
    print("可用的列:")
    for col in df.columns:
        print(f"  - {col}")
    print()
    
    # 自动查找或手动指定指标列
    metric_col = None
    
    if metric:
        # 用户指定了指标名称，尝试精确匹配或模糊匹配
        if metric in df.columns:
            metric_col = metric
        else:
            # 尝试模糊匹配
            for col in df.columns:
                if metric.lower() in col.lower():
                    metric_col = col
                    break
    
    # 如果没有找到或未指定，自动查找 BLEU-1
    if not metric_col:
        for col in df.columns:
            if 'bleu' in col.lower() and '1' in col:
                metric_col = col
                break
    
    if not metric_col:
        print("❌ 找不到指定的指标列")
        print(f"可用的列: {list(df.columns)}")
        print(f"\n💡 可以通过 --metric 参数指定，例如:")
        print(f"   python analyze_results.py --metric 'BLEU-1'")
        print(f"   python analyze_results.py --metric 'CER'")
        return
    
    print(f"✓ 分析指标: {metric_col}\n")
    
    # 分离低分和高分
    low_score_df = df[df[metric_col] < threshold].copy()
    high_score_df = df[df[metric_col] >= threshold].copy()
    
    print(f"🔴 低分样本 ({metric_col} < {threshold}): {len(low_score_df)} 个")
    print(f"🟢 高分样本 ({metric_col} >= {threshold}): {len(high_score_df)} 个")
    print()
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存低分和高分样本
    metric_name = metric_col.replace('-', '_').replace(' ', '_').lower()
    low_score_file = output_path / f"low_score_{metric_name}_below_{threshold}.csv"
    low_score_df.to_csv(low_score_file, index=False, encoding=CSV_ENCODING)
    print(f"✓ 低分样本已保存: {low_score_file}")
    
    # 保存高分样本
    high_score_file = output_path / f"high_score_{metric_name}_above_{threshold}.csv"
    high_score_df.to_csv(high_score_file, index=False, encoding=CSV_ENCODING)
    print(f"✓ 高分样本已保存: {high_score_file}")
    print()
    
    # ========== 统计信息 ==========
    print("-"*80)
    print("📈 统计信息")
    print("-"*80)
    
    # 原始平均分
    original_avg = df[metric_col].mean()
    print(f"\n【原始数据】")
    print(f"  所有样本平均 {metric_col}: {original_avg:.4f}")
    print(f"  最高分: {df[metric_col].max():.4f}")
    print(f"  最低分: {df[metric_col].min():.4f}")
    print(f"  中位数: {df[metric_col].median():.4f}")
    
    # 低分样本统计
    if len(low_score_df) > 0:
        print(f"\n【低分样本】(< {threshold})")
        print(f"  数量: {len(low_score_df)}")
        print(f"  平均 {metric_col}: {low_score_df[metric_col].mean():.4f}")
        print(f"  最高分: {low_score_df[metric_col].max():.4f}")
        print(f"  最低分: {low_score_df[metric_col].min():.4f}")
    else:
        print(f"\n【低分样本】(< {threshold})")
        print(f"  数量: 0")
    
    # 高分样本统计
    if len(high_score_df) > 0:
        print(f"\n【高分样本】(>= {threshold})")
        print(f"  数量: {len(high_score_df)}")
        print(f"  平均 {metric_col}: {high_score_df[metric_col].mean():.4f}")
        print(f"  最高分: {high_score_df[metric_col].max():.4f}")
        print(f"  最低分: {high_score_df[metric_col].min():.4f}")
        print(f"  ✅ 重新计算的平均 {metric_col} (仅高分): {high_score_df[metric_col].mean():.4f}")
    else:
        print(f"\n【高分样本】(>= {threshold})")
        print(f"  数量: 0")
    
    print()
    
    # ========== 低分视频分析 ==========
    if len(low_score_df) > 0:
        print("-"*80)
        print("🔍 低分视频详细列表")
        print("-"*80)
        
        # 按指标分数排序，显示最低的10个
        low_sorted = low_score_df.sort_values(metric_col).head(LOW_SCORE_DISPLAY_COUNT)
        
        for idx, row in low_sorted.iterrows():
            video_col = None
            for col in low_score_df.columns:
                if 'video' in col.lower() or 'file' in col.lower() or 'name' in col.lower():
                    video_col = col
                    break
            
            if video_col:
                print(f"  {row[video_col]}: {metric_col} = {row[metric_col]:.4f}")
            else:
                print(f"  样本 {idx}: {metric_col} = {row[metric_col]:.4f}")
        
        if len(low_score_df) > 10:
            print(f"  ... 以及其他 {len(low_score_df) - 10} 个低分视频\n")
    
    print("="*80)
    print("✅ 分析完成！")
    print("="*80)
    print(f"\n📌 建议：")
    print(f"  1. 查看 {low_score_file.name} 找出低分原因")
    print(f"  2. 可能需要改进 prompt 或模型参数")
    if len(high_score_df) > 0:
        print(f"  3. 高分样本的平均 {metric_col} 为: {high_score_df[metric_col].mean():.4f}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="评测结果分析工具 - 分离低分和高分视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用默认配置（BLEU-1, 阈值 0.5）
  python analyze_results.py
  
  # 分析 BLEU-2 指标，自定义阈值
  python analyze_results.py --metric 'BLEU-2' --threshold 0.4
  
  # 分析 CER 指标（越低越好）
  python analyze_results.py --metric 'CER' --threshold 0.15
  
  # 分析 BLEU-4 指标
  python analyze_results.py --metric 'BLEU-4' --threshold 0.3
  
  # 自定义 CSV 文件和输出目录
  python analyze_results.py --csv results.csv --output_dir analysis --metric 'BLEU-1' --threshold 0.5
        """
    )
    
    parser.add_argument("--csv", type=str, default=None,
                       help=f"评测结果 CSV 文件，默认: {RESULTS_CSV}")
    parser.add_argument("--output_dir", type=str, default=None,
                       help=f"输出目录，默认: {ANALYSIS_DIR}")
    parser.add_argument("--threshold", type=float, default=None,
                       help=f"分数阈值，默认: {BLEU1_LOW_SCORE_THRESHOLD}")
    parser.add_argument("--metric", type=str, default=None,
                       help="分析的指标名称 (如 'BLEU-1', 'BLEU-2', 'BLEU-4', 'CER' 等)，默认自动查找 BLEU-1")
    
    args = parser.parse_args()
    
    analyze_results(
        csv_path=args.csv,
        output_dir=args.output_dir,
        threshold=args.threshold,
        metric=args.metric
    )
