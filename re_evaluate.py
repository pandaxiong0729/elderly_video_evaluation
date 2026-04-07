"""
重新评测工具 - 支持不同的评测指标组合
使用硬编码的不同配置快速切换评测策略
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict
from config import RESULTS_CSV, RESULTS_DIR, BLEU1_FILTER_THRESHOLD, CSV_ENCODING
from metrics.bleu_metrics import BLEU1, BLEUN, CER


# ============================================================================
# 预定义的评测指标组合（硬编码）
# ============================================================================

METRIC_PROFILES = {
    "bleu1_only": {
        "name": "BLEU-1 单指标评测",
        "description": "仅使用 BLEU-1，快速评估字符级别匹配度",
        "metrics": [BLEU1()],
    },
    
    "bleu_family": {
        "name": "BLEU 家族（BLEU-1, BLEU-2, BLEU-3, BLEU-4）",
        "description": "使用全部 BLEU-N，评估多粒度匹配",
        "metrics": [BLEU1(), BLEUN(n=2), BLEUN(n=3), BLEUN(n=4)],
    },
    
    "bleu_cer": {
        "name": "BLEU-1 + CER 组合评测",
        "description": "BLEU-1（精度）+ CER（错误率），综合评估",
        "metrics": [BLEU1(), CER()],
    },
    
    "comprehensive": {
        "name": "综合评测（所有指标）",
        "description": "BLEU-1, BLEU-4, CER 综合评测",
        "metrics": [BLEU1(), BLEUN(n=4), CER()],
    },
}


def get_metric_names(metrics: List) -> List[str]:
    """提取指标名称"""
    return [m.name for m in metrics]


def re_evaluate_with_profile(csv_path: str, profile_key: str = "bleu1_only", 
                             output_dir: str = None) -> Dict:
    """
    使用预定义的指标组合重新评测
    
    Args:
        csv_path: 原始 CSV 结果文件
        profile_key: 指标组合的 key（bleu1_only, bleu_family, bleu_cer, comprehensive）
        output_dir: 输出目录
        
    Returns:
        Dict: 重新评测的结果
    """
    
    if profile_key not in METRIC_PROFILES:
        print(f"❌ 未知的指标组合: {profile_key}")
        print(f"可用的组合: {list(METRIC_PROFILES.keys())}")
        return None
    
    profile = METRIC_PROFILES[profile_key]
    output_dir = output_dir or RESULTS_DIR
    
    print("\n" + "="*80)
    print(f"🔄 重新评测 - {profile['name']}")
    print("="*80)
    print(f"描述: {profile['description']}")
    print(f"指标: {', '.join(get_metric_names(profile['metrics']))}")
    print()
    
    # 读取 CSV
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return None
    
    print(f"📋 加载数据: {len(df)} 个样本\n")
    
    # 查找预测文本和参考文本列
    pred_col = None
    ref_col = None
    
    for col in df.columns:
        if 'predicted' in col.lower() or 'pred' in col.lower():
            pred_col = col
        if 'reference' in col.lower() or 'ref' in col.lower():
            ref_col = col
    
    if not pred_col or not ref_col:
        print(f"❌ 找不到预测文本列或参考文本列")
        print(f"列名: {list(df.columns)}")
        return None
    
    print(f"✓ 找到文本列:")
    print(f"  - 预测: {pred_col}")
    print(f"  - 参考: {ref_col}\n")
    
    # 使用每个指标重新计算
    results = {
        "num_samples": len(df),
        "metrics": {}
    }
    
    print("-"*80)
    print("计算指标...\n")
    
    for metric in profile["metrics"]:
        print(f"  计算 {metric.name}...", end=" ", flush=True)
        
        # 逐个计算分数
        scores = metric.compute_batch(df[pred_col].astype(str), 
                                     df[ref_col].astype(str))
        
        # 计算平均分（使用过滤阈值）
        if metric.name.startswith("BLEU"):
            avg_score = metric.average(scores, min_threshold=BLEU1_FILTER_THRESHOLD)
        else:
            # CER 不使用过滤
            avg_score = sum(scores) / len(scores) if scores else 0.0
        
        results["metrics"][metric.name] = {
            "scores": scores,
            "average": avg_score,
            "max": max(scores) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
        }
        
        print(f"✓ 平均: {avg_score:.4f}")
    
    print()
    print("-"*80)
    print("📊 评测结果\n")
    
    for metric_name, metric_result in results["metrics"].items():
        print(f"{metric_name}:")
        print(f"  平均分: {metric_result['average']:.4f}")
        print(f"  最高分: {metric_result['max']:.4f}")
        print(f"  最低分: {metric_result['min']:.4f}")
    
    # 保存新的 CSV（包含原始数据和新的指标列）
    output_df = df.copy()
    for metric_name, metric_result in results["metrics"].items():
        output_df[metric_name] = metric_result["scores"]
    
    output_file = Path(output_dir) / f"re_evaluated_{profile_key}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False, encoding=CSV_ENCODING)
    
    print(f"\n✅ 结果已保存: {output_file}")
    print("="*80 + "\n")
    
    return results


def main():
    """主函数 - 演示用法"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="重新评测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用 BLEU-1 评测
  python re_evaluate.py --profile bleu1_only
  
  # 使用 BLEU 家族评测
  python re_evaluate.py --profile bleu_family
  
  # 综合评测
  python re_evaluate.py --profile comprehensive
  
可用的指标组合：
""" + "\n".join([f"  - {k}: {v['name']}" for k, v in METRIC_PROFILES.items()])
    )
    
    parser.add_argument("--csv", type=str, default=RESULTS_CSV,
                       help="输入 CSV 文件路径")
    parser.add_argument("--profile", type=str, default="bleu1_only",
                       help="指标组合 (bleu1_only, bleu_family, bleu_cer, comprehensive)")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR,
                       help="输出目录")
    
    args = parser.parse_args()
    
    re_evaluate_with_profile(
        csv_path=args.csv,
        profile_key=args.profile,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
