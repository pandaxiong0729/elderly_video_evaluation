"""
老年人视频识别评测系统 - 简化版主入口

使用方法 1：使用文件夹（推荐）
    python run_evaluation.py --mode folder --video_dir data/videos --ref_dir data/references

使用方法 2：使用配置文件
    python run_evaluation.py --mode config --config config.json

一键完成：模型推理 + 指标计算 + 结果输出
"""

import argparse
from pathlib import Path

from model_adapter.model_examples import DummyModelAdapter, QwenVLAdapter, InternVLAdapter
from data_loader.folder_loader import FolderDataLoader
from metrics.bleu_metrics import BLEU1, BLEUN, CER
from simple_evaluator import SimpleEvaluator


def create_model(model_type: str = "dummy", model_name: str = "test_model"):
    """创建模型适配器"""
    print(f"\n🔧 创建模型：{model_name} ({model_type})")
    
    if model_type == "dummy":
        return DummyModelAdapter(model_name=model_name)
    elif model_type == "qwen_vl":
        return QwenVLAdapter(model_name=model_name)
    elif model_type == "intern_vl":
        return InternVLAdapter(model_name=model_name)
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")


def create_metrics(use_bleu1=True, use_cer=False, use_bleu4=False):
    """创建评测指标"""
    metrics = []
    
    if use_bleu1:
        metrics.append(BLEU1())
        print(f"  ✓ BLEU-1")
    
    if use_bleu4:
        metrics.append(BLEUN(n=4))
        print(f"  ✓ BLEU-4")
    
    if use_cer:
        metrics.append(CER())
        print(f"  ✓ CER")
    
    return metrics


def evaluate_from_folder(video_dir: str, 
                         ref_dir: str,
                         model_type: str = "dummy",
                         model_name: str = "test_model",
                         use_cer: bool = False,
                         save_dir: str = "results"):
    """
    从文件夹加载数据进行评测
    
    Args:
        video_dir: 视频文件夹路径
        ref_dir: 参考文本文件夹路径（srt/txt）
        model_type: 模型类型
        model_name: 模型名称
        use_cer: 是否使用 CER 指标
        save_dir: 结果保存目录
    """
    print("\n" + "="*60)
    print("🎯 老年人视频识别评测系统")
    print("="*60)
    
    # 1. 创建模型
    model = create_model(model_type, model_name)
    
    # 2. 创建数据加载器
    print(f"\n📂 加载数据...")
    data_loader = FolderDataLoader(
        video_dir=video_dir,
        reference_dir=ref_dir
    )
    
    if len(data_loader) == 0:
        print("\n❌ 错误：没有加载到任何数据！")
        return
    
    # 3. 创建评测指标
    print(f"\n📏 创建评测指标...")
    metrics = create_metrics(use_bleu1=True, use_cer=use_cer)
    
    # 4. 创建评测器并执行评测
    evaluator = SimpleEvaluator(model=model, metrics=metrics, verbose=True)
    
    report = evaluator.evaluate(
        data_loader=data_loader,
        save_dir=save_dir,
        save_json=True,
        save_csv=True,
        save_report=True
    )
    
    # 5. 打印最终汇总
    print("\n" + "="*60)
    print("📊 最终结果")
    print("="*60)
    print(f"模型：{report.model_name}")
    print(f"样本数：{report.num_samples}")
    print(f"\n指标汇总:")
    for metric_name, score in report.metrics_summary.items():
        print(f"  {metric_name}: {score:.4f}")
    print("="*60 + "\n")
    
    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="老年人视频识别评测系统")
    
    # 运行模式
    parser.add_argument("--mode", type=str, default="folder", 
                       choices=["folder", "config"],
                       help="运行模式：folder(文件夹) 或 config(配置文件)")
    
    # 文件夹模式参数
    parser.add_argument("--video_dir", type=str, default="data/videos",
                       help="视频文件夹路径")
    parser.add_argument("--ref_dir", type=str, default="data/references",
                       help="参考文本文件夹路径（srt/txt）")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="dummy",
                       choices=["dummy", "qwen_vl", "intern_vl"],
                       help="模型类型")
    parser.add_argument("--model_name", type=str, default="test_model",
                       help="模型名称")
    
    # 评测参数
    parser.add_argument("--use_cer", action="store_true",
                       help="是否使用 CER 指标")
    parser.add_argument("--save_dir", type=str, default="results",
                       help="结果保存目录")
    
    args = parser.parse_args()
    
    # 执行评测
    if args.mode == "folder":
        evaluate_from_folder(
            video_dir=args.video_dir,
            ref_dir=args.ref_dir,
            model_type=args.model_type,
            model_name=args.model_name,
            use_cer=args.use_cer,
            save_dir=args.save_dir
        )
    else:
        print("❌ 配置文件模式暂未实现，请使用文件夹模式")
        print("\n使用示例:")
        print("  python run_evaluation.py --mode folder --video_dir data/videos --ref_dir data/references")


if __name__ == "__main__":
    main()
