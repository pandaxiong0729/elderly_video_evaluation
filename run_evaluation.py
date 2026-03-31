"""
老年人视频识别评测系统 - 主入口

使用方法：
    python run_evaluation.py --video_dir data/videos --ref_dir data/references

添加自己的模型：
    1. 复制 model_adapter/custom_model_example.py
    2. 实现 _load_model() 和 infer() 方法
    3. 在下方 create_model() 函数中注册
"""

import argparse
from pathlib import Path

from model_adapter.model_examples import DummyModelAdapter, QwenVLAdapter, InternVLAdapter
from data_loader.folder_loader import FolderDataLoader
from metrics.bleu_metrics import BLEU1, BLEUN, CER
from simple_evaluator import SimpleEvaluator


def create_model(model_type: str = "dummy", 
                 model_name: str = "test_model",
                 model_path: str = None,
                 **kwargs):
    """
    创建模型适配器
    
    Args:
        model_type: 模型类型
        model_name: 模型名称（用于结果记录）
        model_path: 模型路径（HuggingFace 路径或本地路径）
        **kwargs: 其他配置参数
    
    ============ 添加你的模型 ============
    
    在下方添加 elif 分支，例如：
    
    elif model_type == "qwen_omni":
        from model_adapter.my_model import OmniModelAdapter
        return OmniModelAdapter(
            model_name=model_name,
            config={
                "model_path": model_path or "Qwen/Qwen2.5-Omni-7B",
                "device": "cuda",
                "prompt": "请识别视频中老年人说的话",
                **kwargs
            }
        )
    
    =====================================
    """
    print(f"\n🔧 创建模型：{model_name} ({model_type})")
    if model_path:
        print(f"   模型路径：{model_path}")
    
    # ========== 内置模型 ==========
    if model_type == "dummy":
        return DummyModelAdapter(model_name=model_name)
    elif model_type == "qwen_vl":
        return QwenVLAdapter(model_name=model_name)
    elif model_type == "intern_vl":
        return InternVLAdapter(model_name=model_name)
    
    # ========== 在这里添加你的模型 ==========
    # 
    # elif model_type == "qwen_omni":
    #     from model_adapter.my_model import OmniModelAdapter
    #     return OmniModelAdapter(
    #         model_name=model_name,
    #         config={
    #             "model_path": model_path or "Qwen/Qwen2.5-Omni-7B",
    #             "device": "cuda",
    #             "prompt": "请识别视频中老年人说的话",
    #         }
    #     )
    #
    # =========================================
    
    else:
        print(f"\n❌ 未知的模型类型：{model_type}")
        print(f"\n📝 可用的内置类型：dummy, qwen_vl, intern_vl")
        print(f"\n💡 添加自定义模型的方法：")
        print(f"   1. 复制 model_adapter/custom_model_example.py")
        print(f"   2. 实现你的模型加载和推理逻辑")
        print(f"   3. 在 run_evaluation.py 的 create_model() 函数中添加 elif 分支")
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
                         model_path: str = None,
                         use_cer: bool = False,
                         save_dir: str = "results"):
    """
    从文件夹加载数据进行评测
    
    Args:
        video_dir: 视频文件夹路径
        ref_dir: 参考文本文件夹路径（srt/txt）
        model_type: 模型类型
        model_name: 模型名称
        model_path: 模型路径
        use_cer: 是否使用 CER 指标
        save_dir: 结果保存目录
    """
    print("\n" + "="*60)
    print("🎯 老年人视频识别评测系统")
    print("="*60)
    
    # 1. 创建模型
    model = create_model(model_type, model_name, model_path)
    
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
    parser = argparse.ArgumentParser(
        description="老年人视频识别评测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用虚拟模型测试流程
  python run_evaluation.py --video_dir data/videos --ref_dir data/references

  # 使用自定义模型
  python run_evaluation.py --model_type qwen_omni --model_path Qwen/Qwen2.5-Omni-7B

  # 使用 CER 指标
  python run_evaluation.py --model_type qwen_omni --use_cer
        """
    )
    
    # 数据路径
    parser.add_argument("--video_dir", type=str, default="data/videos",
                       help="视频文件夹路径")
    parser.add_argument("--ref_dir", type=str, default="data/references",
                       help="参考文本文件夹路径（srt/txt）")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="dummy",
                       help="模型类型（如 dummy, qwen_omni, intern_vl 或自定义）")
    parser.add_argument("--model_name", type=str, default="test_model",
                       help="模型名称（用于结果记录）")
    parser.add_argument("--model_path", type=str, default=None,
                       help="模型路径（HuggingFace 路径或本地路径）")
    
    # 评测参数
    parser.add_argument("--use_cer", action="store_true",
                       help="是否使用 CER 指标")
    parser.add_argument("--save_dir", type=str, default="results",
                       help="结果保存目录")
    
    args = parser.parse_args()
    
    # 执行评测
    evaluate_from_folder(
        video_dir=args.video_dir,
        ref_dir=args.ref_dir,
        model_type=args.model_type,
        model_name=args.model_name,
        model_path=args.model_path,
        use_cer=args.use_cer,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
