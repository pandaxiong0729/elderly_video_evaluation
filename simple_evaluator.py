"""
评测核心逻辑（简化版）
整合模型推理和评测指标计算，一键完成评测
"""

import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class EvaluationResult:
    """单个样本的评测结果"""
    video_path: str
    reference_text: str
    predicted_text: str
    scores: Dict[str, float]
    metadata: Dict = None


@dataclass
class EvaluationReport:
    """完整评测报告"""
    model_name: str
    evaluation_time: str
    num_samples: int
    metrics_summary: Dict[str, float]
    detailed_results: List[EvaluationResult]
    config: Dict = None


class SimpleEvaluator:
    """
    简化评测器
    
    一键完成：推理 → 评测 → 输出
    """
    
    def __init__(self, model, metrics: List = None, verbose: bool = True):
        """
        初始化评测器
        
        Args:
            model: 模型适配器实例
            metrics: 评测指标列表（默认使用 BLEU-1）
            verbose: 是否输出详细信息
        """
        self.model = model
        self.metrics = metrics or []
        self.verbose = verbose
        self.results: List[EvaluationResult] = []
    
    def evaluate(self, 
                 data_loader,
                 save_dir: str = "results",
                 save_json: bool = True,
                 save_csv: bool = True,
                 save_report: bool = True) -> EvaluationReport:
        """
        一键完成评测（推理 + 评测 + 输出）
        
        Args:
            data_loader: 数据加载器
            save_dir: 结果保存目录
            save_json: 是否保存 JSON 结果
            save_csv: 是否保存 CSV 结果
            save_report: 是否保存文本报告
            
        Returns:
            EvaluationReport: 完整评测报告
        """
        from tqdm import tqdm
        
        print("\n" + "="*60)
        print(f"🎯 开始评测：{self.model.model_name}")
        print("="*60)
        
        # 不预加载全部数据，只获取数量
        num_samples = len(data_loader)
        
        print(f"\n📊 评测样本数：{num_samples}")
        print(f"📏 评测指标：{', '.join([m.name for m in self.metrics])}")
        
        # ========== 步骤 1: 模型推理（逐个加载，不预加载全部） ==========
        print("\n" + "-"*60)
        print("📍 步骤 1/2: 执行模型推理")
        print("-"*60)
        
        predictions = []
        inference_results = []
        num_samples = len(data_loader)
        
        # 逐个迭代，不预加载所有数据
        for idx in tqdm(range(num_samples), desc="推理中"):
            # 每次只获取一个样本
            sample = data_loader.samples[idx]
            video_path = sample.video_path
            
            try:
                result = self.model.infer(video_path)
                predictions.append(result.predicted_text)
                inference_results.append(result)
                
                if self.verbose:
                    print(f"\n  视频：{Path(video_path).name}")
                    print(f"  预测：{result.predicted_text[:80]}...")
                    
            except Exception as e:
                print(f"\n  ❌ 推理失败 {Path(video_path).name}: {e}")
                predictions.append("")
                inference_results.append(None)
        
        # 获取全部样本和参考文本（用于后续评测）
        samples = data_loader.get_samples()
        references = data_loader.get_references()
        
        # ========== 步骤 2: 计算指标 ==========
        print("\n" + "-"*60)
        print("📍 步骤 2/2: 计算评测指标")
        print("-"*60)
        
        # 为每个样本计算指标
        for i, sample in enumerate(samples):
            pred_text = predictions[i]
            ref_text = references[i]
            
            # 计算所有指标
            scores = {}
            for metric in self.metrics:
                score = metric.compute(pred_text, ref_text)
                scores[metric.name] = score
            
            # 创建结果
            result = EvaluationResult(
                video_path=sample.video_path,
                reference_text=ref_text,
                predicted_text=pred_text,
                scores=scores,
                metadata=sample.metadata
            )
            self.results.append(result)
        
        # 打印指标汇总
        print("\n指标汇总:")
        metrics_summary = self._compute_metrics_summary()
        for metric_name, avg_score in metrics_summary.items():
            print(f"  {metric_name}: {avg_score:.4f}")
        
        # ========== 步骤 3: 输出结果 ==========
        print("\n" + "-"*60)
        print("📍 步骤 3/3: 输出结果")
        print("-"*60)
        
        # 创建保存目录
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = EvaluationReport(
            model_name=self.model.model_name,
            evaluation_time=datetime.now().isoformat(),
            num_samples=len(samples),
            metrics_summary=metrics_summary,
            detailed_results=self.results,
            config=getattr(data_loader, '__dict__', {})
        )
        
        # 保存结果
        saved_files = []
        
        if save_json:
            json_path = save_dir / f"{self.model.model_name}_{timestamp}.json"
            self._save_json(report, json_path)
            saved_files.append(str(json_path))
            print(f"  ✓ JSON: {json_path}")
        
        if save_csv:
            csv_path = save_dir / f"{self.model.model_name}_{timestamp}.csv"
            self._save_csv(report, csv_path)
            saved_files.append(str(csv_path))
            print(f"  ✓ CSV: {csv_path}")
        
        if save_report:
            report_path = save_dir / f"{self.model.model_name}_{timestamp}_report.txt"
            self._save_report(report, report_path)
            saved_files.append(str(report_path))
            print(f"  ✓ 报告：{report_path}")
        
        print("\n" + "="*60)
        print(f"✅ 评测完成！")
        print("="*60)
        
        return report
    
    def _compute_metrics_summary(self) -> Dict[str, float]:
        """计算指标汇总（平均分）"""
        summary = {}
        for metric in self.metrics:
            scores = [result.scores.get(metric.name, 0.0) for result in self.results]
            if scores:
                summary[metric.name] = sum(scores) / len(scores)
            else:
                summary[metric.name] = 0.0
        return summary
    
    def _save_json(self, report: EvaluationReport, save_path: Path):
        """保存为 JSON"""
        data = {
            'model_name': report.model_name,
            'evaluation_time': report.evaluation_time,
            'num_samples': report.num_samples,
            'metrics_summary': report.metrics_summary,
            'detailed_results': [asdict(r) for r in report.detailed_results]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_csv(self, report: EvaluationReport, save_path: Path):
        """保存为 CSV"""
        with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # 表头
            header = ['视频路径', '参考文本', '预测文本']
            header.extend(report.metrics_summary.keys())
            writer.writerow(header)
            
            # 数据行
            for result in report.detailed_results:
                row = [
                    result.video_path,
                    result.reference_text,
                    result.predicted_text
                ]
                row.extend([result.scores.get(metric, '') for metric in report.metrics_summary.keys()])
                writer.writerow(row)
    
    def _save_report(self, report: EvaluationReport, save_path: Path):
        """保存为文本报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("老年人视频识别评测报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"模型名称：{report.model_name}\n")
            f.write(f"评测时间：{report.evaluation_time}\n")
            f.write(f"样本数量：{report.num_samples}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("指标汇总\n")
            f.write("-"*60 + "\n")
            for metric_name, score in report.metrics_summary.items():
                f.write(f"{metric_name}: {score:.4f}\n")
            f.write("\n")
            
            f.write("-"*60 + "\n")
            f.write("详细结果（前 10 个样本）\n")
            f.write("-"*60 + "\n")
            for i, result in enumerate(report.detailed_results[:10], 1):
                f.write(f"\n[{i}] {Path(result.video_path).name}\n")
                f.write(f"参考：{result.reference_text[:100]}...\n")
                f.write(f"预测：{result.predicted_text[:100]}...\n")
                f.write(f"得分：{result.scores}\n")
            
            if len(report.detailed_results) > 10:
                f.write(f"\n... 还有 {len(report.detailed_results) - 10} 个样本\n")
            
            f.write("\n" + "="*60 + "\n")
