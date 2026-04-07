"""
评测指标实现
包含 BLEU-1、BLEU-2、BLEU-3、BLEU-4、BLEU-RT、CER 等指标
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from collections import Counter
import math
import re
from config import HALLUCINATION_PATTERNS, DIGIT_TO_CHINESE


def convert_digits_to_chinese(text: str) -> str:
    """
    将阿拉伯数字转换为对应的中文数字
    例如: "2023年" -> "二零二三年"
    
    Args:
        text: 输入文本
        
    Returns:
        str: 转换后的文本
    """
    return ''.join(DIGIT_TO_CHINESE.get(char, char) for char in text)


def clean_prediction_text(text: str) -> str:
    """
    清理预测文本，去除模型幻觉和无关内容
    
    Args:
        text: 原始预测文本
        
    Returns:
        str: 清理后的文本
    """
    # 去除空白字符
    text = text.strip().replace("\n", "").replace("\r", "").replace("\t", "")
    
    # 移除模型幻觉文本
    for pattern in HALLUCINATION_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 清理多余空格
    text = re.sub(r"\s+", "", text)  # 中文文本通常无空格
    
    # 转换阿拉伯数字为中文数字
    text = convert_digits_to_chinese(text)
    
    return text.strip()


class BaseMetric(ABC):
    """评测指标基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute(self, predicted: str, reference: str) -> float:
        """
        计算单个样本的指标
        Args:
            predicted: 模型预测文本
            reference: 参考文本（ground truth）
            
        Returns:
            float: 指标得分
        """
        pass
    
    def compute_batch(self, predictions: List[str], references: List[str]) -> List[float]:
        """
        批量计算指标
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
        Returns:
            List[float]: 指标得分列表
        """
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.compute(pred, ref)
            scores.append(score)
        return scores
    
    def average(self, scores: List[float], min_threshold: float = 0.1) -> float:
        """
        计算平均分，过滤掉低于阈值的分数
        
        Args:
            scores: 分数列表
            min_threshold: 最小阈值（低于此值的分数不计入平均）
            
        Returns:
            float: 平均分（已过滤）
        """
        if not scores:
            return 0.0
        
        # 过滤掉低分
        filtered_scores = [s for s in scores if s >= min_threshold]
        
        if not filtered_scores:
            # 如果全部低于阈值，返回原始平均分并输出警告
            return sum(scores) / len(scores)
        
        return sum(filtered_scores) / len(filtered_scores)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class BLEU1(BaseMetric):
    """BLEU-1 指标实现（unigram 精度）"""
    
    def __init__(self, tokenize_method: str = 'char'):
        """
        初始化 BLEU-1
        
        Args:
            tokenize_method: 分词方法 ('char' 或 'word')
                - 'char': 按字符分割（适合中文）
                - 'word': 按单词分割（适合英文）
        """
        super().__init__("BLEU-1")
        self.tokenize_method = tokenize_method
    
    def compute(self, predicted: str, reference: str) -> float:
        """
        计算 BLEU-1 分数
        
        BLEU-1 = unigram 精度（考虑 brevity penalty）
        
        Args:
            predicted: 预测文本
            reference: 参考文本
            
        Returns:
            float: BLEU-1 分数 (0-1)
        """
        # 分词
        pred_tokens = self._tokenize(predicted)
        ref_tokens = self._tokenize(reference)
        
        if not pred_tokens:
            return 0.0
        
        # 计算 unigram 重叠
        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)
        
        # clipped counts
        matches = sum(min(pred_counts[word], ref_counts[word]) 
                     for word in pred_counts)
        
        # unigram 精度
        precision = matches / len(pred_tokens)
        
        # brevity penalty (防止过短预测)
        bp = self._brevity_penalty(len(pred_tokens), len(ref_tokens))
        
        return bp * precision
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词
        
        对于中文：按字符分割
        对于英文：按空格分割
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        # 清理文本：去除空白字符和模型幻觉
        text = clean_prediction_text(text).lower()
        
        # 简单判断：如果包含中文字符，按字符分割
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return list(text.replace(" ", ""))
        else:
            return text.split()
    
    def _brevity_penalty(self, pred_len: int, ref_len: int) -> float:
        """
        计算 brevity penalty
        
        Args:
            pred_len: 预测文本长度
            ref_len: 参考文本长度
            
        Returns:
            float: brevity penalty
        """
        if pred_len == 0:
            return 0.0
        
        if pred_len >= ref_len:
            return 1.0
        else:
            return math.exp(1 - ref_len / pred_len)


class BLEUN(BaseMetric):
    """BLEU-N 指标实现（支持 N=1,2,3,4）"""
    
    def __init__(self, n: int = 1):
        """
        初始化 BLEU-N 指标
        
        Args:
            n: n-gram 的 N 值 (1, 2, 3, 4)
        """
        super().__init__(f"BLEU-{n}")
        self.n = n
    
    def compute(self, predicted: str, reference: str) -> float:
        """计算 BLEU-N 分数"""
        pred_tokens = self._tokenize(predicted)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) < self.n:
            return 0.0
        
        # 计算 n-gram
        pred_ngrams = self._get_ngrams(pred_tokens, self.n)
        ref_ngrams = self._get_ngrams(ref_tokens, self.n)
        
        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        # clipped counts
        matches = sum(min(pred_counts[ngram], ref_counts[ngram]) 
                     for ngram in pred_counts)
        
        # n-gram 精度
        precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
        
        # brevity penalty
        bp = self._brevity_penalty(len(pred_tokens), len(ref_tokens))
        
        return bp * precision
    
    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        # 清理文本：去除空白字符和模型幻觉
        text = clean_prediction_text(text).lower()
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return list(text.replace(" ", ""))
        else:
            return text.split()
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """获取 n-gram"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _brevity_penalty(self, pred_len: int, ref_len: int) -> float:
        """计算 brevity penalty"""
        if pred_len == 0:
            return 0.0
        if pred_len >= ref_len:
            return 1.0
        return math.exp(1 - ref_len / pred_len)


class CER(BaseMetric):
    """CER (Character Error Rate) 字符错误率"""
    
    def __init__(self):
        super().__init__("CER")
    
    def compute(self, predicted: str, reference: str) -> float:
        """
        计算 CER
        
        CER = (S + D + I) / N
        S: 替换次数，D: 删除次数，I: 插入次数，N: 参考文本字符数
        
        Args:
            predicted: 预测文本
            reference: 参考文本
            
        Returns:
            float: CER (越小越好)
        """
        # 统一预处理：去除空格、换行、模型幻觉等
        pred_clean = clean_prediction_text(predicted).replace(" ", "")
        ref_clean = clean_prediction_text(reference).replace(" ", "")
        
        pred_chars = list(pred_clean)
        ref_chars = list(ref_clean)
        
        # 使用动态规划计算编辑距离
        edits = self._levenshtein_distance(pred_chars, ref_chars)
        
        if len(ref_chars) == 0:
            return 0.0 if len(pred_chars) == 0 else 1.0
        
        return edits / len(ref_chars)
    
    def _levenshtein_distance(self, s1: List[str], s2: List[str]) -> int:
        """计算 Levenshtein 编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
