"""
語言特徵提取模組
計算各種語言學特徵用於 AI vs Human 文本偵測
"""

import re
import math
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, RobertaForMaskedLM, RobertaTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers 未安裝，將跳過 perplexity 計算")


class LinguisticFeatureExtractor:
    """語言特徵提取器"""
    
    def __init__(self, enable_perplexity: bool = False):
        """
        Args:
            enable_perplexity: 是否啟用 perplexity 計算（需要 transformers）
        """
        self.enable_perplexity = enable_perplexity and TRANSFORMERS_AVAILABLE
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        self.roberta_model = None
        self.roberta_tokenizer = None
        
        if self.enable_perplexity:
            self._load_perplexity_models()
    
    def _load_perplexity_models(self):
        """載入 perplexity 計算模型"""
        try:
            print("📥 載入 GPT-2 模型...")
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_model.eval()
            
            print("📥 載入 RoBERTa 模型...")
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base')
            self.roberta_model.eval()
        except Exception as e:
            print(f"⚠️  無法載入 perplexity 模型: {e}")
            self.enable_perplexity = False
    
    def type_token_ratio(self, text: str) -> float:
        """
        Type-Token Ratio (TTR)
        詞彙多樣性指標
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        try:
            tokens = word_tokenize(text.lower())
            if len(tokens) == 0:
                return 0.0
            unique_tokens = len(set(tokens))
            return unique_tokens / len(tokens)
        except:
            return 0.0
    
    def mean_sentence_length(self, text: str) -> float:
        """平均句子長度（以詞數計算）"""
        if not text or len(text.strip()) == 0:
            return 0.0
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) == 0:
                return 0.0
            
            lengths = []
            for sent in sentences:
                tokens = word_tokenize(sent)
                lengths.append(len(tokens))
            
            return np.mean(lengths) if lengths else 0.0
        except:
            return 0.0
    
    def burstiness(self, text: str) -> float:
        """
        Burstiness: 衡量文本中詞彙出現的集中程度
        計算方式：標準差 / 平均數（對句子長度）
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return 0.0
            
            lengths = []
            for sent in sentences:
                tokens = word_tokenize(sent)
                lengths.append(len(tokens))
            
            if np.mean(lengths) == 0:
                return 0.0
            
            return np.std(lengths) / np.mean(lengths) if lengths else 0.0
        except:
            return 0.0
    
    def punctuation_ratio(self, text: str) -> float:
        """標點符號比例"""
        if not text or len(text.strip()) == 0:
            return 0.0
        
        punctuation_chars = set(string.punctuation + '，。！？；：')
        total_chars = len(text)
        punct_chars = sum(1 for char in text if char in punctuation_chars)
        
        return punct_chars / total_chars if total_chars > 0 else 0.0
    
    def character_entropy(self, text: str) -> float:
        """
        字元層級的熵（Entropy）
        衡量文本的隨機性/複雜度
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        entropy = 0.0
        for count in char_counts.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def gpt2_perplexity(self, text: str) -> Optional[float]:
        """使用 GPT-2 計算 perplexity"""
        if not self.enable_perplexity or not self.gpt2_model:
            return None
        
        try:
            inputs = self.gpt2_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.gpt2_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = math.exp(loss.item())
            return perplexity
        except Exception as e:
            print(f"⚠️  GPT-2 perplexity 計算失敗: {e}")
            return None
    
    def roberta_perplexity(self, text: str) -> Optional[float]:
        """使用 RoBERTa 計算 perplexity"""
        if not self.enable_perplexity or not self.roberta_model:
            return None
        
        try:
            inputs = self.roberta_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.roberta_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = math.exp(loss.item())
            return perplexity
        except Exception as e:
            print(f"⚠️  RoBERTa perplexity 計算失敗: {e}")
            return None
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """提取所有特徵"""
        features = {
            "type_token_ratio": self.type_token_ratio(text),
            "mean_sentence_length": self.mean_sentence_length(text),
            "burstiness": self.burstiness(text),
            "punctuation_ratio": self.punctuation_ratio(text),
            "character_entropy": self.character_entropy(text)
        }
        
        if self.enable_perplexity:
            features["gpt2_perplexity"] = self.gpt2_perplexity(text)
            features["roberta_perplexity"] = self.roberta_perplexity(text)
        
        return features
    
    def extract_features_batch(self, texts: List[str], show_progress: bool = True) -> pd.DataFrame:
        """批量提取特徵"""
        results = []
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 100 == 0:
                print(f"📊 處理進度: {i + 1}/{len(texts)}")
            
            features = self.extract_all_features(str(text))
            results.append(features)
        
        return pd.DataFrame(results)


# 檢查 torch 是否可用（用於 perplexity）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    if TRANSFORMERS_AVAILABLE:
        print("⚠️  torch 未安裝，perplexity 功能將無法使用")


if __name__ == "__main__":
    # 測試
    extractor = LinguisticFeatureExtractor(enable_perplexity=False)
    
    test_text = """
    This is a sample text for testing linguistic features.
    It contains multiple sentences. Some are longer than others.
    We want to see how well our feature extraction works.
    """
    
    features = extractor.extract_all_features(test_text)
    print("提取的特徵:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

