"""
模型載入工具
統一管理所有模型的載入與預測
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ModelLoader:
    """模型載入器"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.vectorizers = {}
        self.loaded_models = {}
    
    def load_tfidf_svm(self) -> bool:
        """載入 TF-IDF + SVM 模型"""
        try:
            model_path = self.models_dir / "tfidf_svm" / "model.pkl"
            vectorizer_path = self.models_dir / "tfidf_svm" / "vectorizer.pkl"
            
            if not model_path.exists() or not vectorizer_path.exists():
                print(f"⚠️  模型檔案不存在: {model_path}")
                return False
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizers['tfidf'] = pickle.load(f)
            
            with open(model_path, 'rb') as f:
                self.loaded_models['tfidf_svm'] = pickle.load(f)
            
            print("✅ TF-IDF + SVM 模型載入成功")
            return True
        except Exception as e:
            print(f"❌ TF-IDF + SVM 載入失敗: {e}")
            return False
    
    def load_tfidf_lr(self) -> bool:
        """載入 TF-IDF + Logistic Regression 模型"""
        try:
            model_path = self.models_dir / "tfidf_lr" / "model.pkl"
            vectorizer_path = self.models_dir / "tfidf_lr" / "vectorizer.pkl"
            
            if not model_path.exists() or not vectorizer_path.exists():
                print(f"⚠️  模型檔案不存在: {model_path}")
                return False
            
            if 'tfidf' not in self.vectorizers:
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizers['tfidf'] = pickle.load(f)
            
            with open(model_path, 'rb') as f:
                self.loaded_models['tfidf_lr'] = pickle.load(f)
            
            print("✅ TF-IDF + LR 模型載入成功")
            return True
        except Exception as e:
            print(f"❌ TF-IDF + LR 載入失敗: {e}")
            return False
    
    def load_bert(self) -> bool:
        """載入 BERT 模型"""
        try:
            model_path = self.models_dir / "bert"
            
            if not model_path.exists():
                print(f"⚠️  模型目錄不存在: {model_path}")
                return False
            
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            model.eval()
            
            self.loaded_models['bert'] = {
                'model': model,
                'tokenizer': tokenizer
            }
            
            print("✅ BERT 模型載入成功")
            return True
        except Exception as e:
            print(f"❌ BERT 載入失敗: {e}")
            return False
    
    def load_roberta_lora(self) -> bool:
        """載入 RoBERTa + LoRA 模型"""
        try:
            model_path = self.models_dir / "roberta_lora"
            
            if not model_path.exists():
                print(f"⚠️  模型目錄不存在: {model_path}")
                return False
            
            # 如果有使用 PEFT，需要特殊處理
            try:
                from peft import PeftModel
                from transformers import AutoModelForSequenceClassification
                
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    "roberta-base",
                    num_labels=2
                )
                model = PeftModel.from_pretrained(base_model, str(model_path))
                tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                model.eval()
                
                self.loaded_models['roberta_lora'] = {
                    'model': model,
                    'tokenizer': tokenizer
                }
            except:
                # 如果沒有 PEFT，嘗試直接載入
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                model.eval()
                
                self.loaded_models['roberta_lora'] = {
                    'model': model,
                    'tokenizer': tokenizer
                }
            
            print("✅ RoBERTa + LoRA 模型載入成功")
            return True
        except Exception as e:
            print(f"❌ RoBERTa + LoRA 載入失敗: {e}")
            return False
    
    def load_hybrid(self) -> bool:
        """載入 Hybrid 模型"""
        try:
            model_path = self.models_dir / "hybrid" / "model.pkl"
            
            if not model_path.exists():
                print(f"⚠️  模型檔案不存在: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.loaded_models['hybrid'] = pickle.load(f)
            
            print("✅ Hybrid 模型載入成功")
            return True
        except Exception as e:
            print(f"❌ Hybrid 載入失敗: {e}")
            return False
    
    def predict_tfidf_svm(self, text: str) -> float:
        """使用 TF-IDF + SVM 預測"""
        if 'tfidf_svm' not in self.loaded_models:
            return 0.5
        
        vectorizer = self.vectorizers.get('tfidf')
        model = self.loaded_models['tfidf_svm']
        
        X = vectorizer.transform([text])
        prob = model.predict_proba(X)[0][1]  # AI 機率
        return float(prob)
    
    def predict_tfidf_lr(self, text: str) -> float:
        """使用 TF-IDF + LR 預測"""
        if 'tfidf_lr' not in self.loaded_models:
            return 0.5
        
        vectorizer = self.vectorizers.get('tfidf')
        model = self.loaded_models['tfidf_lr']
        
        X = vectorizer.transform([text])
        prob = model.predict_proba(X)[0][1]  # AI 機率
        return float(prob)
    
    def predict_bert(self, text: str, max_length: int = 512) -> float:
        """使用 BERT 預測"""
        if 'bert' not in self.loaded_models:
            return 0.5
        
        model_info = self.loaded_models['bert']
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            ai_prob = float(probs[0][1].item())
        
        return ai_prob
    
    def predict_roberta_lora(self, text: str, max_length: int = 512) -> float:
        """使用 RoBERTa + LoRA 預測"""
        if 'roberta_lora' not in self.loaded_models:
            return 0.5
        
        model_info = self.loaded_models['roberta_lora']
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            ai_prob = float(probs[0][1].item())
        
        return ai_prob
    
    def predict_hybrid(self, text: str, linguistic_features: Optional[Dict] = None) -> float:
        """使用 Hybrid 模型預測"""
        if 'hybrid' not in self.loaded_models:
            return 0.5
        
        # Hybrid 模型需要結合多個模型的預測和語言特徵
        # 這裡假設 hybrid 模型已經處理好所有輸入
        # 實際實作可能需要根據你的 hybrid 模型架構調整
        
        model = self.loaded_models['hybrid']
        
        # 如果模型是 sklearn 模型，需要準備特徵
        if hasattr(model, 'predict_proba'):
            # 這裡需要根據實際的 hybrid 模型架構來準備特徵
            # 暫時返回 0.5
            return 0.5
        
        return 0.5
    
    def predict_all(self, text: str) -> Dict[str, float]:
        """使用所有模型預測"""
        results = {
            'svm': self.predict_tfidf_svm(text),
            'lr': self.predict_tfidf_lr(text),
            'bert': self.predict_bert(text),
            'lora': self.predict_roberta_lora(text),
            'hybrid': self.predict_hybrid(text)
        }
        return results
    
    def load_all_models(self) -> Dict[str, bool]:
        """載入所有模型"""
        results = {
            'tfidf_svm': self.load_tfidf_svm(),
            'tfidf_lr': self.load_tfidf_lr(),
            'bert': self.load_bert(),
            'roberta_lora': self.load_roberta_lora(),
            'hybrid': self.load_hybrid()
        }
        return results


if __name__ == "__main__":
    loader = ModelLoader()
    results = loader.load_all_models()
    print("\n模型載入結果:")
    for model_name, success in results.items():
        print(f"  {model_name}: {'✅' if success else '❌'}")

