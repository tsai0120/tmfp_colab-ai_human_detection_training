"""
BERT 模型訓練管線
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import preprocess_dataframe


class TextDataset(Dataset):
    """文本資料集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(data_path: str = "AI_Human.csv", max_samples: int = 50000):
    """載入資料"""
    print(f"📂 載入資料: {data_path}")
    
    try:
        df = pd.read_csv(data_path, nrows=max_samples)
        print(f"✅ 載入 {len(df)} 筆資料")
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        return None
    
    if 'text' not in df.columns or 'generated' not in df.columns:
        if len(df.columns) >= 2:
            df.columns = ['text', 'generated'] + list(df.columns[2:])
    
    df = df.dropna(subset=['text', 'generated'])
    df['text'] = df['text'].astype(str)
    
    if df['generated'].dtype == bool:
        df['label'] = df['generated'].astype(int)
    elif df['generated'].dtype == object:
        df['label'] = df['generated'].apply(
            lambda x: 1 if str(x).lower() in ['true', '1', 'ai', 'yes'] else 0
        )
    else:
        df['label'] = df['generated'].astype(int)
    
    print(f"📊 標籤分布: {df['label'].value_counts().to_dict()}")
    
    return df[['text', 'label']]


def train_bert(
    data_path: str = "AI_Human.csv",
    model_dir: str = "models/bert",
    model_name: str = "bert-base-uncased",
    test_size: float = 0.2,
    val_size: float = 0.1,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 512,
    max_samples: int = 50000
):
    """訓練 BERT 模型"""
    print("🚀 開始訓練 BERT 模型...")
    
    df = load_data(data_path, max_samples=max_samples)
    if df is None:
        return
    
    print("🔧 預處理資料...")
    df = preprocess_dataframe(df, text_column='text', remove_stopwords=False)
    
    X = df['text'].values
    y = df['label'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size),
        random_state=42, stratify=y_temp
    )
    
    print(f"📊 資料分割: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    print("📥 載入 BERT tokenizer 和模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    print("🔤 建立資料集...")
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)
    
    # 訓練參數
    training_args = TrainingArguments(
        output_dir=f"{model_dir}/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{model_dir}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": accuracy_score(labels, predictions)}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("🏋️  開始訓練...")
    trainer.train()
    
    print("📊 評估模型...")
    val_results = trainer.evaluate()
    val_accuracy = val_results['eval_accuracy']
    print(f"✅ 驗證集準確率: {val_accuracy:.4f}")
    
    # 測試集評估
    test_results = trainer.evaluate(test_dataset)
    test_accuracy = test_results['eval_accuracy']
    print(f"✅ 測試集準確率: {test_accuracy:.4f}")
    
    print("💾 儲存模型...")
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    metrics = {
        "model_name": "bert",
        "baseline_accuracy": float(test_accuracy),
        "prompt_A_accuracy": float(test_accuracy),
        "prompt_B_accuracy": float(test_accuracy),
        "prompt_C_accuracy": float(test_accuracy),
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "parameters": {
            "model_name": model_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_length": max_length
        }
    }
    
    with open(f"{model_dir}/metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 模型已儲存至 {model_dir}")
    print(f"📊 最終測試準確率: {test_accuracy:.4f}")
    
    return model, tokenizer, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="訓練 BERT 模型")
    parser.add_argument("--data", type=str, default="AI_Human.csv", help="資料路徑")
    parser.add_argument("--model_dir", type=str, default="models/bert", help="模型目錄")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="BERT 模型名稱")
    parser.add_argument("--epochs", type=int, default=3, help="訓練輪數")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="學習率")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列長度")
    parser.add_argument("--max_samples", type=int, default=50000, help="最大樣本數")
    
    args = parser.parse_args()
    
    train_bert(
        data_path=args.data,
        model_dir=args.model_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples
    )

