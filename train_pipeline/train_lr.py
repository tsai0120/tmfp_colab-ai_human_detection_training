"""
TF-IDF + Logistic Regression 模型訓練管線
"""

import os
import sys
import pickle
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import preprocess_dataframe


def load_data(data_path: str = "AI_Human.csv"):
    """載入資料"""
    print(f"📂 載入資料: {data_path}")
    
    try:
        df = pd.read_csv(data_path, nrows=100000)
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


def train_lr(
    data_path: str = "AI_Human.csv",
    model_dir: str = "models/tfidf_lr",
    test_size: float = 0.2,
    val_size: float = 0.1,
    C: float = 1.0,
    max_features: int = 5000,
    use_grid_search: bool = False
):
    """訓練 TF-IDF + Logistic Regression 模型"""
    print("🚀 開始訓練 TF-IDF + LR 模型...")
    
    df = load_data(data_path)
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
    
    print("🔤 建立 TF-IDF 向量化器...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✅ TF-IDF 特徵維度: {X_train_tfidf.shape[1]}")
    
    if use_grid_search:
        print("🔍 使用 GridSearch 尋找最佳參數...")
        print("⏳ 這可能需要較長時間，請耐心等待...")
        start_time = time.time()
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs']
        }
        lr = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        lr.fit(X_train_tfidf, y_train)
        elapsed_time = time.time() - start_time
        print(f"✅ 最佳參數: {lr.best_params_}")
        print(f"⏱️  訓練時間: {elapsed_time/60:.2f} 分鐘")
        model = lr.best_estimator_
    else:
        print(f"🏋️  訓練 LR (C={C})...")
        print(f"📊 訓練資料量: {len(X_train)} 筆")
        print("⏳ 訓練中，這通常需要 1-5 分鐘...")
        
        start_time = time.time()
        model = LogisticRegression(C=C, random_state=42, max_iter=1000, verbose=1)
        
        print("\n" + "="*50)
        print("開始訓練...")
        print("="*50)
        
        model.fit(X_train_tfidf, y_train)
        
        elapsed_time = time.time() - start_time
        print("="*50)
        print(f"✅ 訓練完成！")
        print(f"⏱️  訓練時間: {elapsed_time/60:.2f} 分鐘 ({elapsed_time:.2f} 秒)")
        print("="*50)
    
    print("📊 評估模型...")
    
    y_val_pred = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"✅ 驗證集準確率: {val_accuracy:.4f}")
    
    y_test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"✅ 測試集準確率: {test_accuracy:.4f}")
    
    print("💾 儲存模型...")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    metrics = {
        "model_name": "tfidf_lr",
        "baseline_accuracy": float(test_accuracy),
        "prompt_A_accuracy": float(test_accuracy),
        "prompt_B_accuracy": float(test_accuracy),
        "prompt_C_accuracy": float(test_accuracy),
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "parameters": {
            "C": float(C) if not use_grid_search else float(model.C),
            "max_features": max_features
        }
    }
    
    with open(f"{model_dir}/metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 模型已儲存至 {model_dir}")
    print(f"📊 最終測試準確率: {test_accuracy:.4f}")
    
    return model, vectorizer, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="訓練 TF-IDF + LR 模型")
    parser.add_argument("--data", type=str, default="AI_Human.csv", help="資料路徑")
    parser.add_argument("--model_dir", type=str, default="models/tfidf_lr", help="模型目錄")
    parser.add_argument("--C", type=float, default=1.0, help="LR C 參數")
    parser.add_argument("--max_features", type=int, default=5000, help="TF-IDF 最大特徵數")
    parser.add_argument("--grid_search", action="store_true", help="使用 GridSearch")
    
    args = parser.parse_args()
    
    train_lr(
        data_path=args.data,
        model_dir=args.model_dir,
        C=args.C,
        max_features=args.max_features,
        use_grid_search=args.grid_search
    )

