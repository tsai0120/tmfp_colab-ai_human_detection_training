"""
Hybrid 模型訓練管線
結合多個模型的預測結果和語言特徵
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import preprocess_dataframe
from utils.linguistic_features import LinguisticFeatureExtractor


def load_data(data_path: str = "AI_Human.csv", max_samples: int = 10000):
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


def extract_base_model_predictions(
    texts,
    svm_model_path: str = "models/tfidf_svm/model.pkl",
    svm_vectorizer_path: str = "models/tfidf_svm/vectorizer.pkl",
    lr_model_path: str = "models/tfidf_lr/model.pkl",
    lr_vectorizer_path: str = "models/tfidf_lr/vectorizer.pkl"
):
    """提取基礎模型的預測機率"""
    predictions = {
        'svm': [],
        'lr': []
    }
    
    # 載入 SVM
    try:
        with open(svm_vectorizer_path, 'rb') as f:
            svm_vectorizer = pickle.load(f)
        with open(svm_model_path, 'rb') as f:
            svm_model = pickle.load(f)
        
        X_svm = svm_vectorizer.transform(texts)
        svm_probs = svm_model.predict_proba(X_svm)[:, 1]
        predictions['svm'] = svm_probs
        print("✅ SVM 預測完成")
    except Exception as e:
        print(f"⚠️  SVM 載入失敗: {e}")
        predictions['svm'] = [0.5] * len(texts)
    
    # 載入 LR
    try:
        with open(lr_vectorizer_path, 'rb') as f:
            lr_vectorizer = pickle.load(f)
        with open(lr_model_path, 'rb') as f:
            lr_model = pickle.load(f)
        
        X_lr = lr_vectorizer.transform(texts)
        lr_probs = lr_model.predict_proba(X_lr)[:, 1]
        predictions['lr'] = lr_probs
        print("✅ LR 預測完成")
    except Exception as e:
        print(f"⚠️  LR 載入失敗: {e}")
        predictions['lr'] = [0.5] * len(texts)
    
    return predictions


def train_hybrid(
    data_path: str = "AI_Human.csv",
    model_dir: str = "models/hybrid",
    test_size: float = 0.2,
    val_size: float = 0.1,
    hidden_layer_sizes: tuple = (128, 64),
    max_samples: int = 10000,
    use_base_models: bool = True
):
    """訓練 Hybrid 模型"""
    print("🚀 開始訓練 Hybrid 模型...")
    
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
    
    # 提取語言特徵
    print("📊 提取語言特徵...")
    feature_extractor = LinguisticFeatureExtractor(enable_perplexity=False)
    
    train_features = feature_extractor.extract_features_batch(
        X_train.tolist(), show_progress=True
    )
    val_features = feature_extractor.extract_features_batch(
        X_val.tolist(), show_progress=False
    )
    test_features = feature_extractor.extract_features_batch(
        X_test.tolist(), show_progress=False
    )
    
    print(f"✅ 語言特徵維度: {train_features.shape[1]}")
    
    # 提取基礎模型預測（如果可用）
    if use_base_models:
        print("🔮 提取基礎模型預測...")
        train_preds = extract_base_model_predictions(X_train.tolist())
        val_preds = extract_base_model_predictions(X_val.tolist())
        test_preds = extract_base_model_predictions(X_test.tolist())
        
        # 合併特徵
        train_features['svm_prob'] = train_preds['svm']
        train_features['lr_prob'] = train_preds['lr']
        
        val_features['svm_prob'] = val_preds['svm']
        val_features['lr_prob'] = val_preds['lr']
        
        test_features['svm_prob'] = test_preds['svm']
        test_features['lr_prob'] = test_preds['lr']
    
    # 標準化特徵
    print("🔧 標準化特徵...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_val_scaled = scaler.transform(val_features)
    X_test_scaled = scaler.transform(test_features)
    
    # 訓練 MLP
    print(f"🏋️  訓練 MLP (hidden_layers={hidden_layer_sizes})...")
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    # 評估
    print("📊 評估模型...")
    
    y_val_pred = mlp.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"✅ 驗證集準確率: {val_accuracy:.4f}")
    
    y_test_pred = mlp.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"✅ 測試集準確率: {test_accuracy:.4f}")
    
    # 儲存模型
    print("💾 儲存模型...")
    os.makedirs(model_dir, exist_ok=True)
    
    hybrid_model = {
        'mlp': mlp,
        'scaler': scaler,
        'feature_extractor': feature_extractor,
        'use_base_models': use_base_models
    }
    
    with open(f"{model_dir}/model.pkl", 'wb') as f:
        pickle.dump(hybrid_model, f)
    
    metrics = {
        "model_name": "hybrid",
        "baseline_accuracy": float(test_accuracy),
        "prompt_A_accuracy": float(test_accuracy),
        "prompt_B_accuracy": float(test_accuracy),
        "prompt_C_accuracy": float(test_accuracy),
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "parameters": {
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "use_base_models": use_base_models,
            "feature_count": train_features.shape[1]
        }
    }
    
    with open(f"{model_dir}/metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 模型已儲存至 {model_dir}")
    print(f"📊 最終測試準確率: {test_accuracy:.4f}")
    
    return hybrid_model, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="訓練 Hybrid 模型")
    parser.add_argument("--data", type=str, default="AI_Human.csv", help="資料路徑")
    parser.add_argument("--model_dir", type=str, default="models/hybrid", help="模型目錄")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[128, 64], help="隱藏層大小")
    parser.add_argument("--max_samples", type=int, default=10000, help="最大樣本數")
    parser.add_argument("--no_base_models", action="store_true", help="不使用基礎模型預測")
    
    args = parser.parse_args()
    
    hidden_layers = tuple(args.hidden_layers) if len(args.hidden_layers) > 1 else (args.hidden_layers[0],)
    
    train_hybrid(
        data_path=args.data,
        model_dir=args.model_dir,
        hidden_layer_sizes=hidden_layers,
        max_samples=args.max_samples,
        use_base_models=not args.no_base_models
    )

