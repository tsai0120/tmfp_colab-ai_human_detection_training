"""
TF-IDF + SVM 模型訓練管線
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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("💡 提示: 安裝 tqdm 可顯示進度條 (pip install tqdm)")

# 加入 utils 路徑
sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import preprocess_dataframe


def load_data(data_path: str = "AI_Human.csv"):
    """載入資料"""
    print(f"📂 載入資料: {data_path}")
    
    # 嘗試讀取 CSV
    try:
        df = pd.read_csv(data_path, nrows=100000)  # 限制讀取數量以避免記憶體問題
        print(f"✅ 載入 {len(df)} 筆資料")
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        return None
    
    # 檢查欄位
    if 'text' not in df.columns or 'generated' not in df.columns:
        print("⚠️  資料格式不符合預期，嘗試自動調整...")
        # 嘗試找到正確的欄位
        if len(df.columns) >= 2:
            df.columns = ['text', 'generated'] + list(df.columns[2:])
    
    # 清理資料
    df = df.dropna(subset=['text', 'generated'])
    df['text'] = df['text'].astype(str)
    
    # 轉換標籤：generated 可能是 0/1, True/False, 或 'AI'/'Human'
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


def train_svm(
    data_path: str = "AI_Human.csv",
    model_dir: str = "models/tfidf_svm",
    test_size: float = 0.2,
    val_size: float = 0.1,
    C: float = 1.0,
    kernel: str = 'rbf',
    max_features: int = 5000,
    use_grid_search: bool = False
):
    """
    訓練 TF-IDF + SVM 模型
    
    Args:
        data_path: 資料路徑
        model_dir: 模型儲存目錄
        test_size: 測試集比例
        val_size: 驗證集比例
        C: SVM 正則化參數
        kernel: SVM kernel
        max_features: TF-IDF 最大特徵數
        use_grid_search: 是否使用 GridSearch
    """
    print("🚀 開始訓練 TF-IDF + SVM 模型...")
    
    # 載入資料
    df = load_data(data_path)
    if df is None:
        return
    
    # 預處理
    print("🔧 預處理資料...")
    df = preprocess_dataframe(df, text_column='text', remove_stopwords=False)
    
    # 分割資料
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
    
    # 建立 TF-IDF 向量化器
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
    
    # 訓練模型
    if use_grid_search:
        print("🔍 使用 GridSearch 尋找最佳參數...")
        print("⏳ 這可能需要較長時間，請耐心等待...")
        start_time = time.time()
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly']
        }
        svm = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        svm.fit(X_train_tfidf, y_train)
        elapsed_time = time.time() - start_time
        print(f"✅ 最佳參數: {svm.best_params_}")
        print(f"⏱️  訓練時間: {elapsed_time/60:.2f} 分鐘")
        model = svm.best_estimator_
    else:
        print(f"🏋️  訓練 SVM (C={C}, kernel={kernel})...")
        print(f"📊 訓練資料量: {len(X_train)} 筆")
        print("⏳ 訓練中，這可能需要 5-30 分鐘（取決於資料量和 kernel）...")
        print("💡 提示: RBF kernel 較慢但通常效果較好，Linear kernel 較快")
        
        start_time = time.time()
        model = SVC(C=C, kernel=kernel, probability=True, random_state=42, verbose=True)
        
        # 顯示進度
        print("\n" + "="*50)
        print("開始訓練...")
        print("="*50)
        
        model.fit(X_train_tfidf, y_train)
        
        elapsed_time = time.time() - start_time
        print("="*50)
        print(f"✅ 訓練完成！")
        print(f"⏱️  訓練時間: {elapsed_time/60:.2f} 分鐘 ({elapsed_time:.2f} 秒)")
        print("="*50)
    
    # 評估
    print("📊 評估模型...")
    
    # 驗證集
    y_val_pred = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"✅ 驗證集準確率: {val_accuracy:.4f}")
    
    # 測試集
    y_test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"✅ 測試集準確率: {test_accuracy:.4f}")
    
    # 如果有 prompt_type 資訊，計算各 prompt 的準確率
    prompt_accuracies = {}
    baseline_accuracy = test_accuracy
    
    # 儲存模型
    print("💾 儲存模型...")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # 儲存 metrics
    metrics = {
        "model_name": "tfidf_svm",
        "baseline_accuracy": float(baseline_accuracy),
        "prompt_A_accuracy": float(baseline_accuracy),  # 預設值，可後續更新
        "prompt_B_accuracy": float(baseline_accuracy),
        "prompt_C_accuracy": float(baseline_accuracy),
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "parameters": {
            "C": float(C) if not use_grid_search else float(model.C),
            "kernel": kernel if not use_grid_search else model.kernel,
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
    
    parser = argparse.ArgumentParser(description="訓練 TF-IDF + SVM 模型")
    parser.add_argument("--data", type=str, default="AI_Human.csv", help="資料路徑")
    parser.add_argument("--model_dir", type=str, default="models/tfidf_svm", help="模型目錄")
    parser.add_argument("--C", type=float, default=1.0, help="SVM C 參數")
    parser.add_argument("--kernel", type=str, default="rbf", choices=['linear', 'rbf', 'poly'], help="SVM kernel")
    parser.add_argument("--max_features", type=int, default=5000, help="TF-IDF 最大特徵數")
    parser.add_argument("--grid_search", action="store_true", help="使用 GridSearch")
    
    args = parser.parse_args()
    
    train_svm(
        data_path=args.data,
        model_dir=args.model_dir,
        C=args.C,
        kernel=args.kernel,
        max_features=args.max_features,
        use_grid_search=args.grid_search
    )

