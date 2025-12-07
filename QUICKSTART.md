# 🚀 快速開始指南

## 步驟 1: 安裝依賴

```bash
# Python 依賴
pip install -r requirements.txt

# 前端依賴
cd frontend
npm install
cd ..
```

## 步驟 2: 準備資料

您的 `AI_Human.csv` 已經在專案根目錄，可以直接使用。

**（可選）收集人類文本：**
- 參考 `人類文本收集說明.md`
- 建議收集 100-200 篇人類文本
- 合併到 `AI_Human.csv` 或儲存在 `data/human_texts.csv`

## 步驟 3: 訓練模型

### 快速訓練（使用預設參數）

```bash
# 1. TF-IDF + SVM
python train_pipeline/train_svm.py --data AI_Human.csv

# 2. TF-IDF + LR
python train_pipeline/train_lr.py --data AI_Human.csv

# 3. BERT（需要 GPU，較慢）
python train_pipeline/train_bert.py --data AI_Human.csv --epochs 2 --max_samples 20000

# 4. RoBERTa + LoRA（需要 GPU）
python train_pipeline/train_lora.py --data AI_Human.csv --epochs 2 --max_samples 20000

# 5. Hybrid（需要先訓練 SVM 和 LR）
python train_pipeline/train_hybrid.py --data AI_Human.csv --max_samples 10000
```

### 自訂參數訓練

```bash
# SVM 範例
python train_pipeline/train_svm.py \
    --data AI_Human.csv \
    --C 10.0 \
    --kernel rbf \
    --max_features 10000

# BERT 範例
python train_pipeline/train_bert.py \
    --data AI_Human.csv \
    --epochs 3 \
    --learning_rate 2e-5 \
    --batch_size 16 \
    --max_samples 50000
```

## 步驟 4: 啟動服務

### 方法 1: 使用啟動腳本（推薦）

```bash
chmod +x start_services.sh
./start_services.sh
```

### 方法 2: 手動啟動

**終端 1 - 推論 API：**
```bash
python api/predict.py
```

**終端 2 - 訓練 API：**
```bash
python api/train.py
```

**終端 3 - 前端：**
```bash
cd frontend
npm run dev
```

## 步驟 5: 使用系統

1. **開啟瀏覽器**：http://localhost:3000

2. **登入**：
   - Admin: `admin` / `admin123` → 可進入模型管理
   - User: `user` / `user123` → 只能使用推論功能

3. **推論頁面**：
   - 貼上文本
   - 點擊「開始偵測」
   - 查看五個模型的預測結果

4. **模型管理（Admin）**：
   - 查看模型效能
   - 調整訓練參數
   - 重新訓練模型

## 📊 測試 API

### 推論 API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "這是一段測試文本..."}'
```

### 訓練 API（需要 Admin token）

```bash
curl -X POST http://localhost:8001/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "svm",
    "parameters": {"C": 10.0, "kernel": "rbf"},
    "admin_token": "admin_secret_token_12345"
  }'
```

## 🔧 常見問題

### Q: 訓練時記憶體不足？
A: 減少 `--max_samples` 參數，例如：
```bash
python train_pipeline/train_svm.py --data AI_Human.csv --max_samples 50000
```

### Q: BERT/LoRA 訓練很慢？
A: 
- 使用 GPU（CUDA）
- 減少 `--epochs` 和 `--max_samples`
- 增加 `--batch_size`（如果記憶體足夠）

### Q: 前端無法連接到 API？
A: 檢查：
- API 是否正在運行（Port 8000, 8001）
- `frontend/next.config.js` 中的 API_URL 設定
- CORS 設定是否正確

### Q: 模型載入失敗？
A: 確保：
- 模型已訓練並儲存在 `models/` 目錄
- 模型檔案完整（.pkl, .json 等）

## 📝 下一步

1. ✅ 訓練所有模型
2. ✅ 測試推論功能
3. ✅ 收集更多人類文本優化模型
4. ✅ 調整超參數提升效能
5. ✅ 分析不同 Prompt 類型的表現

## 🎯 效能優化建議

1. **資料平衡**：確保 AI 和 Human 文本數量相當
2. **特徵工程**：調整 TF-IDF 的 `max_features`
3. **超參數調優**：使用 GridSearch 尋找最佳參數
4. **模型融合**：使用 Hybrid 模型結合多個模型優勢

## 📚 更多資訊

- `README.md` - 完整專案說明
- `人類文本收集說明.md` - 人類文本收集指南
- `data/README_HUMAN_TEXTS.md` - 詳細收集說明

