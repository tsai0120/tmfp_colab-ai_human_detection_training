# AI vs Human 文本偵測平台

一個完整的 AI 與人類文本偵測系統，包含資料生成、模型訓練、模型管理和推論 API，以及現代化的前端介面。

## 📁 專案架構

```
FP/
├── data/                 # 資料目錄
│   ├── topics.txt        # 主題列表
│   └── generated_ai.csv  # 生成的 AI 文本
├── models/               # 訓練後的模型
│   ├── tfidf_svm/
│   ├── tfidf_lr/
│   ├── bert/
│   ├── roberta_lora/
│   └── hybrid/
├── api/                  # 後端 API
│   ├── predict.py        # 推論 API
│   └── train.py          # 訓練 API (Admin)
├── frontend/             # Next.js 前端
│   ├── pages/
│   │   ├── auth/login.tsx
│   │   ├── inference/
│   │   └── dashboard/
│   └── ...
├── utils/                # 工具模組
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── linguistic_features.py
│   └── model_loader.py
└── train_pipeline/        # 訓練腳本
    ├── train_svm.py
    ├── train_lr.py
    ├── train_bert.py
    ├── train_lora.py
    └── train_hybrid.py
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
# Python 依賴
pip install -r requirements.txt

# 前端依賴
cd frontend
npm install
```

### 2. 準備資料

將你的 `AI_Human.csv` 放在專案根目錄。

### 3. 訓練模型

```bash
# 訓練 SVM
python train_pipeline/train_svm.py --data AI_Human.csv

# 訓練 LR
python train_pipeline/train_lr.py --data AI_Human.csv

# 訓練 BERT
python train_pipeline/train_bert.py --data AI_Human.csv --epochs 3

# 訓練 LoRA
python train_pipeline/train_lora.py --data AI_Human.csv --epochs 3

# 訓練 Hybrid
python train_pipeline/train_hybrid.py --data AI_Human.csv
```

### 4. 啟動後端 API

```bash
# 推論 API (Port 8000)
python api/predict.py

# 訓練 API (Port 8001)
python api/train.py
```

### 5. 啟動前端

```bash
cd frontend
npm run dev
```

前端將在 http://localhost:3000 啟動。

## 🔐 登入資訊

- **Admin**: `admin` / `admin123` - 可進入模型管理頁面
- **User**: `user` / `user123` - 只能使用推論功能

## 📊 模型說明

### 1. TF-IDF + SVM
- 使用 TF-IDF 向量化文本
- SVM 分類器
- 可調參數：C, kernel

### 2. TF-IDF + Logistic Regression
- 使用 TF-IDF 向量化文本
- 邏輯回歸分類器
- 可調參數：C

### 3. BERT
- 使用 BERT-base 模型
- 微調分類頭
- 可調參數：epochs, learning_rate, batch_size

### 4. RoBERTa + LoRA
- 使用 RoBERTa-base 作為基礎模型
- LoRA 低秩適應技術
- 可調參數：lora_rank, lora_alpha, epochs

### 5. Hybrid
- 結合多個模型的預測結果
- 使用語言特徵（TTR, burstiness, entropy 等）
- MLP 分類器
- 可調參數：hidden_layer_sizes

## 🎯 API 端點

### 推論 API (Port 8000)

- `POST /predict` - 文本偵測
- `GET /health` - 健康檢查
- `GET /models` - 可用模型列表

### 訓練 API (Port 8001, Admin Only)

- `POST /train` - 訓練模型
- `GET /metrics` - 取得所有模型效能
- `GET /metrics/{model_name}` - 取得特定模型效能
- `GET /status` - 訓練狀態

## 📝 語言特徵說明

系統會計算以下語言特徵：

1. **Type-Token Ratio (TTR)**: 詞彙多樣性
2. **Mean Sentence Length**: 平均句子長度
3. **Burstiness**: 詞彙集中程度
4. **Punctuation Ratio**: 標點符號比例
5. **Character Entropy**: 字元層級熵
6. **Perplexity** (可選): GPT-2 / RoBERTa 困惑度

## 🔧 環境變數

建立 `.env` 檔案（可選）：

```env
OPENAI_API_KEY=your_api_key_here
ADMIN_TOKEN=admin_secret_token_12345
API_URL=http://localhost:8000
TRAIN_API_URL=http://localhost:8001
```

## 📌 關於人類文本收集

為了優化語言特徵分析，建議收集以下人類文本：

### 建議數量
- **至少 100-200 篇**人類撰寫的文本
- 與 AI 生成文本數量相當（平衡資料集）

### 建議主題
參考 `data/topics.txt` 中的主題，收集相同主題的人類文本，以便：
1. 與 AI 生成文本進行公平比較
2. 確保主題一致性
3. 提高模型泛化能力

### 儲存位置
將人類文本儲存在 `data/human_texts.csv`，格式：

```csv
text,topic,label
"人類撰寫的文本內容...","人工智慧的未來發展","Human"
```

或合併到主資料集，確保 `generated` 欄位標記為 `False` 或 `0`。

## 🛠️ 開發說明

### 資料生成

```python
from utils.data_generation import generate_ai_texts

generate_ai_texts(
    output_path="data/generated_ai.csv",
    num_per_prompt=15,
    temperature=0.7,
    max_tokens=500,
    api_key="your_openai_api_key"
)
```

### 自訂訓練參數

各訓練腳本支援命令列參數，例如：

```bash
python train_pipeline/train_svm.py \
    --data AI_Human.csv \
    --C 10.0 \
    --kernel rbf \
    --max_features 10000
```

## 📄 授權

本專案僅供學術研究使用。

## 🤝 貢獻

歡迎提出 Issue 和 Pull Request！

