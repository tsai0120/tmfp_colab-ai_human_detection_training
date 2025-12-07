# AI vs Human 文本偵測模型訓練 - Google Colab 版本

本專案提供完整的 Google Colab Notebook，用於訓練五個 AI vs Human 文本偵測模型。

## 📋 模型列表

1. **TF-IDF + SVM** - 傳統機器學習方法
2. **TF-IDF + Logistic Regression** - 快速且有效的分類器
3. **BERT** - 深度學習模型（需要 GPU）
4. **RoBERTa + LoRA** - 高效能的微調模型（需要 GPU）
5. **Hybrid** - 結合多個模型的混合模型

## 🚀 快速開始

### 步驟 1: 打開 Google Colab

1. 訪問 [Google Colab](https://colab.research.google.com/)
2. 點擊「檔案」→「上傳 Notebook」
3. 上傳 `ai_human_detection_training.ipynb`

### 步驟 2: 啟用 GPU（推薦）

**對於 BERT 和 LoRA 模型，GPU 是必需的！**

1. 點擊「執行階段」→「變更執行階段類型」
2. 硬體加速器：選擇「**GPU**」
3. GPU 類型：選擇「**T4**」（免費）或「**A100**」（付費）
4. 點擊「儲存」

**驗證 GPU：**
```python
# 執行此 cell 確認 GPU 可用
import torch
print(f"GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
```

### 步驟 3: 安裝環境

執行第一個 cell（環境設置），會自動安裝所有必要的套件：
- pandas, numpy, scikit-learn
- nltk
- transformers, torch, peft
- tqdm

### 步驟 4: 上傳資料

1. 執行「上傳資料」cell
2. 點擊「選擇檔案」
3. 選擇您的 `AI_Human.csv` 檔案
4. 等待上傳完成

**資料格式要求：**
- CSV 檔案
- 必須包含 `text` 和 `generated` 欄位
- `generated`: `True/False` 或 `1/0` 或 `'AI'/'Human'`

### 步驟 5: 建立工具模組

執行「建立工具模組」cell，會自動建立：
- `preprocessing.py` - 文本預處理工具
- `linguistic_features.py` - 語言特徵提取工具

### 步驟 6: 開始訓練

依序執行各個模型的訓練 cell：

#### 6.1 訓練 TF-IDF + SVM

```python
# 執行 SVM 訓練 cell
# 預期時間：5-30 分鐘（取決於資料量）
# 不需要 GPU
```

**特點：**
- 使用 RBF kernel（效果較好但較慢）
- 可調整參數：`C`, `kernel`, `max_features`
- 訓練完成後會自動下載模型

#### 6.2 訓練 TF-IDF + Logistic Regression

```python
# 執行 LR 訓練 cell
# 預期時間：1-5 分鐘
# 不需要 GPU
```

**特點：**
- 訓練速度快
- 適合快速驗證
- 為 Hybrid 模型做準備

#### 6.3 訓練 BERT（需要 GPU）

```python
# 執行 BERT 訓練 cell
# 預期時間：30 分鐘 - 2 小時
# 需要 GPU
```

**特點：**
- 使用 `bert-base-uncased` 模型
- 預設使用 20,000 筆資料（可調整）
- 2 個 epochs（可調整）
- 自動使用 FP16 加速（如果有 GPU）

**如果沒有 GPU：**
- 可以改用 CPU 訓練（非常慢，不推薦）
- 或使用較小的模型（如 `distilbert-base-uncased`）

#### 6.4 訓練 RoBERTa + LoRA（需要 GPU）

```python
# 執行 LoRA 訓練 cell
# 預期時間：20-40 分鐘
# 需要 GPU
```

**特點：**
- 使用 LoRA（Low-Rank Adaptation）技術
- 參數效率高，訓練速度快
- 效果通常比完整微調好

#### 6.5 訓練 Hybrid 模型

```python
# 執行 Hybrid 訓練 cell
# 預期時間：10-20 分鐘
# 需要先訓練 SVM 和 LR
```

**特點：**
- 結合多個模型的預測結果
- 使用語言特徵（TTR, burstiness, entropy 等）
- 通常效果最好

### 步驟 7: 下載模型

每個模型訓練完成後會自動打包下載，或執行最後一個 cell 下載所有模型：

```python
# 打包所有模型
!zip -r all_models.zip models/
files.download('all_models.zip')
```

## 📊 訓練時間估算

| 模型 | 資料量 | GPU | 預期時間 | 備註 |
|------|--------|-----|----------|------|
| SVM | 100K | ❌ | 10-30 分鐘 | RBF kernel 較慢 |
| LR | 100K | ❌ | 2-5 分鐘 | 最快 |
| BERT | 20K | ✅ | 30-60 分鐘 | 需要 GPU |
| LoRA | 20K | ✅ | 20-40 分鐘 | 需要 GPU |
| Hybrid | 10K | ❌ | 10-15 分鐘 | 需要 SVM+LR |

## 💡 使用技巧

### 1. 加速訓練

**減少資料量：**
```python
# 在載入資料時
df = pd.read_csv('AI_Human.csv', nrows=20000)  # 只使用 20000 筆
```

**使用較小的模型：**
```python
# BERT 改為 DistilBERT（更快）
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)
```

**減少 epochs：**
```python
training_args = TrainingArguments(
    num_train_epochs=2,  # 改為 2 個 epoch（預設）
    ...
)
```

### 2. 監控訓練

**查看 GPU 使用率：**
```python
!nvidia-smi
```

**查看訓練進度：**
- BERT/LoRA 訓練會自動顯示進度條
- 觀察 loss 和 accuracy 的變化

### 3. 處理中斷

**自動儲存：**
- Colab 會自動儲存 Notebook
- 已訓練的模型會保留在 `models/` 目錄

**繼續訓練：**
- 重新連接 Colab
- 檢查 `models/` 目錄中的模型
- 從最後完成的 cell 繼續
- 已訓練的模型不需要重新訓練

### 4. 下載模型

**個別下載：**
- 每個模型訓練完成後會自動下載 zip 檔案

**批量下載：**
```python
# 執行最後一個 cell
!zip -r all_models.zip models/
files.download('all_models.zip')
```

**使用 Google Drive：**
```python
from google.colab import drive
drive.mount('/content/drive')

# 複製到 Google Drive
!cp -r models/ /content/drive/MyDrive/
```

## ⚠️ 注意事項

### Colab 限制

1. **使用時間**
   - 免費版：每次最多 12 小時
   - 長時間訓練可能被中斷

2. **GPU 配額**
   - 免費版：每天有限配額
   - 用完需等待重置（通常 24 小時）

3. **記憶體**
   - 免費版：約 12GB RAM
   - 如果不足，減少資料量或批次大小

4. **儲存空間**
   - 臨時儲存約 80GB
   - 訓練完成後記得下載模型

### 資料安全

- ⚠️ 上傳的資料會儲存在 Colab 環境
- ⚠️ 敏感資料請謹慎使用
- ⚠️ 訓練完成後記得刪除上傳的資料

## 🔧 故障排除

### 問題 1: GPU 無法使用

**解決方法：**
```python
# 檢查 GPU
import torch
print(torch.cuda.is_available())

# 如果沒有 GPU：
# 1. 等待配額重置
# 2. 使用 CPU 訓練（較慢）
# 3. 升級到 Colab Pro
```

### 問題 2: 記憶體不足

**解決方法：**
```python
# 減少資料量
df = pd.read_csv('AI_Human.csv', nrows=10000)

# 減少批次大小
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # 改為 8
    ...
)

# 清理記憶體
import gc
torch.cuda.empty_cache()
gc.collect()
```

### 問題 3: 訓練中斷

**解決方法：**
1. 重新連接 Colab
2. 檢查 `models/` 目錄中的模型
3. 從最後完成的 cell 繼續
4. 已訓練的模型不需要重新訓練

### 問題 4: 下載失敗

**解決方法：**
```python
# 使用 Google Drive 儲存
from google.colab import drive
drive.mount('/content/drive')

# 複製到 Google Drive
!cp -r models/ /content/drive/MyDrive/
```

## 📝 模型使用

訓練完成後，下載的模型可以：

1. **解壓到本地專案：**
   ```bash
   unzip all_models.zip -d /path/to/your/project/models/
   ```

2. **在本地 API 中使用：**
   - 將模型放在 `models/` 目錄
   - 啟動 API：`python api/predict.py`
   - 使用前端進行推論

3. **查看模型效能：**
   - 每個模型目錄都有 `metrics.json`
   - 包含準確率等指標

## 🎯 最佳實踐

1. **先測試小資料集**
   - 使用 1000 筆資料測試流程
   - 確認無誤後再使用完整資料

2. **分階段訓練**
   - 先訓練簡單模型（SVM, LR）
   - 再訓練深度學習模型（BERT, LoRA）

3. **定期下載**
   - 每完成一個模型就下載
   - 避免中斷後重新訓練

4. **記錄參數**
   - 在 Notebook 中記錄使用的參數
   - 方便後續調整和比較

## 📚 相關資源

- [Google Colab 官方文檔](https://colab.research.google.com/notebooks/intro.ipynb)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch 文檔](https://pytorch.org/docs/stable/index.html)
- [專案主 README](../README.md)

## 📞 問題回報

如有問題，請：
1. 檢查本 README 的故障排除部分
2. 查看 Notebook 中的註解
3. 在 GitHub 上開 Issue

## 📄 授權

本專案僅供學術研究使用。

---

**祝訓練順利！** 🚀
