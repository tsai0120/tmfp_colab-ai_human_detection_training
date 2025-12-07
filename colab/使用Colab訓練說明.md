# 📘 使用 Google Colab 訓練模型完整指南

## 🎯 為什麼使用 Colab？

### 優勢
- ✅ **免費 GPU** - 訓練 BERT/LoRA 模型必備
- ✅ **無需本地配置** - 開箱即用
- ✅ **雲端運算** - 不占用本地資源
- ✅ **自動儲存** - 不怕中斷

### 適用場景
- 🚀 訓練 BERT 和 LoRA 模型（需要 GPU）
- 🚀 本地電腦效能不足
- 🚀 想要快速測試不同參數

## 📋 完整步驟

### 步驟 1: 準備檔案

1. **下載 Notebook**
   - 位置：`colab/ai_human_detection_training.ipynb`
   - 或直接從專案中取得

2. **準備資料**
   - 確保 `AI_Human.csv` 已準備好
   - 檔案大小建議 < 500MB（Colab 上傳限制）

### 步驟 2: 上傳到 Colab

1. **打開 Google Colab**
   ```
   https://colab.research.google.com/
   ```

2. **上傳 Notebook**
   - 點擊「檔案」→「上傳 Notebook」
   - 選擇 `ai_human_detection_training.ipynb`

3. **或從 GitHub 載入**
   ```python
   # 在 Colab 中執行
   !git clone https://github.com/your-username/your-repo.git
   %cd your-repo/colab
   ```

### 步驟 3: 啟用 GPU

1. **變更執行階段**
   - 點擊「執行階段」→「變更執行階段類型」
   - 硬體加速器：選擇「GPU」
   - GPU 類型：選擇「T4」（免費）或「A100」（付費）

2. **確認 GPU**
   ```python
   # 執行此 cell 確認
   !nvidia-smi
   ```

### 步驟 4: 執行訓練

#### 4.1 環境設置
```python
# 執行第一個 cell
# 會自動安裝所有必要的套件
```

#### 4.2 上傳資料
```python
# 執行第二個 cell
# 點擊「選擇檔案」上傳 AI_Human.csv
```

#### 4.3 建立工具模組
```python
# 執行第三和第四個 cell
# 建立預處理和特徵提取模組
```

#### 4.4 訓練模型

**訓練順序建議：**

1. **先訓練 SVM**（5-30 分鐘）
   - 不需要 GPU
   - 可以驗證流程是否正確

2. **再訓練 LR**（1-5 分鐘）
   - 很快完成
   - 為 Hybrid 模型做準備

3. **訓練 BERT**（30 分鐘 - 2 小時）
   - 需要 GPU
   - 建議使用較少資料（20000 筆）

4. **訓練 LoRA**（30 分鐘 - 2 小時）
   - 需要 GPU
   - 比 BERT 稍快

5. **最後訓練 Hybrid**（10-20 分鐘）
   - 需要先有 SVM 和 LR 模型
   - 不需要 GPU

### 步驟 5: 下載模型

每個模型訓練完成後會自動打包下載，或執行最後一個 cell 下載所有模型。

## 💡 實用技巧

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
    num_train_epochs=2,  # 改為 2 個 epoch
    ...
)
```

### 2. 監控訓練

**查看 GPU 使用率：**
```python
!nvidia-smi
```

**查看訓練日誌：**
```python
# BERT 訓練會自動顯示進度
# 觀察 loss 和 accuracy 的變化
```

### 3. 處理中斷

**自動儲存：**
- Colab 會自動儲存 Notebook
- 已訓練的模型會保留在 `models/` 目錄

**繼續訓練：**
```python
# 如果中斷，可以從最後完成的 cell 繼續
# 已訓練的模型不需要重新訓練
```

### 4. 下載模型

**個別下載：**
- 每個模型訓練完成後會自動下載 zip 檔案

**批量下載：**
```python
# 執行最後一個 cell
!zip -r all_models.zip models/
files.download('all_models.zip')
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
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# 如果沒有 GPU，可以：
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

## 📊 訓練時間估算

| 模型 | 資料量 | GPU | 預期時間 |
|------|--------|-----|----------|
| SVM | 100K | ❌ | 10-30 分鐘 |
| LR | 100K | ❌ | 2-5 分鐘 |
| BERT | 20K | ✅ | 30-60 分鐘 |
| LoRA | 20K | ✅ | 20-40 分鐘 |
| Hybrid | 10K | ❌ | 10-15 分鐘 |

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

## 📝 範例工作流程

```
1. 上傳 Notebook 和資料
   ↓
2. 啟用 GPU
   ↓
3. 執行環境設置
   ↓
4. 訓練 SVM（驗證流程）
   ↓
5. 訓練 LR
   ↓
6. 訓練 BERT（使用 GPU）
   ↓
7. 訓練 LoRA（使用 GPU）
   ↓
8. 訓練 Hybrid
   ↓
9. 下載所有模型
   ↓
10. 解壓到本地專案
```

## 🔗 相關資源

- [Colab Notebook 檔案](../colab/ai_human_detection_training.ipynb)
- [Colab README](./README.md)
- [專案主 README](../README.md)

