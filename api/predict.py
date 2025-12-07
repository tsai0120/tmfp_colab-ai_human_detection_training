"""
推論 API - FastAPI
提供文本偵測服務
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn

# 加入專案路徑
sys.path.append(str(Path(__file__).parent.parent))
from utils.model_loader import ModelLoader

app = FastAPI(title="AI vs Human 文本偵測 API")

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應限制特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域模型載入器
model_loader = ModelLoader(models_dir="models")


class TextRequest(BaseModel):
    """文本請求模型"""
    text: str
    selected_model: Optional[str] = None  # 可選：只使用特定模型


class PredictionResponse(BaseModel):
    """預測回應模型"""
    selected_model: str
    probability_ai: float
    label: str  # "AI" 或 "Human"
    details: Dict[str, float]  # 所有模型的預測結果


@app.on_event("startup")
async def startup_event():
    """啟動時載入模型"""
    print("🚀 啟動 API 服務...")
    print("📥 載入模型...")
    results = model_loader.load_all_models()
    
    loaded_count = sum(1 for success in results.values() if success)
    print(f"✅ 已載入 {loaded_count}/{len(results)} 個模型")
    
    for model_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {model_name}")


@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "AI vs Human 文本偵測 API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - 文本偵測",
            "/health": "GET - 健康檢查",
            "/models": "GET - 可用模型列表"
        }
    }


@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "models_loaded": len(model_loader.loaded_models)
    }


@app.get("/models")
async def list_models():
    """列出可用模型"""
    available_models = list(model_loader.loaded_models.keys())
    return {
        "available_models": available_models,
        "count": len(available_models)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """
    文本偵測
    
    Args:
        request: 包含文本和可選的模型選擇
    
    Returns:
        預測結果，包含 AI 機率和所有模型的詳細結果
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="文本不能為空")
    
    text = request.text.strip()
    
    # 如果指定了特定模型
    if request.selected_model:
        model_name = request.selected_model.lower()
        
        if model_name == "svm" or model_name == "tfidf_svm":
            prob = model_loader.predict_tfidf_svm(text)
            return PredictionResponse(
                selected_model="svm",
                probability_ai=prob,
                label="AI" if prob > 0.5 else "Human",
                details={"svm": prob}
            )
        elif model_name == "lr" or model_name == "tfidf_lr":
            prob = model_loader.predict_tfidf_lr(text)
            return PredictionResponse(
                selected_model="lr",
                probability_ai=prob,
                label="AI" if prob > 0.5 else "Human",
                details={"lr": prob}
            )
        elif model_name == "bert":
            prob = model_loader.predict_bert(text)
            return PredictionResponse(
                selected_model="bert",
                probability_ai=prob,
                label="AI" if prob > 0.5 else "Human",
                details={"bert": prob}
            )
        elif model_name == "lora" or model_name == "roberta_lora":
            prob = model_loader.predict_roberta_lora(text)
            return PredictionResponse(
                selected_model="lora",
                probability_ai=prob,
                label="AI" if prob > 0.5 else "Human",
                details={"lora": prob}
            )
        elif model_name == "hybrid":
            prob = model_loader.predict_hybrid(text)
            return PredictionResponse(
                selected_model="hybrid",
                probability_ai=prob,
                label="AI" if prob > 0.5 else "Human",
                details={"hybrid": prob}
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"未知的模型: {model_name}。可用模型: svm, lr, bert, lora, hybrid"
            )
    
    # 使用所有模型預測
    try:
        all_predictions = model_loader.predict_all(text)
        
        # 選擇最高機率作為最終結果（或使用 hybrid）
        if 'hybrid' in all_predictions and all_predictions['hybrid'] != 0.5:
            final_prob = all_predictions['hybrid']
            selected_model = "hybrid"
        else:
            # 使用平均機率
            valid_probs = [p for p in all_predictions.values() if p != 0.5]
            if valid_probs:
                final_prob = sum(valid_probs) / len(valid_probs)
            else:
                final_prob = 0.5
            selected_model = "ensemble"
        
        return PredictionResponse(
            selected_model=selected_model,
            probability_ai=final_prob,
            label="AI" if final_prob > 0.5 else "Human",
            details=all_predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(texts: List[str]):
    """
    批量預測
    
    Args:
        texts: 文本列表
    
    Returns:
        批量預測結果
    """
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="文本列表不能為空")
    
    results = []
    for text in texts:
        try:
            all_predictions = model_loader.predict_all(text)
            valid_probs = [p for p in all_predictions.values() if p != 0.5]
            final_prob = sum(valid_probs) / len(valid_probs) if valid_probs else 0.5
            
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "probability_ai": final_prob,
                "label": "AI" if final_prob > 0.5 else "Human",
                "details": all_predictions
            })
        except Exception as e:
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "error": str(e)
            })
    
    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    uvicorn.run(
        "predict:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

