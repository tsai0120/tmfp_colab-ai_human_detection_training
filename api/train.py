"""
Admin 訓練 API - FastAPI
提供模型訓練和管理功能（僅限 Admin）
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
from datetime import datetime

# 加入專案路徑
sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(title="AI vs Human 模型訓練 API (Admin)")

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 簡單的認證（生產環境應使用 JWT）
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin_secret_token_12345")

# 訓練狀態追蹤
training_status = {
    "is_training": False,
    "current_model": None,
    "progress": 0,
    "message": "",
    "start_time": None,
    "end_time": None
}


def verify_admin(token: str):
    """驗證 Admin token"""
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="未授權：需要 Admin 權限")
    return True


class TrainRequest(BaseModel):
    """訓練請求模型"""
    model_name: str  # svm, lr, bert, lora, hybrid
    parameters: Optional[Dict] = None
    admin_token: str


class MetricsResponse(BaseModel):
    """模型效能回應"""
    model_name: str
    metrics: Dict


@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "AI vs Human 模型訓練 API (Admin Only)",
        "version": "1.0.0",
        "endpoints": {
            "/train": "POST - 訓練模型",
            "/metrics/{model_name}": "GET - 取得模型效能",
            "/metrics": "GET - 取得所有模型效能",
            "/status": "GET - 訓練狀態"
        }
    }


@app.get("/status")
async def get_training_status():
    """取得訓練狀態"""
    return training_status


@app.get("/metrics")
async def get_all_metrics():
    """取得所有模型的效能指標"""
    models_dir = Path("models")
    all_metrics = {}
    
    model_names = ["tfidf_svm", "tfidf_lr", "bert", "roberta_lora", "hybrid"]
    
    for model_name in model_names:
        metrics_path = models_dir / model_name / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    all_metrics[model_name] = json.load(f)
            except Exception as e:
                all_metrics[model_name] = {"error": str(e)}
        else:
            all_metrics[model_name] = {"error": "模型尚未訓練"}
    
    return all_metrics


@app.get("/metrics/{model_name}", response_model=MetricsResponse)
async def get_model_metrics(model_name: str):
    """取得特定模型的效能指標"""
    models_dir = Path("models")
    metrics_path = models_dir / model_name / "metrics.json"
    
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"模型 {model_name} 的效能指標不存在"
        )
    
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        return MetricsResponse(
            model_name=model_name,
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"讀取效能指標失敗: {str(e)}"
        )


def run_training(model_name: str, parameters: Dict):
    """執行訓練（背景任務）"""
    global training_status
    
    training_status["is_training"] = True
    training_status["current_model"] = model_name
    training_status["progress"] = 0
    training_status["message"] = "初始化訓練..."
    training_status["start_time"] = datetime.now().isoformat()
    training_status["end_time"] = None
    
    try:
        # 根據模型名稱執行對應的訓練腳本
        script_map = {
            "svm": "train_pipeline/train_svm.py",
            "tfidf_svm": "train_pipeline/train_svm.py",
            "lr": "train_pipeline/train_lr.py",
            "tfidf_lr": "train_pipeline/train_lr.py",
            "bert": "train_pipeline/train_bert.py",
            "lora": "train_pipeline/train_lora.py",
            "roberta_lora": "train_pipeline/train_lora.py",
            "hybrid": "train_pipeline/train_hybrid.py"
        }
        
        script_path = script_map.get(model_name)
        if not script_path:
            raise ValueError(f"未知的模型: {model_name}")
        
        training_status["message"] = f"準備訓練腳本: {script_path}"
        training_status["progress"] = 5
        
        # 建立命令
        cmd = ["python", script_path]
        
        # 添加參數
        if parameters:
            if model_name in ["svm", "tfidf_svm"]:
                if "C" in parameters:
                    cmd.extend(["--C", str(parameters["C"])])
                if "kernel" in parameters:
                    cmd.extend(["--kernel", str(parameters["kernel"])])
                if "max_features" in parameters:
                    cmd.extend(["--max_features", str(parameters["max_features"])])
            
            elif model_name in ["lr", "tfidf_lr"]:
                if "C" in parameters:
                    cmd.extend(["--C", str(parameters["C"])])
                if "max_features" in parameters:
                    cmd.extend(["--max_features", str(parameters["max_features"])])
            
            elif model_name == "bert":
                if "epochs" in parameters:
                    cmd.extend(["--epochs", str(parameters["epochs"])])
                if "learning_rate" in parameters:
                    cmd.extend(["--learning_rate", str(parameters["learning_rate"])])
                if "batch_size" in parameters:
                    cmd.extend(["--batch_size", str(parameters["batch_size"])])
            
            elif model_name in ["lora", "roberta_lora"]:
                if "epochs" in parameters:
                    cmd.extend(["--epochs", str(parameters["epochs"])])
                if "learning_rate" in parameters:
                    cmd.extend(["--learning_rate", str(parameters["learning_rate"])])
                if "lora_rank" in parameters:
                    cmd.extend(["--lora_rank", str(parameters["lora_rank"])])
                if "lora_alpha" in parameters:
                    cmd.extend(["--lora_alpha", str(parameters["lora_alpha"])])
            
            elif model_name == "hybrid":
                if "hidden_layers" in parameters:
                    layers = parameters["hidden_layers"]
                    if isinstance(layers, list):
                        cmd.extend(["--hidden_layers"] + [str(l) for l in layers])
        
        training_status["message"] = "載入資料中..."
        training_status["progress"] = 10
        
        # 執行訓練，即時讀取輸出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=Path(__file__).parent.parent
        )
        
        # 解析輸出以更新進度
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            line_lower = line.lower()
            
            # 根據輸出更新進度
            if "載入資料" in line or "loading" in line_lower:
                training_status["progress"] = 15
                training_status["message"] = "載入資料中..."
            elif "預處理" in line or "preprocessing" in line_lower:
                training_status["progress"] = 25
                training_status["message"] = "預處理資料中..."
            elif "tfidf" in line_lower or "向量化" in line:
                training_status["progress"] = 35
                training_status["message"] = "建立特徵向量..."
            elif "訓練" in line or "training" in line_lower or "fit" in line_lower:
                training_status["progress"] = 40
                training_status["message"] = "訓練模型中...（這可能需要 5-30 分鐘）"
            elif "評估" in line or "evaluat" in line_lower:
                training_status["progress"] = 85
                training_status["message"] = "評估模型效能..."
            elif "儲存" in line or "saving" in line_lower or "save" in line_lower:
                training_status["progress"] = 90
                training_status["message"] = "儲存模型中..."
            elif "完成" in line or "完成" in line or "done" in line_lower:
                training_status["progress"] = 95
                training_status["message"] = "訓練完成，正在儲存..."
        
        # 等待進程完成
        process.wait()
        
        if process.returncode == 0:
            training_status["message"] = "✅ 訓練成功完成！"
            training_status["progress"] = 100
            training_status["end_time"] = datetime.now().isoformat()
        else:
            error_msg = "\n".join(output_lines[-10:])  # 最後 10 行錯誤訊息
            raise Exception(f"訓練失敗: {error_msg}")
    
    except Exception as e:
        training_status["message"] = f"❌ 訓練錯誤: {str(e)}"
        training_status["progress"] = 0
        training_status["end_time"] = datetime.now().isoformat()
        raise
    
    finally:
        training_status["is_training"] = False


@app.post("/train")
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks
):
    """
    訓練模型（Admin only）
    
    Args:
        request: 訓練請求，包含模型名稱和參數
        background_tasks: FastAPI 背景任務
    
    Returns:
        訓練任務已啟動的確認
    """
    # 驗證 Admin token
    verify_admin(request.admin_token)
    
    # 檢查是否正在訓練
    if training_status["is_training"]:
        raise HTTPException(
            status_code=400,
            detail="已有訓練任務正在執行中"
        )
    
    # 驗證模型名稱
    valid_models = [
        "svm", "tfidf_svm", "lr", "tfidf_lr",
        "bert", "lora", "roberta_lora", "hybrid"
    ]
    
    if request.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"無效的模型名稱。可用模型: {', '.join(valid_models)}"
        )
    
    # 在背景執行訓練
    background_tasks.add_task(
        run_training,
        request.model_name,
        request.parameters or {}
    )
    
    return {
        "message": f"模型 {request.model_name} 的訓練任務已啟動",
        "model_name": request.model_name,
        "status": "training_started"
    }


if __name__ == "__main__":
    uvicorn.run(
        "train:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

