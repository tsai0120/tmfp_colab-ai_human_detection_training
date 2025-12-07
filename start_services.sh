#!/bin/bash

# 啟動所有服務的腳本

echo "🚀 啟動 AI vs Human 文本偵測平台..."

# 啟動推論 API
echo "📡 啟動推論 API (Port 8000)..."
python api/predict.py &
PREDICT_PID=$!

# 啟動訓練 API
echo "🔧 啟動訓練 API (Port 8001)..."
python api/train.py &
TRAIN_PID=$!

# 等待一下讓 API 啟動
sleep 3

# 啟動前端
echo "🎨 啟動前端 (Port 5173)..."
cd frontend-vite
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ 所有服務已啟動！"
echo "📡 推論 API: http://localhost:8000"
echo "🔧 訓練 API: http://localhost:8001"
echo "🎨 前端: http://localhost:5173"
echo ""
echo "按 Ctrl+C 停止所有服務"

# 等待中斷信號
trap "kill $PREDICT_PID $TRAIN_PID $FRONTEND_PID; exit" INT TERM
wait

