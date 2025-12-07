import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import { TRAIN_API_URL } from '../../App'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

export default function ModelsDashboard({ user, logout }: { user: any; logout: () => void }) {
  const navigate = useNavigate()
  const [metrics, setMetrics] = useState<any>({})
  const [loading, setLoading] = useState(true)
  const [trainingStatus, setTrainingStatus] = useState<any>(null)
  const [selectedModel, setSelectedModel] = useState('svm')
  const [trainParams, setTrainParams] = useState<any>({})
  const [training, setTraining] = useState(false)

  useEffect(() => {
    fetchMetrics()
    fetchTrainingStatus()
    
    // 如果正在訓練，更頻繁地更新狀態
    const updateInterval = trainingStatus?.is_training ? 1000 : 5000
    const interval = setInterval(() => {
      fetchTrainingStatus()
    }, updateInterval)
    
    return () => clearInterval(interval)
  }, [trainingStatus?.is_training])

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${TRAIN_API_URL}/metrics`)
      setMetrics(response.data)
    } catch (err) {
      console.error('取得效能指標失敗:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchTrainingStatus = async () => {
    try {
      const response = await axios.get(`${TRAIN_API_URL}/status`)
      const newStatus = response.data
      setTrainingStatus(newStatus)
      
      // 如果訓練剛完成，重新載入指標
      if (!newStatus.is_training && training) {
        setTraining(false)
        setTimeout(() => {
          fetchMetrics() // 訓練完成後重新載入指標
        }, 1000)
      }
    } catch (err) {
      console.error('取得訓練狀態失敗:', err)
    }
  }

  const handleTrain = async () => {
    setTraining(true)
    try {
      const token = localStorage.getItem('token') || 'admin_secret_token_12345'
      await axios.post(`${TRAIN_API_URL}/train`, {
        model_name: selectedModel,
        parameters: trainParams[selectedModel] || {},
        admin_token: token
      })
      alert('訓練任務已啟動！')
    } catch (err: any) {
      alert(err.response?.data?.detail || '啟動訓練失敗')
      setTraining(false)
    }
  }

  const updateParam = (model: string, key: string, value: any) => {
    setTrainParams((prev: any) => ({
      ...prev,
      [model]: {
        ...prev[model],
        [key]: value
      }
    }))
  }

  const modelNames = ['tfidf_svm', 'tfidf_lr', 'bert', 'roberta_lora', 'hybrid']
  const displayNames: any = {
    'tfidf_svm': 'TF-IDF + SVM',
    'tfidf_lr': 'TF-IDF + LR',
    'bert': 'BERT',
    'roberta_lora': 'RoBERTa + LoRA',
    'hybrid': 'Hybrid'
  }

  // 準備圖表資料
  const chartData = {
    labels: modelNames.map(name => displayNames[name]),
    datasets: [
      {
        label: 'Baseline Accuracy',
        data: modelNames.map(name => metrics[name]?.baseline_accuracy || 0),
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1,
      },
      {
        label: 'Prompt A Accuracy',
        data: modelNames.map(name => metrics[name]?.prompt_A_accuracy || 0),
        backgroundColor: 'rgba(16, 185, 129, 0.5)',
        borderColor: 'rgba(16, 185, 129, 1)',
        borderWidth: 1,
      },
      {
        label: 'Prompt B Accuracy',
        data: modelNames.map(name => metrics[name]?.prompt_B_accuracy || 0),
        backgroundColor: 'rgba(245, 158, 11, 0.5)',
        borderColor: 'rgba(245, 158, 11, 1)',
        borderWidth: 1,
      },
      {
        label: 'Prompt C Accuracy',
        data: modelNames.map(name => metrics[name]?.prompt_C_accuracy || 0),
        backgroundColor: 'rgba(239, 68, 68, 0.5)',
        borderColor: 'rgba(239, 68, 68, 1)',
        borderWidth: 1,
      },
    ],
  }

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">模型管理儀表板</h1>
          <div className="flex gap-4 items-center">
            <span className="text-gray-600">Admin: {user.username}</span>
            <button
              onClick={() => navigate('/inference')}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              推論頁面
            </button>
            <button
              onClick={logout}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
            >
              登出
            </button>
          </div>
        </div>

        {/* Training Status */}
        {trainingStatus?.is_training && (
          <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-6 mb-6 shadow-lg">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                  <h3 className="text-lg font-bold text-blue-800">訓練進行中</h3>
                </div>
                <span className="text-sm font-semibold text-blue-600">
                  {trainingStatus.progress}%
                </span>
              </div>
              
              <div>
                <p className="text-sm font-medium text-blue-700 mb-2">
                  {trainingStatus.message || '訓練中，請稍候...'}
                </p>
                <p className="text-xs text-blue-600">
                  模型: <span className="font-semibold">{trainingStatus.current_model}</span>
                  {trainingStatus.start_time && (
                    <> | 開始時間: {new Date(trainingStatus.start_time).toLocaleTimeString()}</>
                  )}
                </p>
              </div>
              
              <div className="w-full bg-blue-200 rounded-full h-4 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-blue-500 to-blue-600 h-4 rounded-full transition-all duration-500 ease-out flex items-center justify-end pr-2"
                  style={{ width: `${trainingStatus.progress}%` }}
                >
                  {trainingStatus.progress > 10 && (
                    <span className="text-xs font-semibold text-white">
                      {trainingStatus.progress}%
                    </span>
                  )}
                </div>
              </div>
              
              {/* 時間估算 */}
              {trainingStatus.progress > 10 && trainingStatus.progress < 100 && (
                <div className="text-xs text-blue-600 italic">
                  💡 提示: SVM 訓練通常需要 5-30 分鐘，請耐心等待
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Training Completed */}
        {trainingStatus && !trainingStatus.is_training && trainingStatus.end_time && (
          <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2">
              <span className="text-2xl">✅</span>
              <div>
                <h3 className="font-semibold text-green-800">訓練完成</h3>
                <p className="text-sm text-green-600">
                  完成時間: {new Date(trainingStatus.end_time).toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Metrics Chart */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">模型效能比較</h2>
          {loading ? (
            <div className="text-center py-8">載入中...</div>
          ) : (
            <Bar
              data={chartData}
              options={{
                responsive: true,
                plugins: {
                  legend: {
                    position: 'top' as const,
                  },
                  title: {
                    display: true,
                    text: '各模型在不同 Prompt 下的準確率',
                  },
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 1,
                  },
                },
              }}
            />
          )}
        </div>

        {/* Model Details and Training */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Metrics */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">模型效能詳情</h2>
            <div className="space-y-4">
              {modelNames.map((modelName) => {
                const modelMetrics = metrics[modelName]
                return (
                  <div
                    key={modelName}
                    className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer"
                    onClick={() => setSelectedModel(modelName)}
                  >
                    <h3 className="font-semibold text-gray-800 mb-2">
                      {displayNames[modelName]}
                    </h3>
                    {modelMetrics && !modelMetrics.error ? (
                      <div className="text-sm text-gray-600 space-y-1">
                        <div>Baseline: {(modelMetrics.baseline_accuracy * 100).toFixed(2)}%</div>
                        <div>Prompt A: {(modelMetrics.prompt_A_accuracy * 100).toFixed(2)}%</div>
                        <div>Prompt B: {(modelMetrics.prompt_B_accuracy * 100).toFixed(2)}%</div>
                        <div>Prompt C: {(modelMetrics.prompt_C_accuracy * 100).toFixed(2)}%</div>
                      </div>
                    ) : (
                      <div className="text-sm text-red-600">模型尚未訓練</div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Training Panel */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">模型訓練</h2>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                選擇模型
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                {modelNames.map(name => (
                  <option key={name} value={name}>{displayNames[name]}</option>
                ))}
              </select>
            </div>

            {/* SVM Parameters */}
            {selectedModel === 'tfidf_svm' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    C 參數
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={trainParams[selectedModel]?.C || 1.0}
                    onChange={(e) => updateParam(selectedModel, 'C', parseFloat(e.target.value))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Kernel
                  </label>
                  <select
                    value={trainParams[selectedModel]?.kernel || 'rbf'}
                    onChange={(e) => updateParam(selectedModel, 'kernel', e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  >
                    <option value="linear">Linear</option>
                    <option value="rbf">RBF</option>
                    <option value="poly">Polynomial</option>
                  </select>
                </div>
              </div>
            )}

            {/* LR Parameters */}
            {selectedModel === 'tfidf_lr' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  C 參數
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={trainParams[selectedModel]?.C || 1.0}
                  onChange={(e) => updateParam(selectedModel, 'C', parseFloat(e.target.value))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                />
              </div>
            )}

            {/* BERT Parameters */}
            {selectedModel === 'bert' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Epochs
                  </label>
                  <input
                    type="number"
                    value={trainParams[selectedModel]?.epochs || 3}
                    onChange={(e) => updateParam(selectedModel, 'epochs', parseInt(e.target.value))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    step="1e-6"
                    value={trainParams[selectedModel]?.learning_rate || 2e-5}
                    onChange={(e) => updateParam(selectedModel, 'learning_rate', parseFloat(e.target.value))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
              </div>
            )}

            {/* LoRA Parameters */}
            {selectedModel === 'roberta_lora' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    LoRA Rank
                  </label>
                  <input
                    type="number"
                    value={trainParams[selectedModel]?.lora_rank || 8}
                    onChange={(e) => updateParam(selectedModel, 'lora_rank', parseInt(e.target.value))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    LoRA Alpha
                  </label>
                  <input
                    type="number"
                    value={trainParams[selectedModel]?.lora_alpha || 16}
                    onChange={(e) => updateParam(selectedModel, 'lora_alpha', parseInt(e.target.value))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
              </div>
            )}

            {/* Hybrid Parameters */}
            {selectedModel === 'hybrid' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Hidden Layers (用逗號分隔，例如: 128,64)
                </label>
                <input
                  type="text"
                  value={trainParams[selectedModel]?.hidden_layers?.join(',') || '128,64'}
                  onChange={(e) => updateParam(selectedModel, 'hidden_layers', e.target.value.split(',').map(Number))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                />
              </div>
            )}

            <button
              onClick={handleTrain}
              disabled={training || trainingStatus?.is_training}
              className="w-full mt-6 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {training || trainingStatus?.is_training ? '訓練中...' : '開始訓練'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

