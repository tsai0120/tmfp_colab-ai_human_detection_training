import { useState } from 'react'
import { useRouter } from 'next/router'
import axios from 'axios'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  ArcElement
} from 'chart.js'
import { Line, Bar, Radar } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  ArcElement
)

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Inference({ user, logout }: any) {
  const router = useRouter()
  const [text, setText] = useState('')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // 如果未登入，導向登入頁
  if (!user) {
    router.push('/auth/login')
    return null
  }

  const handleDetect = async () => {
    if (!text.trim()) {
      setError('請輸入文本')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        text: text.trim()
      })

      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || '偵測失敗，請稍後再試')
    } finally {
      setLoading(false)
    }
  }

  const chartData = result ? {
    labels: Object.keys(result.details),
    datasets: [
      {
        label: 'AI 機率',
        data: Object.values(result.details),
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
      },
    ],
  } : null

  const radarData = result ? {
    labels: Object.keys(result.details),
    datasets: [
      {
        label: 'AI 機率',
        data: Object.values(result.details),
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
      },
    ],
  } : null

  const getColorClass = (prob: number) => {
    if (prob > 0.7) return 'text-red-600 bg-red-50'
    if (prob > 0.5) return 'text-orange-600 bg-orange-50'
    return 'text-green-600 bg-green-50'
  }

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">文本偵測</h1>
          <div className="flex gap-4 items-center">
            <span className="text-gray-600">歡迎，{user.username}</span>
            {user.role === 'admin' && (
              <button
                onClick={() => router.push('/dashboard/models')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                模型管理
              </button>
            )}
            <button
              onClick={logout}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
            >
              登出
            </button>
          </div>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">輸入文本</h2>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full h-48 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            placeholder="請在此貼上或輸入要偵測的文本..."
          />
          <div className="mt-4 flex justify-between items-center">
            <span className="text-sm text-gray-500">字數: {text.length}</span>
            <button
              onClick={handleDetect}
              disabled={loading || !text.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? '偵測中...' : '開始偵測'}
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Main Result */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">偵測結果</h2>
              <div className="flex items-center gap-4">
                <div className={`px-6 py-4 rounded-lg ${getColorClass(result.probability_ai)}`}>
                  <div className="text-sm font-medium mb-1">判定結果</div>
                  <div className="text-2xl font-bold">{result.label}</div>
                </div>
                <div className="flex-1">
                  <div className="text-sm text-gray-600 mb-1">AI 機率</div>
                  <div className="text-3xl font-bold text-gray-800">
                    {(result.probability_ai * 100).toFixed(2)}%
                  </div>
                  <div className="text-sm text-gray-500 mt-1">
                    使用模型: {result.selected_model}
                  </div>
                </div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Bar Chart */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">各模型預測結果（長條圖）</h3>
                {chartData && <Bar data={chartData} options={{
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 1,
                    },
                  },
                }} />}
              </div>

              {/* Radar Chart */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">各模型預測結果（雷達圖）</h3>
                {radarData && <Radar data={radarData} options={{
                  scales: {
                    r: {
                      beginAtZero: true,
                      max: 1,
                    },
                  },
                }} />}
              </div>
            </div>

            {/* Line Chart */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">各模型預測結果（折線圖）</h3>
              {chartData && <Line data={chartData} options={{
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 1,
                  },
                },
              }} />}
            </div>

            {/* Details Table */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">詳細結果</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        模型
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        AI 機率
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        判定
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {Object.entries(result.details).map(([model, prob]: [string, any]) => (
                      <tr key={model}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {model.toUpperCase()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {(prob * 100).toFixed(2)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                            prob > 0.5 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                          }`}>
                            {prob > 0.5 ? 'AI' : 'Human'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

