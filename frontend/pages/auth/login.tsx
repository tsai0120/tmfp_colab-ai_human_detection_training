import { useState } from 'react'
import { useRouter } from 'next/router'

export default function Login({ setUser }: any) {
  const router = useRouter()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    // 簡單的認證邏輯（生產環境應使用 JWT）
    const users = {
      admin: { password: 'admin123', role: 'admin' },
      user: { password: 'user123', role: 'user' }
    }

    if (users[username as keyof typeof users]?.password === password) {
      const userData = {
        username,
        role: users[username as keyof typeof users].role
      }
      
      localStorage.setItem('user', JSON.stringify(userData))
      localStorage.setItem('token', `token_${username}_${Date.now()}`)
      setUser(userData)

      // 根據角色導向不同頁面
      if (userData.role === 'admin') {
        router.push('/dashboard/models')
      } else {
        router.push('/inference')
      }
    } else {
      setError('使用者名稱或密碼錯誤')
    }

    setLoading(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="bg-white p-8 rounded-lg shadow-xl w-full max-w-md">
        <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">
          AI vs Human 文本偵測平台
        </h1>
        <h2 className="text-xl font-semibold text-center mb-8 text-gray-600">
          登入系統
        </h2>

        <form onSubmit={handleLogin} className="space-y-6">
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
              使用者名稱
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="admin 或 user"
              required
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
              密碼
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="admin123 或 user123"
              required
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {loading ? '登入中...' : '登入'}
          </button>
        </form>

        <div className="mt-6 text-sm text-gray-600">
          <p className="font-semibold mb-2">測試帳號：</p>
          <p>Admin: admin / admin123</p>
          <p>User: user / user123</p>
        </div>
      </div>
    </div>
  )
}

