import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import Login from './pages/auth/Login'
import Inference from './pages/inference/Inference'
import ModelsDashboard from './pages/dashboard/ModelsDashboard'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const TRAIN_API_URL = import.meta.env.VITE_TRAIN_API_URL || 'http://localhost:8001'

// 導出 API URLs 供其他組件使用
export { API_URL, TRAIN_API_URL }

function App() {
  const [user, setUser] = useState<{ role: string; username: string } | null>(null)

  useEffect(() => {
    // 檢查本地儲存的用戶資訊
    const storedUser = localStorage.getItem('user')
    if (storedUser) {
      setUser(JSON.parse(storedUser))
    }
  }, [])

  const logout = () => {
    localStorage.removeItem('user')
    localStorage.removeItem('token')
    setUser(null)
  }

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Routes>
          <Route path="/auth/login" element={<Login setUser={setUser} />} />
          <Route
            path="/inference"
            element={
              <ProtectedRoute user={user}>
                <Inference user={user} logout={logout} />
              </ProtectedRoute>
            }
          />
          <Route
            path="/dashboard/models"
            element={
              <AdminRoute user={user}>
                <ModelsDashboard user={user} logout={logout} />
              </AdminRoute>
            }
          />
          <Route path="/" element={<Navigate to="/auth/login" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

// 保護路由：需要登入
function ProtectedRoute({ children, user }: { children: React.ReactElement; user: any }) {
  if (!user) {
    return <Navigate to="/auth/login" replace />
  }
  return children
}

// Admin 路由：需要 Admin 權限
function AdminRoute({ children, user }: { children: React.ReactElement; user: any }) {
  if (!user) {
    return <Navigate to="/auth/login" replace />
  }
  if (user.role !== 'admin') {
    return <Navigate to="/inference" replace />
  }
  return children
}

export default App

