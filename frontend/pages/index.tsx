import { useEffect } from 'react'
import { useRouter } from 'next/router'

export default function Home() {
  const router = useRouter()

  useEffect(() => {
    // 檢查是否已登入
    const user = localStorage.getItem('user')
    if (user) {
      const userData = JSON.parse(user)
      if (userData.role === 'admin') {
        router.push('/dashboard/models')
      } else {
        router.push('/inference')
      }
    } else {
      router.push('/auth/login')
    }
  }, [])

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-2xl font-bold mb-4">載入中...</h1>
      </div>
    </div>
  )
}

