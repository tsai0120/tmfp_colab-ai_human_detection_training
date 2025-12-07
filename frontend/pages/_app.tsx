import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'

export default function App({ Component, pageProps }: AppProps) {
  const router = useRouter()
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
    router.push('/auth/login')
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Component {...pageProps} user={user} setUser={setUser} logout={logout} />
    </div>
  )
}

