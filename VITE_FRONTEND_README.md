# 🎉 Vite 前端已建立完成！

## 📍 位置

新的 Vite 前端位於：`frontend-vite/`

原有的 Next.js 前端仍在：`frontend/`（可保留作為備份）

## 🚀 快速開始

### 1. 進入 Vite 前端目錄

```bash
cd frontend-vite
```

### 2. 安裝依賴

```bash
npm install
```

### 3. 啟動開發伺服器

```bash
npm run dev
```

前端將在 **http://localhost:3000** 啟動

## ✨ Vite 的優勢

### ⚡ 更快的啟動速度
- Next.js: 通常需要 5-10 秒
- Vite: 通常只需 1-2 秒

### 🔥 即時熱更新
- 修改代碼後立即看到效果
- 不需要等待重新編譯

### 📦 更小的打包體積
- 更好的 tree-shaking
- 更優化的代碼分割

### 🛠️ 更簡單的配置
- 開箱即用
- 配置更直觀

## 🔄 主要變更

### 路由系統
- **Next.js**: 使用文件系統路由（`pages/` 目錄）
- **Vite**: 使用 React Router（`src/App.tsx` 中定義）

### 環境變數
- **Next.js**: `process.env.NEXT_PUBLIC_*`
- **Vite**: `import.meta.env.VITE_*`

### 構建工具
- **Next.js**: 自帶構建系統
- **Vite**: 使用 Vite（基於 Rollup）

## 📁 檔案對應關係

| Next.js | Vite |
|---------|------|
| `pages/auth/login.tsx` | `src/pages/auth/Login.tsx` |
| `pages/inference/index.tsx` | `src/pages/inference/Inference.tsx` |
| `pages/dashboard/models.tsx` | `src/pages/dashboard/ModelsDashboard.tsx` |
| `pages/_app.tsx` | `src/App.tsx` |
| `styles/globals.css` | `src/index.css` |

## 🎯 功能對比

所有功能都已完整遷移：

- ✅ 登入系統（Admin/User）
- ✅ 文本偵測頁面
- ✅ 模型管理頁面（Admin）
- ✅ 圖表視覺化
- ✅ 訓練進度顯示
- ✅ 響應式設計

## 🔧 配置說明

### API 連接

在 `vite.config.ts` 中已配置代理：

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',  // 推論 API
      changeOrigin: true,
    },
    '/train-api': {
      target: 'http://localhost:8001',  // 訓練 API
      changeOrigin: true,
    }
  }
}
```

### 環境變數

創建 `.env` 文件（可選）：

```env
VITE_API_URL=http://localhost:8000
VITE_TRAIN_API_URL=http://localhost:8001
```

## 📝 使用方式

### 開發模式

```bash
cd frontend-vite
npm run dev
```

### 生產構建

```bash
npm run build
```

構建後的檔案會在 `dist/` 目錄

### 預覽生產版本

```bash
npm run preview
```

## 🆚 選擇哪個版本？

### 使用 Vite 版本（推薦）如果：
- ✅ 想要更快的開發體驗
- ✅ 不需要 SSR（服務端渲染）
- ✅ 想要更簡單的配置
- ✅ 專注於 SPA（單頁應用）

### 使用 Next.js 版本如果：
- ✅ 需要 SSR/SSG
- ✅ 需要文件系統路由
- ✅ 需要 Next.js 生態系統

## 💡 建議

**建議使用 Vite 版本**，因為：
1. 啟動速度更快
2. 開發體驗更好
3. 對於這個專案，不需要 SSR
4. 配置更簡單

## 🐛 問題排查

### 端口被占用

如果 3000 端口被占用，Vite 會自動使用下一個可用端口。

### API 連接失敗

確保後端 API 正在運行：
- 推論 API: http://localhost:8000
- 訓練 API: http://localhost:8001

### 依賴安裝失敗

嘗試清除快取：
```bash
rm -rf node_modules package-lock.json
npm install
```

## 📚 更多資訊

- [Vite 官方文檔](https://vitejs.dev/)
- [React Router 文檔](https://reactrouter.com/)
- [TailwindCSS 文檔](https://tailwindcss.com/)

