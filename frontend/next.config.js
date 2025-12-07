/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    API_URL: process.env.API_URL || 'http://localhost:8000',
    TRAIN_API_URL: process.env.TRAIN_API_URL || 'http://localhost:8001',
  },
}

module.exports = nextConfig

