import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // dev 단계에선 `NEXT_PUBLIC_API_BASE`로 프론트엔드가 FastAPI(8000)에 직접 fetch.
  // Next.js dev proxy + 큰 multipart streaming 조합이 ECONNRESET을 일으키는 케이스를
  // 회피. CORS는 FastAPI에서 localhost:3000 allowlist. prod 동일 origin 배포 시엔
  // base URL 비워두면 relative path가 그대로 동작.
};

export default nextConfig;
