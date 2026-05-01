import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        // 127.0.0.1, not localhost — on Windows, localhost resolves to both
        // ::1 (IPv6) and 127.0.0.1, and uvicorn binds IPv4 only by default.
        // The IPv6 attempt fails noisily (AggregateError [ECONNREFUSED])
        // even though IPv4 succeeds.
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
