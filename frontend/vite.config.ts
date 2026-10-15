import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The browser only ever talks to this dev server; /api is forwarded to the backend,
// so there is no cross-origin request (no CORS setup needed). Override the target
// with API_TARGET, e.g. API_TARGET=http://127.0.0.1:8001 npm run dev
const apiTarget = process.env.API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
});
