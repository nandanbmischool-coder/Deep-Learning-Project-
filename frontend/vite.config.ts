/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const backend = env.VITE_BACKEND_URL || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: { port: 5173, proxy: { "/api": { target: backend, changeOrigin: false } } },
    test: { environment: "jsdom", globals: true, setupFiles: "./src/test-setup.ts", css: false },
  };
});
