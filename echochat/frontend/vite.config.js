import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === "build" ? "/static/" : "/",
  server: {
    port: 3000,
    strictPort: true,
    proxy: {
      "/upload": "http://127.0.0.1:5000",
      "/chat": "http://127.0.0.1:5000",
      "/session": "http://127.0.0.1:5000",
      "/train": "http://127.0.0.1:5000",
      "/health": "http://127.0.0.1:5000"
    }
  }
}));
