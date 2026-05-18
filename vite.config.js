import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // /demo altbase path — geliştirmede "/" yeterli
  base: "/",
  build: { outDir: "dist" },
  server: {
    // Geliştirmede serverless fonksiyonları proxy'le
    proxy: { "/api": "http://localhost:3000" }
  }
});
