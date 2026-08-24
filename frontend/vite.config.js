import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
export default defineConfig({
    plugins: [react(), tailwindcss()],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
    server: {
        host: true,
        proxy: {
            "/api": {
                // Native `npm run dev` talks to localhost:8000; inside docker-compose
                // the frontend and backend are separate containers, so compose sets
                // this to the backend service's DNS name instead.
                target: process.env.VITE_API_TARGET || "http://localhost:8000",
                changeOrigin: true,
            },
        },
    },
});
