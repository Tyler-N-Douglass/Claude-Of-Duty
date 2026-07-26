import { defineConfig } from 'vite';
import path from 'node:path';

export default defineConfig({
  resolve: {
    alias: { '@': path.resolve(__dirname, 'src') },
  },
  server: { port: 5173, host: true },
  build: {
    target: 'esnext',
    sourcemap: true,
    chunkSizeWarningLimit: 4000,
  },
});
