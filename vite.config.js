import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  resolve: {
    alias: [
      {
        find: /^three$/,
        replacement: "three/webgpu",
      },
    ],
  },
  optimizeDeps: {
    esbuildOptions: {
      target: "esnext",
    },
  },
  build: {
    target: "esnext",
    chunkSizeWarningLimit: 1500,
  },
  assetsInclude: ["**/*.glb", "**/*.bin", "**/*.hdr"],
});
