module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    // Create and populate the Python virtual environment
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r app/requirements.txt",
          "python app/llama_setup.py",
        ],
      },
    },
    // Download a vision-capable default model (~2.5 GB): Huihui-Qwen3-VL-4B
    // Instruct abliterated. Lets the app steer prompt generation from an
    // uploaded reference image out of the box.
    {
      method: "hf.download",
      params: {
        "_": ["noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF", "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf"],
        "local-dir": "models",
      },
    },
    // Download the matching vision projector (~836 MB). llama-server loads
    // this alongside the model (via --mmproj) to enable image inputs.
    {
      method: "hf.download",
      params: {
        "_": ["noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF", "mmproj-F16.gguf"],
        "local-dir": "models",
      },
    },
  ],
};
