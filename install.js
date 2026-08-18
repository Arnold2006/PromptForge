module.exports = {
  run: [
    // Create and populate the Python virtual environment
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r app/requirements.txt",
        ],
      },
    },
  ],
};
