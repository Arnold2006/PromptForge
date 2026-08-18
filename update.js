module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "git pull",
          "uv pip install -r app/requirements.txt --upgrade",
        ],
      },
    },
  ],
};
