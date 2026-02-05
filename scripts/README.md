# scripts

這裡放常用指令，分為：
- `scripts/windows/*.ps1`
- `scripts/linux/*.sh`

原則：
- **不要把 secret 寫進檔案**（例如 `OPENAI_API_KEY`），用環境變數或 CI secret。
- `config.yaml` 不會進 git，請用 `config.example.yaml` 複製一份成 `config.yaml`。

