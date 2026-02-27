import os
from pathlib import Path

# モデルの保存先を models/ フォルダに強制設定（既存の環境変数を上書き）
os.environ["HF_HOME"] = str(Path(__file__).parent / "models")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )
