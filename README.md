# Language Caption Player

動画から字幕を自動生成し、2言語同時再生できるデスクトップアプリです。

## 機能

- **字幕生成**: Qwen3-ASR-1.7B で動画を文字起こし → SRT ファイル出力
- **日本語翻訳**: Qwen3-1.7B で原文字幕を日本語に翻訳 → 日本語 SRT 出力
- **2言語プレイヤー**: 原文と日本語訳を同時表示（Electron）
- **単語 hover**: 単語にカーソルを乗せると辞書・発音などのエフェクトを表示（予定）

## 必要環境

| ソフトウェア | バージョン |
|---|---|
| Python | 3.10 以上 |
| conda | 任意のバージョン |
| ffmpeg | システムにインストール済みであること |
| CUDA（任意） | GPU 推論を使う場合 |

> ffmpeg のインストール: https://ffmpeg.org/download.html
> インストール後、`ffmpeg -version` でパスが通っていることを確認してください。

## セットアップ

### 1. conda 環境を作成

```bash
conda create -n main python=3.11
conda activate main
```

### 2. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

> `transformers` は Qwen3-ASR 対応のため GitHub 開発版が必要です。
> `requirements.txt` に記載済みのため、上記コマンドで自動的にインストールされます。

### 3. モデルをダウンロード

```bash
# HF_HOME を models/ に向けて実行
set HF_HOME=./models

python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='Qwen/Qwen3-ASR-1.7B')"
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B')"
```

> モデルは `models/hub/` 以下に保存されます（gitignore 済み）。

## 起動方法

### バックエンドのみ（Python FastAPI）

```bash
start.bat
```

または手動で：

```bash
conda activate main
python run_backend.py
```

起動後、`http://127.0.0.1:8765/health` で `{"status":"ok"}` が返れば準備完了です。

## API エンドポイント

### `GET /health`
サーバーの起動確認。

### `POST /transcribe`
動画を文字起こしして `.original.srt` を生成する。

```json
{
  "video_path": "C:/path/to/video.mp4",
  "language": null
}
```

- `language`: `"en"` / `"zh"` / `"ko"` など。`null` で自動検出。
- 進捗は **SSE（Server-Sent Events）** でストリーミングされる。

### `POST /translate`
`.original.srt` を日本語に翻訳して `.japanese.srt` を生成する。

```json
{
  "srt_path": "C:/path/to/video.original.srt"
}
```

- セグメントごとに翻訳し、進捗を SSE でストリーミングする。

## 出力ファイル

```
動画ファイル: movie.mp4
  → movie.original.srt    # Qwen3-ASR で生成した原文字幕
  → movie.japanese.srt    # Qwen3 で翻訳した日本語字幕
```

## プロジェクト構成

```
language-caption-player/
├── models/                  # HuggingFace モデルキャッシュ（gitignore）
├── backend/
│   ├── asr.py               # Qwen3-ASR-1.7B 推論
│   ├── translator.py        # Qwen3-1.7B 翻訳
│   ├── subtitle.py          # SRT 生成・読み込みユーティリティ
│   └── server.py            # FastAPI サーバー
├── frontend/                # Electron UI（今後実装）
├── run_backend.py           # uvicorn 起動エントリーポイント
├── start.bat                # Windows 起動スクリプト
└── requirements.txt
```

## 使用モデル

| モデル | 用途 |
|---|---|
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 音声認識（多言語対応） |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 日本語翻訳 |
