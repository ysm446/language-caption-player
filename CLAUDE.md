# CLAUDE.md

Claude Code がこのプロジェクトで作業する際の参照ドキュメント。

## プロジェクト概要

動画から字幕を自動生成し、2言語同時再生するデスクトップアプリ。

- **バックエンド**: Python + FastAPI（ポート 8765）
- **フロントエンド**: Electron（今後実装）
- **モデル**: Qwen3-ASR-1.7B（ASR）、Qwen3-1.7B（日本語翻訳）

## 環境

- Python: conda 環境名 `main`
- OS: Windows 11
- Shell: bash（Unix 構文を使う）
- GPU: CUDA があれば使用、なければ CPU にフォールバック

## 重要なパス

| パス | 説明 |
|---|---|
| `models/hub/` | HuggingFace モデルキャッシュ（`HF_HOME=./models`） |
| `backend/` | Python バックエンドパッケージ |
| `run_backend.py` | uvicorn 起動エントリーポイント |
| `start.bat` | Windows 起動スクリプト |

## モデル

| モデルID | 用途 | ローカルパス |
|---|---|---|
| `Qwen/Qwen3-ASR-1.7B` | 音声認識 | `models/hub/models--Qwen--Qwen3-ASR-1.7B/` |
| `Qwen/Qwen3-1.7B` | 日本語翻訳 | `models/hub/models--Qwen--Qwen3-1.7B/` |

## アーキテクチャ

```
動画ファイル
  → [backend/asr.py] ffmpeg で音声抽出 → Qwen3-ASR → セグメント
  → [backend/subtitle.py] SRT 生成 → video.original.srt

video.original.srt
  → [backend/translator.py] Qwen3-1.7B（thinking OFF）→ 日本語テキスト
  → [backend/subtitle.py] SRT 生成 → video.japanese.srt

FastAPI SSE でフロントエンドに進捗をストリーミング
```

## API エンドポイント

- `GET  /health` — 起動確認
- `POST /transcribe` — 動画→原文SRT（SSE）
- `POST /translate`  — 原文SRT→日本語SRT（SSE）

## 依存パッケージの注意点

- `qwen-asr`（公式パッケージ）を使用するため `transformers==4.57.6` に固定
- `qwen-asr` は conda 環境にインストール済み（`pip install qwen-asr`）
- `transformers` を 5.x 系（dev版含む）にアップグレードすると `qwen-asr` が壊れる
- `asr.py` は `transformers.pipeline` ではなく `qwen_asr.Qwen3ASRModel` を使用

## 翻訳の実装方針

- Qwen3-1.7B をチャットUIではなくバッチ翻訳用途で使用
- system prompt に `/no_think` を付与して thinking モードを無効化
- セグメント1件ずつ翻訳し、SSE で進捗をフロントエンドへ通知

## 出力ファイル命名規則

```
video.mp4 → video.original.srt  （ASR生成）
          → video.japanese.srt  （翻訳生成）
```

## 今後の実装予定

- [ ] Electron フロントエンド
  - 動画プレイヤー（HTML5 video）
  - 2言語字幕オーバーレイ
  - 単語ごと `<span>` 分割 + hover エフェクト（辞書・ハイライト等）
- [ ] 字幕生成ページ UI（動画選択・進捗バー・言語選択）
