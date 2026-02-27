"""
バックエンド API の動作テストスクリプト
使い方: python test_api.py <動画ファイルパス>
"""

import sys
import json
import requests

BASE_URL = "http://127.0.0.1:8765"


def check_health():
    print("=== /health ===")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        print(r.json())
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] バックエンドに接続できません。")
        print(f"  先に別ターミナルで 'python run_backend.py' を起動してください。")
        sys.exit(1)
    print()


def transcribe(video_path: str) -> str | None:
    print("=== /transcribe ===")
    print(f"動画: {video_path}")
    print("文字起こし中... (SSE 進捗)")

    srt_path = None
    with requests.post(
        f"{BASE_URL}/transcribe",
        json={"video_path": video_path, "language": None},
        stream=True,
        timeout=600,
    ) as r:
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                event = json.loads(line[6:])
                status = event.get("status")
                if status == "extracting_audio":
                    print("  [1/3] 音声を抽出中...")
                elif status == "saving_srt":
                    print(f"  [2/3] SRT を生成中... ({event.get('segments')} セグメント)")
                elif status == "done":
                    srt_path = event.get("srt_path")
                    print(f"  [3/3] 完了: {srt_path}")
                elif status == "error":
                    print(f"  [ERROR] {event.get('message')}")
    print()
    return srt_path


def translate(srt_path: str) -> str | None:
    print("=== /translate ===")
    print(f"原文 SRT: {srt_path}")
    print("日本語翻訳中... (SSE 進捗)")

    out_path = None
    with requests.post(
        f"{BASE_URL}/translate",
        json={"srt_path": srt_path},
        stream=True,
        timeout=3600,
    ) as r:
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                event = json.loads(line[6:])
                status = event.get("status")
                if status == "translating":
                    current = event.get("current", 0)
                    total = event.get("total", 1)
                    pct = int(current / total * 100) if total else 0
                    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                    print(f"\r  [{bar}] {pct:3d}% ({current}/{total})", end="", flush=True)
                elif status == "done":
                    out_path = event.get("srt_path")
                    print(f"\n  完了: {out_path}")
                elif status == "error":
                    print(f"\n  [ERROR] {event.get('message')}")
    print()
    return out_path


def show_srt(path: str, lines: int = 10):
    print(f"=== SRT 先頭 {lines} 行 ({path}) ===")
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= lines:
                    print("  ...")
                    break
                print(" ", line, end="")
    except FileNotFoundError:
        print("  ファイルが見つかりません")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python test_api.py <動画ファイルパス>")
        print("例:     python test_api.py C:/Videos/test.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    check_health()

    # Step 1: 文字起こし
    srt_path = transcribe(video_path)
    if not srt_path:
        print("文字起こしに失敗しました。")
        sys.exit(1)

    show_srt(srt_path)

    # Step 2: 日本語翻訳
    answer = input("続けて日本語翻訳を実行しますか？ [y/N]: ").strip().lower()
    if answer == "y":
        jp_path = translate(srt_path)
        if jp_path:
            show_srt(jp_path)
