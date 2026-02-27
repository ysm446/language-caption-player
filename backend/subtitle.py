from datetime import timedelta
from pathlib import Path

import srt


def _to_td(seconds: float) -> timedelta:
    return timedelta(seconds=max(0.0, float(seconds)))


def segments_to_srt(segments: list[dict]) -> str:
    """
    ASRセグメントリストを SRT 文字列に変換する。

    Args:
        segments: [{"text": str, "timestamp": (start, end)}, ...]
    """
    subtitles = []
    for i, seg in enumerate(segments, 1):
        start, end = seg["timestamp"]
        if end is None:
            end = start + 4.0  # 終端が不明な場合のフォールバック
        subtitles.append(
            srt.Subtitle(
                index=i,
                start=_to_td(start),
                end=_to_td(end),
                content=seg["text"].strip(),
            )
        )
    return srt.compose(subtitles)


def srt_file_to_segments(srt_path: str) -> list[dict]:
    """
    SRT ファイルを読み込み、セグメントリストに変換する。

    Returns:
        [{"text": str, "timestamp": (start_sec, end_sec)}, ...]
    """
    text = Path(srt_path).read_text(encoding="utf-8")
    return [
        {
            "text": sub.content,
            "timestamp": (
                sub.start.total_seconds(),
                sub.end.total_seconds(),
            ),
        }
        for sub in srt.parse(text)
    ]


def save_srt(content: str, path: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def make_output_path(video_path: str, suffix: str) -> str:
    """
    動画パスからSRT出力パスを生成する。
    例: video.mp4 → video.original.srt / video.japanese.srt
    """
    p = Path(video_path)
    return str(p.parent / f"{p.stem}.{suffix}.srt")
