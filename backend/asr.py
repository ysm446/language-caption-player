import os
import tempfile
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import ffmpeg
from qwen_asr import Qwen3ASRModel

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# SRT 用タイムスタンプの粒度（秒）
# 小さくするほど字幕が細かくなるが推論回数が増える
DEFAULT_CHUNK_SEC = 30


class ASRProcessor:
    def __init__(self):
        self.model = None

    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        print(f"[ASR] Loaded {MODEL_ID}")

    def extract_audio(self, video_path: str) -> str:
        """動画ファイルから 16kHz モノラル WAV を一時ファイルに抽出する"""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        (
            ffmpeg.input(video_path)
            .output(tmp.name, ar=16000, ac=1, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )
        return tmp.name

    def transcribe(
        self,
        video_path: str,
        language: str = None,
        chunk_sec: float = DEFAULT_CHUNK_SEC,
    ) -> list[dict]:
        """
        動画を文字起こしし、タイムスタンプ付きセグメントを返す。

        音声を chunk_sec 秒ごとに分割し、各チャンクを個別に推論することで
        SRT 用のタイムスタンプを生成する。

        Args:
            video_path: 動画ファイルのパス
            language: 言語コード（"en"/"zh"/"ja" など）。None で自動検出
            chunk_sec: 1 セグメントの長さ（秒）

        Returns:
            [{"text": str, "timestamp": (start_sec, end_sec)}, ...]
        """
        audio_path = self.extract_audio(video_path)
        try:
            data, sr = sf.read(audio_path, dtype="float32")
        finally:
            os.unlink(audio_path)

        # モノラル保証
        if data.ndim > 1:
            data = data.mean(axis=1)

        chunk_samples = int(chunk_sec * sr)
        segments = []

        for start_sample in range(0, len(data), chunk_samples):
            chunk = data[start_sample : start_sample + chunk_samples]

            # 0.5 秒未満の端切れは無音とみなしてスキップ
            if len(chunk) < sr * 0.5:
                continue

            start_sec = start_sample / sr
            end_sec = min(start_sample + chunk_samples, len(data)) / sr

            result = self.model.transcribe(
                (chunk, sr),
                language=language,
            )
            text = result[0].text.strip() if result else ""

            if text:
                segments.append({"text": text, "timestamp": (start_sec, end_sec)})

        return segments
