import os
import re
import tempfile

import torch
import soundfile as sf
import ffmpeg
from qwen_asr import Qwen3ASRModel

MODEL_ID           = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_ID  = "Qwen/Qwen3-ForcedAligner-0.6B"

# ForcedAligner の最大入力長（ライブラリ定数に合わせた安全値）
MAX_ALIGN_SEC = 170


class ASRProcessor:
    def __init__(self):
        self.model = None

    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map="auto",
            forced_aligner=FORCED_ALIGNER_ID,
            forced_aligner_kwargs={"torch_dtype": dtype, "device_map": "auto"},
        )
        print(f"[ASR] Loaded {MODEL_ID} + {FORCED_ALIGNER_ID}")

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

    def _align_to_segments(
        self,
        align_result,
        offset_sec: float,
        max_words: int = 12,
    ) -> list[dict]:
        """
        ForcedAlignResult の単語アイテムを字幕セグメントにグループ化する。

        グループ区切りの優先順位:
          1. 文末句読点（. ! ?）の直後
          2. max_words 単語に達したとき

        offset_sec: このチャンクの開始時刻（秒）。各アイテムの時刻に加算する。
        """
        segments: list[dict] = []
        current_words: list[tuple[str, float, float]] = []  # (text, start, end)
        seg_start: float | None = None

        def flush():
            nonlocal seg_start
            if not current_words:
                return
            text    = " ".join(w[0] for w in current_words)
            seg_end = current_words[-1][2]
            segments.append({"text": text, "timestamp": (seg_start, seg_end)})
            current_words.clear()
            seg_start = None

        for item in align_result:
            word = item.text.strip()
            if not word:
                continue

            w_start = item.start_time + offset_sec
            w_end   = item.end_time   + offset_sec

            if seg_start is None:
                seg_start = w_start

            current_words.append((word, w_start, w_end))

            ends_sentence = bool(re.search(r"[.!?]$", word))
            if ends_sentence or len(current_words) >= max_words:
                flush()

        flush()  # 残りのワードを確定
        return segments

    def transcribe(
        self,
        video_path: str,
        language: str = None,
    ) -> list[dict]:
        """
        動画を文字起こしし、ForcedAligner による正確なタイムスタンプ付きセグメントを返す。

        音声を MAX_ALIGN_SEC 秒ごとに分割して推論し、
        各チャンクのオフセットを ForcedAlignItem の時刻に加算することで
        動画全体の絶対時刻を正確に再現する。

        Args:
            video_path: 動画ファイルのパス
            language:   言語コード（"en"/"zh"/"ja" など）。None で自動検出

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

        chunk_samples = int(MAX_ALIGN_SEC * sr)
        segments: list[dict] = []

        for start_sample in range(0, len(data), chunk_samples):
            chunk = data[start_sample : start_sample + chunk_samples]

            # 0.5 秒未満の端切れは無音とみなしてスキップ
            if len(chunk) < sr * 0.5:
                continue

            start_sec = start_sample / sr

            results = self.model.transcribe(
                (chunk, sr),
                language=language,
                return_time_stamps=True,
            )

            if not results:
                continue

            result = results[0]

            if result.time_stamps:
                # ForcedAligner が成功した場合：単語レベルのタイムスタンプを使用
                chunk_segs = self._align_to_segments(result.time_stamps, offset_sec=start_sec)
                segments.extend(chunk_segs)
            else:
                # フォールバック：ForcedAligner が失敗した場合はチャンク全体を1セグメントに
                text = result.text.strip()
                if text:
                    end_sec = min(start_sample + chunk_samples, len(data)) / sr
                    segments.append({"text": text, "timestamp": (start_sec, end_sec)})

        return segments
