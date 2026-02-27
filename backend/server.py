import os
import json
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# モデルロード前に HF_HOME を設定
os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "models")

from .asr import ASRProcessor
from .translator import Translator
from .subtitle import segments_to_srt, srt_file_to_segments, save_srt, make_output_path

asr = ASRProcessor()
translator = Translator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("モデルをロード中...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, asr.load)
    await loop.run_in_executor(None, translator.load)
    print("全モデルの準備完了")
    yield


app = FastAPI(title="Language Caption Player API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse(data: dict) -> str:
    """Server-Sent Events 形式にシリアライズ"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------- リクエストモデル ----------

class TranscribeRequest(BaseModel):
    video_path: str
    language: Optional[str] = None  # "en" / "zh" / "ko" など。None で自動検出


class TranslateRequest(BaseModel):
    srt_path: str  # 翻訳対象の SRT ファイルパス（通常は .original.srt）


# ---------- エンドポイント ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    """
    動画を文字起こしして original SRT を生成する。
    進捗は SSE でストリーミング。

    Events:
      {"status": "extracting_audio"}
      {"status": "transcribing"}
      {"status": "saving_srt", "segments": int}
      {"status": "done", "srt_path": str, "segments": int}
      {"status": "error", "message": str}
    """
    video_path = Path(req.video_path)
    if not video_path.exists():
        raise HTTPException(404, f"動画ファイルが見つかりません: {video_path}")

    async def stream():
        loop = asyncio.get_event_loop()

        yield sse({"status": "extracting_audio"})
        try:
            segments = await loop.run_in_executor(
                None, asr.transcribe, str(video_path), req.language
            )
        except Exception as e:
            yield sse({"status": "error", "message": str(e)})
            return

        yield sse({"status": "saving_srt", "segments": len(segments)})

        srt_content = segments_to_srt(segments)
        out_path = make_output_path(str(video_path), "original")
        save_srt(srt_content, out_path)

        yield sse({"status": "done", "srt_path": out_path, "segments": len(segments)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/translate")
async def translate(req: TranslateRequest):
    """
    原文 SRT を日本語に翻訳して japanese SRT を生成する。
    セグメントごとに進捗を SSE でストリーミング。

    Events:
      {"status": "translating", "current": int, "total": int}
      {"status": "done", "srt_path": str, "total": int}
      {"status": "error", "message": str}
    """
    srt_path = Path(req.srt_path)
    if not srt_path.exists():
        raise HTTPException(404, f"SRT ファイルが見つかりません: {srt_path}")

    segments = srt_file_to_segments(str(srt_path))
    total = len(segments)

    async def stream():
        loop = asyncio.get_event_loop()
        translated = []

        yield sse({"status": "translating", "current": 0, "total": total})

        for i, seg in enumerate(segments):
            try:
                jp_text = await loop.run_in_executor(
                    None, translator.translate, seg["text"]
                )
            except Exception as e:
                yield sse({"status": "error", "message": str(e)})
                return

            translated.append({**seg, "text": jp_text})
            yield sse({"status": "translating", "current": i + 1, "total": total})

        # japanese.srt を保存
        # .original.srt → .japanese.srt、それ以外は .japanese.srt を付加
        stem = srt_path.name
        if stem.endswith(".original.srt"):
            out_name = stem.replace(".original.srt", ".japanese.srt")
        else:
            out_name = srt_path.stem + ".japanese.srt"
        out_path = str(srt_path.parent / out_name)

        save_srt(segments_to_srt(translated), out_path)
        yield sse({"status": "done", "srt_path": out_path, "total": total})

    return StreamingResponse(stream(), media_type="text/event-stream")
