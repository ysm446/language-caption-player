import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_ID = "Qwen/Qwen3-1.7B"

# /no_think でthinkingモードをOFF → 字幕バッチ翻訳に最適化
SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a subtitle translator. Translate the given text to natural Japanese "
    "suitable for subtitle display. Output only the Japanese translation, nothing else."
)

LOOKUP_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a bilingual dictionary assistant. When given an English word, respond in Japanese "
    "with this exact format (no extra text):\n"
    "【品詞】名詞／動詞／形容詞 など\n"
    "【意味】日本語の意味（簡潔に）\n"
    "【例文】An example sentence. ／ 日本語訳\n"
    "If the word has multiple common meanings, list up to 2."
)


class Translator:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print(f"[Translator] Loaded {MODEL_ID}")

    def _ensure_loaded(self):
        """モデルが未ロードであればオンデマンドでロードする"""
        if self.model is None:
            self.load()

    def translate(self, text: str) -> str:
        """テキスト1件を日本語に翻訳して返す"""
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate to Japanese:\n{text}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # GenerationConfig を明示的に渡してモデルのデフォルト設定（temperature等）を上書き
        gen_config = GenerationConfig(do_sample=False, max_new_tokens=256)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )

        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # thinkingトークンが残っている場合は除去
        if "</think>" in result:
            result = result.split("</think>", 1)[-1].strip()

        return result

    def lookup(self, word: str) -> str:
        """英単語の日本語定義を生成して返す"""
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": LOOKUP_SYSTEM_PROMPT},
            {"role": "user", "content": word.strip()},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_config = GenerationConfig(do_sample=False, max_new_tokens=128)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )

        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if "</think>" in result:
            result = result.split("</think>", 1)[-1].strip()

        return result
