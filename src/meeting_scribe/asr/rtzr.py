"""ReturnZero(RTZR/VITO) STT 어댑터 — 한국어 특화, 키워드 부스팅(최대 100개), 단어 타임스탬프.
https://developers.rtzr.ai/docs/ 참고. 무료 10시간, 이후 약 ₩1,000/시간."""
import os
import time
from pathlib import Path

import httpx

from .base import AsrAdapter, AsrSegment, AsrWord

BASE = "https://openapi.vito.ai"


class RtzrAdapter(AsrAdapter):
    name = "rtzr"
    word_timestamps = True

    def __init__(self):
        self.client_id = os.environ["RTZR_CLIENT_ID"]
        self.client_secret = os.environ["RTZR_CLIENT_SECRET"]

    def _token(self, client: httpx.Client) -> str:
        resp = client.post(f"{BASE}/v1/authenticate",
                           data={"client_id": self.client_id, "client_secret": self.client_secret})
        resp.raise_for_status()
        return resp.json()["access_token"]

    def transcribe(self, wav_path: Path, language: str,
                   hotwords: list[str], context_hint: str = "") -> list[AsrSegment]:
        config = {
            "model_name": "sommers",
            "use_itn": True,
            "use_disfluency_filter": False,
            "use_paragraph_splitter": True,
            "keywords": hotwords[:100],
        }
        import json as _json
        with httpx.Client(timeout=300) as client:
            token = self._token(client)
            headers = {"Authorization": f"Bearer {token}"}
            with open(wav_path, "rb") as f:
                resp = client.post(
                    f"{BASE}/v1/transcribe", headers=headers,
                    files={"file": (wav_path.name, f, "audio/wav")},
                    data={"config": _json.dumps(config, ensure_ascii=False)},
                )
            resp.raise_for_status()
            job_id = resp.json()["id"]
            for _ in range(720):  # 최대 1시간 폴링
                time.sleep(5)
                r = client.get(f"{BASE}/v1/transcribe/{job_id}", headers=headers)
                r.raise_for_status()
                body = r.json()
                if body.get("status") == "completed":
                    return _parse(body)
                if body.get("status") == "failed":
                    raise RuntimeError(f"RTZR 전사 실패: {body}")
        raise TimeoutError("RTZR 전사 폴링 시간 초과")


def _parse(body: dict) -> list[AsrSegment]:
    segments: list[AsrSegment] = []
    for utt in body.get("results", {}).get("utterances", []):
        start = float(utt.get("start_at", 0)) / 1000.0
        dur = float(utt.get("duration", 0)) / 1000.0
        words = [
            AsrWord(w.get("text", "").strip(),
                    float(w.get("start_at", 0)) / 1000.0,
                    (float(w.get("start_at", 0)) + float(w.get("duration", 0))) / 1000.0)
            for w in utt.get("words", []) or []
        ]
        segments.append(AsrSegment(text=utt.get("msg", "").strip(),
                                   start=start, end=start + dur, words=words))
    return segments
