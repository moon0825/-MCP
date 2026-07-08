"""경로·환경변수 설정. 모든 데이터는 데이터 디렉터리(SQLite/모델/작업 결과) 아래에만 저장한다."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("MEETING_SCRIBE_DATA", Path.home() / ".meeting-scribe"))
MODELS_DIR = DATA_DIR / "models"
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "scribe.db"

# 화자 임베딩 코사인 유사도 제안 임계값 (자동 확정 아님 — 제안만)
SPEAKER_SUGGEST_THRESHOLD = float(os.environ.get("MEETING_SCRIBE_SPK_THRESHOLD", "0.55"))
# 클러스터링 임계값 (작을수록 화자를 더 잘게 나눔)
CLUSTER_THRESHOLD = float(os.environ.get("MEETING_SCRIBE_CLUSTER_THRESHOLD", "0.8"))
# ASR 부스팅에 넣을 최대 용어 수 (과다 부스팅은 역효과)
MAX_HOTWORDS = int(os.environ.get("MEETING_SCRIBE_MAX_HOTWORDS", "80"))
# 이 값보다 낮은 avg_logprob 세그먼트는 [?] 불확실 표시
LOW_CONFIDENCE_LOGPROB = float(os.environ.get("MEETING_SCRIBE_LOW_CONF", "-0.8"))

DEFAULT_LANGUAGE = os.environ.get("MEETING_SCRIBE_LANGUAGE", "ko")


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)
