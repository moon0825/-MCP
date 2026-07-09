"""경로·환경변수 설정. 모든 데이터는 데이터 디렉터리(SQLite/모델/작업 결과) 아래에만 저장한다."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("MEETING_SCRIBE_DATA", Path.home() / ".meeting-scribe"))
MODELS_DIR = DATA_DIR / "models"
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "scribe.db"

# 화자 임베딩 코사인 유사도 제안 임계값 (자동 확정 아님 — 제안만)
SPEAKER_SUGGEST_THRESHOLD = float(os.environ.get("MEETING_SCRIBE_SPK_THRESHOLD", "0.55"))
# 클러스터링 임계값 (작을수록 화자를 더 잘게 나눔).
# 폰 원거리 실녹음 116분 스윕 결과: 0.8=107클러스터(과분할), 1.0=43, 1.1=18, 1.2=과병합 경향.
# 화자 수를 알면 num_speakers 지정이 항상 더 정확하다 (임계값 무시됨).
CLUSTER_THRESHOLD = float(os.environ.get("MEETING_SCRIBE_CLUSTER_THRESHOLD", "1.1"))
# ASR 부스팅에 넣을 최대 용어 수 (과다 부스팅은 역효과)
MAX_HOTWORDS = int(os.environ.get("MEETING_SCRIBE_MAX_HOTWORDS", "80"))
# 이 값보다 낮은 avg_logprob 세그먼트는 [?] 불확실 표시
LOW_CONFIDENCE_LOGPROB = float(os.environ.get("MEETING_SCRIBE_LOW_CONF", "-0.8"))
# 재시작 복구 시 'running' 작업을 재큐잉하는 기준: updated_at이 이 시간 이상 오래되면
# 소유 프로세스가 죽었다고 판정. 살아있는 다른 프로세스의 작업을 뺏어 이중 실행하는 것 방지.
STALE_RUNNING_SECONDS = int(os.environ.get("MEETING_SCRIBE_STALE_RUNNING_SEC", "600"))

DEFAULT_LANGUAGE = os.environ.get("MEETING_SCRIBE_LANGUAGE", "ko")


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)
