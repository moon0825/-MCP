"""엔드투엔드 스모크 테스트: 화자분리 → 전사 → 화자 학습 → 재인식.

4인 대화 샘플(sherpa-onnx 공개 테스트 오디오)로:
1) 1차 전사 — 등록 화자가 없으니 전원 미확정이어야 함
2) 화자1을 '테스트화자A'로 확정(학습)
3) 같은 오디오 2차 전사 — 화자 중 하나가 '테스트화자A'로 제안되어야 함 (핵심 기능 검증)

로컬 whisper(tiny)로 ASR까지 포함해 파이프라인 전체를 검증한다.
"""
import os
import sys
import tempfile
from pathlib import Path

TEST_DATA = Path(os.environ.get("SMOKE_DATA_DIR", tempfile.mkdtemp(prefix="scribe-smoke-")))
os.environ["MEETING_SCRIBE_DATA"] = str(TEST_DATA / "data")
os.environ.setdefault("MEETING_SCRIBE_ASR", "local")
os.environ.setdefault("MEETING_SCRIBE_WHISPER_MODEL", "tiny")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_scribe import db, jobs  # noqa: E402
from meeting_scribe.config import ensure_dirs  # noqa: E402
from meeting_scribe.pipeline import run_job, track_name_from_file  # noqa: E402

WAV_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-segmentation-models/0-four-speakers-zh.wav")


def fetch_test_wav() -> Path:
    import httpx
    wav = TEST_DATA / "four-speakers.wav"
    if not wav.exists():
        print(f"테스트 오디오 다운로드: {WAV_URL}")
        with httpx.stream("GET", WAV_URL, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(wav, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
    return wav


def run_sync(kind: str, path: str, project_id, options) -> str:
    job_id = jobs.enqueue(kind, path, project_id, options)
    run_job(job_id)  # 테스트에서는 워커 대신 동기 실행
    job = db.get_job(job_id)
    assert job["status"] == "done", f"작업 실패: {job['error']}"
    return job_id


def main() -> None:
    ensure_dirs()
    db.init_db()

    # 유닛 체크: Zoom 트랙 파일명 → 화자명
    assert track_name_from_file(Path("audio_only_김대리_1.m4a")) == "김대리"
    assert track_name_from_file(Path("audioPark2.m4a")) == "Park"
    print("✓ Zoom 트랙 이름 파싱")

    pid = db.upsert_project("smoke-test", "스모크 테스트 프로젝트")
    db.add_glossary_terms(pid, [{"term": "테스트용어", "note": "검증용"}])
    db.add_participants(pid, [{"name": "테스트화자A", "role": "발표자"}])
    assert db.hotwords_for_project(pid, 10) == ["테스트화자A", "테스트용어"]
    print("✓ 프로젝트/용어집/참석자")

    wav = fetch_test_wav()

    # 1차: 등록 화자 없음 → 제안도 없어야 함
    job1 = run_sync("single", str(wav), pid, {"language": "zh", "num_speakers": 4})
    map1 = db.get_job_speaker_map(job1)
    clusters = [m["cluster_label"] for m in map1]
    assert len(clusters) >= 2, f"화자분리 실패: {clusters}"
    assert all(m["suggested_speaker_id"] is None for m in map1)
    md = (Path(db.get_job(job1)["result_dir"]) / "transcript.md").read_text(encoding="utf-8")
    assert "화자" in md and len(md) > 200
    print(f"✓ 1차 전사 완료 — 감지 화자 {len(clusters)}명, 전사본 {len(md)}자")

    # 화자 학습: 화자1 → 테스트화자A
    sid = db.upsert_speaker("테스트화자A", "smoke test consent")
    emb = db.confirm_cluster(job1, clusters[0], sid)
    assert emb is not None
    db.add_speaker_embedding(sid, emb, source=f"job:{job1}")
    print("✓ 화자 확정·보이스프린트 등록")

    # 2차: 같은 오디오 → 테스트화자A가 제안되어야 함
    job2 = run_sync("single", str(wav), pid, {"language": "zh", "num_speakers": 4})
    map2 = db.get_job_speaker_map(job2)
    suggested = {m["cluster_label"]: (m["suggested_name"], m["similarity"]) for m in map2}
    hits = [(c, n, s) for c, (n, s) in suggested.items() if n == "테스트화자A"]
    assert hits, f"화자 재인식 실패: {suggested}"
    print(f"✓ 화자 재인식 성공 — {hits[0][0]} → 테스트화자A (유사도 {hits[0][2]:.3f})")

    md2 = (Path(db.get_job(job2)["result_dir"]) / "transcript.md").read_text(encoding="utf-8")
    assert "테스트화자A(?)" in md2, "제안 화자명이 전사본에 반영되지 않음"
    print("✓ 전사본에 제안 화자명 표기")

    print(f"\n--- 2차 전사본 미리보기 ---\n{md2[:800]}")
    print("\n모든 스모크 테스트 통과 ✅")


if __name__ == "__main__":
    main()
