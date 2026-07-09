"""네트워크·모델 없이 도는 오프라인 테스트.

검증 범위:
- merge: 단어 타임스탬프 × 화자 구간 병합 (세그먼트 유지/분할, 불확실 표시, 연속 발화 합치기)
- audio: ffmpeg 정규화 → 16kHz mono, wav 읽기, 분할
- 파이프라인 E2E (화자분리·임베딩·ASR을 가짜로 대체): 1차 전사 → 화자 학습 → 2차 전사에서
  재인식 제안 → confirm_speaker 동의 게이트 → 전사본 재생성 → save_minutes 컨텍스트 축적
- job 소유권 가드: done 작업 스킵, done을 error로 덮어쓰기 방지, 스테일 running만 재큐잉
- asr/groq: 429/5xx 재시도 (Retry-After 존중 + 지수 백오프, 가짜 클라이언트로 네트워크 없이)
- server: FastMCP 툴 등록·직접 호출

실행: .venv/bin/python tests/test_offline.py
"""
import os
import sys
import tempfile
from pathlib import Path

TEST_DATA = Path(tempfile.mkdtemp(prefix="scribe-offline-"))
os.environ["MEETING_SCRIBE_DATA"] = str(TEST_DATA / "data")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from meeting_scribe import db, jobs  # noqa: E402
from meeting_scribe.asr.base import AsrAdapter, AsrSegment, AsrWord  # noqa: E402
from meeting_scribe.config import ensure_dirs  # noqa: E402
from meeting_scribe.diarize import DiarSegment, match_speakers  # noqa: E402
from meeting_scribe.merge import merge, render_markdown  # noqa: E402


def test_merge() -> None:
    diar = [DiarSegment(0.0, 5.0, "화자1"), DiarSegment(5.2, 10.0, "화자2")]
    words_a = [AsrWord("안녕하세요", 0.5, 1.2), AsrWord("반갑습니다", 1.4, 2.2)]
    words_b = [AsrWord("네", 5.5, 5.8), AsrWord("안녕하세요", 6.0, 7.0)]
    segments = [
        AsrSegment("안녕하세요 반갑습니다", 0.5, 2.2, avg_logprob=-0.2, words=words_a),
        AsrSegment("네 안녕하세요", 5.5, 7.0, avg_logprob=-1.5, words=words_b),  # 불확실
    ]
    t = merge(segments, diar)
    assert len(t.utterances) == 2
    assert t.utterances[0].cluster == "화자1" and not t.utterances[0].uncertain
    assert t.utterances[0].text == "안녕하세요 반갑습니다"  # 한 화자 → 세그먼트 원문 유지
    assert t.utterances[1].cluster == "화자2" and t.utterances[1].uncertain

    # 한 세그먼트 안에서 화자 전환 → 단어 단위 분할
    mixed = [AsrSegment("여기까지 하고 네 알겠습니다", 3.0, 7.0, words=[
        AsrWord("여기까지", 3.0, 3.5), AsrWord("하고", 3.6, 4.9),
        AsrWord("네", 5.5, 6.0), AsrWord("알겠습니다", 6.1, 7.0)])]
    t2 = merge(mixed, diar)
    assert [u.cluster for u in t2.utterances] == ["화자1", "화자2"]
    assert t2.utterances[1].text == "네 알겠습니다"

    # 단어 없음(Gemini류) → 겹침 최대 화자 배정
    t3 = merge([AsrSegment("겹침 테스트", 4.0, 9.0)], diar)
    assert t3.utterances[0].cluster == "화자2"

    md = render_markdown(t, {"화자1": "김대리"}, "테스트", ["- 헤더"])
    assert "김대리" in md and "[?]" in md and "화자2" in md
    print("✓ merge/render")


def test_audio() -> None:
    from meeting_scribe.audio import (duration_seconds, ffmpeg_exe, normalize,
                                      read_wav_float32, split_wav)
    import subprocess

    src = TEST_DATA / "tone_44k_stereo.wav"
    subprocess.run([ffmpeg_exe(), "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=3", "-ac", "2", "-ar", "44100",
                    str(src)], capture_output=True, check=True)
    out = normalize(src, TEST_DATA / "norm.wav")
    samples, sr = read_wav_float32(out)
    assert sr == 16000 and abs(len(samples) / sr - 3.0) < 0.1
    assert 0.05 < np.abs(samples).max() <= 1.0  # ffmpeg sine 기본 진폭 ≈ 0.09
    assert abs(duration_seconds(out) - 3.0) < 0.1
    chunks = split_wav(out, 1, TEST_DATA / "chunks")
    assert len(chunks) == 3 and chunks[1][1] == 1.0
    print("✓ audio normalize/read/split")


def test_speaker_matching_math() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(size=512).astype(np.float32)
    b = rng.normal(size=512).astype(np.float32)
    enrolled = [(1, "김대리", a), (2, "박과장", b)]
    noisy_a = a + rng.normal(scale=0.1, size=512).astype(np.float32)
    res = match_speakers({"화자1": noisy_a, "화자2": rng.normal(size=512).astype(np.float32)},
                         enrolled, threshold=0.6)
    assert res["화자1"][1] == "김대리" and res["화자1"][2] > 0.9
    assert res["화자2"][0] is None  # 무관한 목소리는 매칭 안 됨
    print("✓ 임베딩 코사인 매칭")


class FakeAsr(AsrAdapter):
    name = "fake"
    word_timestamps = True

    def transcribe(self, wav_path, language, hotwords, context_hint=""):
        assert "김대리" in hotwords, "프로젝트 참석자가 hotwords에 포함되어야 함"
        return [
            AsrSegment("프로젝트 일정 공유드립니다", 0.2, 1.4, avg_logprob=-0.2,
                       words=[AsrWord("프로젝트", 0.2, 0.5), AsrWord("일정", 0.6, 0.8),
                              AsrWord("공유드립니다", 0.9, 1.4)]),
            AsrSegment("네 확인했습니다", 1.8, 2.6, avg_logprob=-0.3,
                       words=[AsrWord("네", 1.8, 2.0), AsrWord("확인했습니다", 2.1, 2.6)]),
        ]


def _fake_models(monkey_ns: dict) -> None:
    """pipeline의 모델 의존 함수를 결정적 가짜로 대체."""
    from meeting_scribe import pipeline

    rng = np.random.default_rng(7)
    emb1 = rng.normal(size=512).astype(np.float32)
    emb2 = rng.normal(size=512).astype(np.float32)

    pipeline.diarize = lambda samples, sr, num_speakers=None, **kw: [
        DiarSegment(0.0, 1.5, "화자1"), DiarSegment(1.7, 3.0, "화자2")]
    pipeline.extract_cluster_embeddings = lambda samples, sr, segs, **kw: {
        "화자1": emb1 + rng.normal(scale=0.05, size=512).astype(np.float32),
        "화자2": emb2 + rng.normal(scale=0.05, size=512).astype(np.float32)}
    pipeline.get_adapter = lambda name=None: FakeAsr()
    monkey_ns["emb1"] = emb1
    monkey_ns["emb2"] = emb2


def test_pipeline_e2e_with_fakes() -> None:
    from meeting_scribe.pipeline import run_job
    ns: dict = {}
    _fake_models(ns)

    pid = db.upsert_project("오프라인테스트", "가짜 모델 E2E")
    db.add_participants(pid, [{"name": "김대리", "role": ""}])
    db.add_glossary_terms(pid, [{"term": "무인공장", "note": ""}])

    import subprocess
    from meeting_scribe.audio import ffmpeg_exe
    wav = TEST_DATA / "meeting.wav"
    subprocess.run([ffmpeg_exe(), "-y", "-f", "lavfi", "-i",
                    "sine=frequency=300:duration=3", "-ac", "1", "-ar", "16000",
                    str(wav)], capture_output=True, check=True)

    # 1차: 등록 화자 없음 → 제안 없음
    job1 = jobs.enqueue("single", str(wav), pid, {"language": "ko"})
    run_job(job1)
    j1 = db.get_job(job1)
    assert j1["status"] == "done", j1["error"]
    map1 = db.get_job_speaker_map(job1)
    assert len(map1) == 2 and all(m["suggested_speaker_id"] is None for m in map1)
    md1 = (Path(j1["result_dir"]) / "transcript.md").read_text(encoding="utf-8")
    assert "화자1" in md1 and "프로젝트 일정 공유드립니다" in md1
    print("✓ 1차 전사 (제안 없음)")

    # 화자 학습 (동의 게이트 포함, server 툴 직접 호출)
    from meeting_scribe import server
    denied = server.confirm_speaker(job1, "화자1", "김대리", consent_confirmed=False)
    assert denied["enrolled"] is False
    assert not db.list_speakers(), "동의 없이 화자가 등록되면 안 됨"
    ok = server.confirm_speaker(job1, "화자1", "김대리", consent_confirmed=True)
    assert ok["enrolled"] is True
    md1b = (Path(j1["result_dir"]) / "transcript.md").read_text(encoding="utf-8")
    assert "김대리" in md1b, "확정 후 전사본에 실명 반영"
    print("✓ 동의 게이트 + 화자 확정 + 전사본 재생성")

    # 2차: 같은 목소리(비슷한 임베딩) → 김대리 제안
    job2 = jobs.enqueue("single", str(wav), pid, {"language": "ko"})
    run_job(job2)
    map2 = {m["cluster_label"]: m for m in db.get_job_speaker_map(job2)}
    assert map2["화자1"]["suggested_name"] == "김대리", map2
    assert map2["화자1"]["similarity"] > 0.9
    assert map2["화자2"]["suggested_name"] is None
    md2 = (Path(db.get_job(job2)["result_dir"]) / "transcript.md").read_text(encoding="utf-8")
    assert "김대리(?)" in md2
    print(f"✓ 화자 재인식 (유사도 {map2['화자1']['similarity']:.3f}) — 핵심 기능")

    # 회의록 저장 → 컨텍스트 축적
    server.save_minutes("오프라인테스트", job2, "주간회의", "# 회의록\n...",
                        "일정 공유 및 확인.", new_terms=["AMR :: 자율이동로봇"])
    ctx = server.get_project_context("오프라인테스트")
    assert ctx["recent_meetings"][0]["summary"] == "일정 공유 및 확인."
    assert any(g["term"] == "AMR" for g in ctx["glossary"])
    print("✓ 회의록 저장 + 컨텍스트 축적")

    # 참석자 한정 화자분리: 김대리·박과장 보이스프린트 등록 후 participants로 후보 제한
    # (박과장은 일부러 프로젝트 참석자 테이블에 넣지 않음 — confirm_speaker만 한 화자 시나리오)
    import json
    kid = db.upsert_speaker("김대리", "테스트 재등록")
    db.add_speaker_embedding(kid, ns["emb1"], source="test")
    pkid = db.upsert_speaker("박과장", "테스트 등록")
    db.add_speaker_embedding(pkid, ns["emb2"], source="test")

    # 대조군: participants 미지정 → 전체 등록 화자가 후보 (기존 동작 보존).
    # 박과장은 프로젝트 참석자가 아니어도 제안되어야 함 (프로젝트 명단 fallback 금지).
    job3 = jobs.enqueue("single", str(wav), pid, {"language": "ko"})
    run_job(job3)
    map3 = {m["cluster_label"]: m for m in db.get_job_speaker_map(job3)}
    assert map3["화자2"]["suggested_name"] == "박과장", map3
    print("✓ participants 미지정 시 전체 등록 화자 후보 (기존 동작 보존)")

    # participants=["김대리"] → 박과장은 어떤 클러스터에도 제안되지 않아야 함
    res = server.submit_transcription(str(wav), project="오프라인테스트",
                                      participants=["김대리 :: PM"])
    job4 = res["job_id"]
    run_job(job4)
    j4 = db.get_job(job4)
    assert j4["status"] == "done", j4["error"]
    opts4 = json.loads(j4["options"])
    assert opts4["participants"] == ["김대리"], opts4  # 역할(:: PM)은 제거
    assert "num_speakers" not in opts4, opts4  # 참석자 수로 강제하지 않음 (자동 추정 유지)
    map4 = {m["cluster_label"]: m for m in db.get_job_speaker_map(job4)}
    assert map4["화자1"]["suggested_name"] == "김대리", map4
    assert all(m["suggested_name"] != "박과장" for m in map4.values()), map4
    print("✓ 참석자 한정 화자분리 (명단 밖 화자 미제안 + 화자 수 자동 추정 유지)")

    # participants 파싱: 중복 제거, 괄호 역할 제거, 형식 오류 즉시 에러, 무교집합 경고
    res5 = server.submit_transcription(str(wav), project="오프라인테스트",
                                       participants=["김대리 :: PM", "김대리 :: 서기",
                                                     "박과장(과장)"])
    opts5 = json.loads(db.get_job(res5["job_id"])["options"])
    assert opts5["participants"] == ["김대리", "박과장"], opts5  # 중복·괄호 정리
    assert "num_speakers" not in opts5, opts5
    for bad in (["김대리, 박과장"], ["김대리 : PM"]):
        try:
            server.submit_transcription(str(wav), project="오프라인테스트", participants=bad)
            raise AssertionError(f"형식 오류 participants가 통과됨: {bad}")
        except ValueError:
            pass
    res6 = server.submit_transcription(str(wav), project="오프라인테스트",
                                       participants=["미등록인물"])
    assert "warning" in res6, res6  # 등록 화자·프로젝트 참석자와 무교집합 → 경고
    print("✓ participants 파싱 검증 (중복 제거 + 형식 오류 에러 + 무교집합 경고)")

    # purge (동의 철회)
    assert server.purge_speaker("김대리")["deleted"] is True
    assert server.purge_speaker("박과장")["deleted"] is True
    assert not db.all_speaker_embeddings()
    print("✓ 보이스프린트 삭제")


def test_job_ownership_guards() -> None:
    from datetime import datetime, timedelta, timezone
    from meeting_scribe import pipeline

    # 1) run_job은 이미 done인 작업을 건드리지 않음 (이중 실행 방지)
    done_job = jobs.enqueue("single", str(TEST_DATA / "없는파일.wav"), None, {})
    db.update_job(done_job, status="done", stage="완료")
    pipeline.run_job(done_job)  # 가드 없으면 입력 파일이 없어 error로 덮어씀
    j = db.get_job(done_job)
    assert j["status"] == "done" and j["error"] == "", dict(j)

    # 2) 실행 중 예외가 나도, 그 사이 다른 프로세스가 done으로 끝냈으면 error로 덮어쓰지 않음
    #    (실사례: job f3fed1198f20 — 이전 인스턴스가 완료한 작업을 error로 덮어씀)
    race_job = jobs.enqueue("single", str(TEST_DATA / "없는파일.wav"), None, {})
    orig_run_single = pipeline.run_single

    def _other_process_finishes_then_we_fail(job_id):
        db.update_job(job_id, status="done", stage="완료")  # 다른 프로세스가 먼저 완료
        raise RuntimeError("이중 실행된 쪽의 실패")

    pipeline.run_single = _other_process_finishes_then_we_fail
    try:
        pipeline.run_job(race_job)
    finally:
        pipeline.run_single = orig_run_single
    j2 = db.get_job(race_job)
    assert j2["status"] == "done" and j2["error"] == "", dict(j2)

    # 3) 재시작 복구: queued는 항상, running은 updated_at이 스테일한 것만 재큐잉
    q_job = jobs.enqueue("single", str(TEST_DATA / "없는파일.wav"), None, {})
    fresh_run = jobs.enqueue("single", str(TEST_DATA / "없는파일.wav"), None, {})
    db.update_job(fresh_run, status="running", stage="ASR")  # 방금 갱신 → 살아있는 프로세스 소유
    stale_run = jobs.enqueue("single", str(TEST_DATA / "없는파일.wav"), None, {})
    db.update_job(stale_run, status="running", stage="ASR")
    old = (datetime.now(timezone.utc) - timedelta(seconds=1200)).isoformat(timespec="seconds")
    with db.get_conn() as conn:
        conn.execute("UPDATE jobs SET updated_at=? WHERE id=?", (old, stale_run))
    resumable = db.pending_jobs(stale_running_seconds=600)
    assert q_job in resumable, resumable
    assert stale_run in resumable, resumable  # 죽은 프로세스의 작업은 재개 (기존 기능 유지)
    assert fresh_run not in resumable, resumable
    assert done_job not in resumable
    assert fresh_run in db.pending_jobs()  # 스테일 판정 없는 기본 동작은 그대로
    print("✓ job 소유권 가드 (done 스킵 + error 덮어쓰기 방지 + 스테일 재큐잉)")


def test_groq_retry() -> None:
    import types

    import httpx
    from meeting_scribe.asr import groq

    fake_chunk = TEST_DATA / "retry_chunk.wav"
    fake_chunk.write_bytes(b"RIFF")  # 내용 무관 — 시도마다 다시 열 수 있으면 됨

    class FakeClient:
        def __init__(self, statuses):
            self.statuses = list(statuses)
            self.calls = 0

        def post(self, url, **kw):
            self.calls += 1
            status = self.statuses.pop(0)
            headers = {"retry-after": "7"} if status == 429 else {}
            return httpx.Response(status, headers=headers, json={"text": "ok"},
                                  request=httpx.Request("POST", url))

    sleeps: list[float] = []
    orig_time = groq.time
    groq.time = types.SimpleNamespace(sleep=sleeps.append)
    try:
        # 429 → 500 → 성공: Retry-After 존중, 헤더 없으면 지수 백오프
        c = FakeClient([429, 500, 200])
        resp = groq._post_with_retry(c, fake_chunk, "fake-key", {})
        assert resp.status_code == 200 and c.calls == 3
        assert sleeps == [7.0, 4.0], sleeps  # 7=Retry-After, 4=2*2^1 백오프

        # 재시도 불가 4xx는 즉시 실패
        sleeps.clear()
        c2 = FakeClient([400])
        try:
            groq._post_with_retry(c2, fake_chunk, "fake-key", {})
            raise AssertionError("400이 통과됨")
        except httpx.HTTPStatusError:
            pass
        assert c2.calls == 1 and not sleeps

        # 재시도 소진 (초기 1회 + 재시도 3회) 후 429 그대로 실패
        sleeps.clear()
        c3 = FakeClient([429, 429, 429, 429])
        try:
            groq._post_with_retry(c3, fake_chunk, "fake-key", {})
            raise AssertionError("429 소진이 통과됨")
        except httpx.HTTPStatusError:
            pass
        assert c3.calls == 4 and len(sleeps) == 3
    finally:
        groq.time = orig_time
    print("✓ Groq 429/5xx 재시도 (Retry-After 존중 + 백오프 + 소진 시 실패)")


def test_server_tools_registered() -> None:
    from meeting_scribe import server
    import anyio
    tools = anyio.run(server.mcp.list_tools)
    names = {t.name for t in tools}
    expected = {"system_check", "create_project", "add_glossary", "add_participants",
                "list_projects", "get_project_context", "submit_transcription",
                "submit_zoom_tracks", "get_job_status", "get_transcript",
                "confirm_speaker", "list_speakers", "purge_speaker", "save_minutes"}
    assert expected <= names, expected - names
    check = server.system_check()
    assert check["cpu_cores"] and check["data_dir"]
    print(f"✓ MCP 툴 {len(names)}개 등록, system_check 동작")


if __name__ == "__main__":
    ensure_dirs()
    db.init_db()
    test_merge()
    test_audio()
    test_speaker_matching_math()
    test_pipeline_e2e_with_fakes()
    test_job_ownership_guards()
    test_groq_retry()
    test_server_tools_registered()
    print("\n오프라인 테스트 전부 통과 ✅")
