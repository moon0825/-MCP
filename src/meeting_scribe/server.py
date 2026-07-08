"""meeting-scribe MCP 서버 (stdio).

역할 분담: 이 서버는 '화자명 붙은 정확한 전사본 + 프로젝트 컨텍스트'까지만 책임진다.
회의록(요약/결정/액션아이템) 작성과 용어 추론·보정은 MCP 클라이언트(Claude)가 수행한다.
"""
import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import db
from .asr import available_adapters
from .config import DATA_DIR, ensure_dirs
from .jobs import enqueue, start_worker
from .models import models_ready
from .pipeline import name_map_for_job, rerender_transcript

INSTRUCTIONS = """회의 녹음 전사 + 화자 학습 MCP 서버. 권장 워크플로우:

1. (최초 1회) create_project로 프로젝트 등록, add_glossary/add_participants로 용어·참석자 등록
2. submit_transcription(녹음 파일) 또는 submit_zoom_tracks(Zoom 참가자별 오디오 폴더) → job_id
3. get_job_status로 진행 확인 (전사는 수 분 걸림 — 블로킹하지 말 것)
4. get_transcript로 전사본 확인. speaker_map의 미확정 화자는 사용자에게 물어본 뒤
   confirm_speaker로 확정 (당사자 동의 필수 — consent_confirmed=True). 확정하면 다음
   회의부터 자동 인식됨.
5. 회의록 작성: get_project_context의 용어집·참석자·이전 회의 요약을 근거로,
   전사 오류(특히 [?] 표시 구간)를 프로젝트 문맥으로 보수적으로 추론·교정하되
   원문에 없는 내용을 지어내지 말 것. 확신 없는 부분은 회의록에 (불명확) 표기.
   구성: 요약 → 논의 내용 → 결정사항 → 액션아이템(담당자/기한).
6. save_minutes로 회의록 저장 — 요약이 다음 회의의 컨텍스트가 되어 갈수록 똑똑해짐.
   새로 등장한 용어는 new_terms로 함께 등록할 것."""

mcp = FastMCP("meeting-scribe", instructions=INSTRUCTIONS)


def _project_id(project: str | None) -> int | None:
    if not project:
        return None
    row = db.get_project(project)
    if row is None:
        raise ValueError(f"프로젝트 '{project}'가 없습니다. create_project로 먼저 등록하세요.")
    return row["id"]


@mcp.tool()
def system_check() -> dict:
    """실행 환경 점검: CPU/RAM/GPU, ffmpeg, 화자분리 모델, 사용 가능한 ASR 엔진,
    로컬 whisper 실행 계획(GPU 자동 감지)."""
    from .audio import ffmpeg_exe
    from .hw import local_whisper_plan, ram_gb

    try:
        ff = ffmpeg_exe()
    except Exception:
        ff = None
    adapters = available_adapters()
    plan = local_whisper_plan()
    gpu = plan.pop("gpu")
    if plan["device"] == "cuda":
        rec = (f"GPU({gpu['name']})에서 로컬 whisper {plan['model']}가 실시간보다 빠르게 "
               "동작합니다. 민감한 회의도 부담 없이 asr='local'로 처리하세요.")
    elif gpu:
        rec = (f"GPU({gpu['name']})가 있지만 CUDA 런타임이 없어 CPU로 동작합니다. "
               "NVIDIA 드라이버 + CUDA 12 라이브러리(cuBLAS/cuDNN) 설치 시 로컬 전사가 크게 빨라집니다.")
    else:
        rec = "GPU 미감지 — 클라우드 ASR + 로컬 화자분리 하이브리드가 가장 빠릅니다."
    return {
        "cpu_cores": os.cpu_count(),
        "ram_gb": ram_gb(),
        "gpu": gpu or "없음",
        "ffmpeg": ff or "없음 — pip install imageio-ffmpeg",
        "diarization_models": "준비됨" if models_ready() else "최초 작업 시 자동 다운로드 (~35MB)",
        "asr_adapters": adapters or ["없음 — GROQ_API_KEY 등 환경변수 필요"],
        "default_asr": adapters[0] if adapters else None,
        "local_whisper_plan": plan,
        "data_dir": str(DATA_DIR),
        "recommendation": rec,
    }


@mcp.tool()
def create_project(name: str, description: str = "", minutes_template: str = "") -> dict:
    """프로젝트 등록/수정. description은 ASR 힌트와 회의록 작성 맥락으로 쓰인다."""
    pid = db.upsert_project(name, description, minutes_template)
    return {"project_id": pid, "name": name}


@mcp.tool()
def add_glossary(project: str, terms: list[str]) -> dict:
    """프로젝트 용어집에 용어 추가. 각 항목은 '용어' 또는 '용어 :: 설명' 형식.
    참석자 이름과 함께 ASR 부스팅 + 회의록 추론의 근거가 된다."""
    pid = _project_id(project)
    parsed = []
    for t in terms:
        term, _, note = t.partition("::")
        parsed.append({"term": term.strip(), "note": note.strip()})
    n = db.add_glossary_terms(pid, parsed)
    return {"added_or_updated": n, "total_terms": len(db.get_context(pid)["glossary"])}


@mcp.tool()
def add_participants(project: str, names: list[str]) -> dict:
    """프로젝트 참석자 등록. 각 항목은 '이름' 또는 '이름 :: 역할' 형식."""
    pid = _project_id(project)
    parsed = []
    for t in names:
        name, _, role = t.partition("::")
        parsed.append({"name": name.strip(), "role": role.strip()})
    n = db.add_participants(pid, parsed)
    return {"added_or_updated": n}


@mcp.tool()
def list_projects() -> list[dict]:
    """등록된 프로젝트 목록과 용어/참석자/회의 수."""
    return db.list_projects()


@mcp.tool()
def get_project_context(project: str) -> dict:
    """회의록 작성용 프로젝트 컨텍스트: 설명, 용어집, 참석자, 최근 회의 요약, 회의록 템플릿."""
    return db.get_context(_project_id(project))


@mcp.tool()
def submit_transcription(file_path: str, project: str = "", language: str = "ko",
                         num_speakers: int = 0, asr: str = "",
                         skip_diarization: bool = False) -> dict:
    """녹음 파일 전사 작업 등록 (즉시 job_id 반환, 백그라운드 처리).
    - file_path: 오디오 파일 절대 경로 (m4a/mp3/wav 등)
    - num_speakers: 화자 수를 알면 지정 (정확도 향상), 0이면 자동
    - asr: rtzr|groq|gemini|local 강제 지정 (기본: 자동 선택. 민감한 회의는 'local')"""
    p = Path(file_path)
    if not p.is_file():
        raise ValueError(f"파일 없음: {file_path}")
    options = {"language": language, "skip_diarization": skip_diarization}
    if num_speakers > 0:
        options["num_speakers"] = num_speakers
    if asr:
        options["asr"] = asr
    job_id = enqueue("single", file_path, _project_id(project), options)
    return {"job_id": job_id, "status": "queued",
            "hint": "get_job_status로 진행 확인. 하이브리드 기본 설정 기준 수 분 소요."}


@mcp.tool()
def submit_zoom_tracks(dir_path: str, project: str = "", language: str = "ko",
                       asr: str = "") -> dict:
    """Zoom '참가자별 개별 오디오 파일' 폴더 전사 (화자분리 불필요 — 파일명이 곧 화자).
    Zoom 설정 > 레코딩 > '각 참가자의 오디오 파일을 개별 녹음'을 켜고 로컬 녹음하면
    Documents/Zoom/<회의>/Audio Record/ 에 참가자별 m4a가 생긴다. 그 폴더 경로를 넘길 것."""
    p = Path(dir_path)
    if not p.is_dir():
        raise ValueError(f"폴더 없음: {dir_path}")
    options = {"language": language}
    if asr:
        options["asr"] = asr
    job_id = enqueue("zoom_tracks", dir_path, _project_id(project), options)
    return {"job_id": job_id, "status": "queued"}


@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """작업 상태·현재 단계·오류 확인."""
    job = db.get_job(job_id)
    if job is None:
        raise ValueError(f"작업 없음: {job_id}")
    return {"job_id": job_id, "status": job["status"], "stage": job["stage"],
            "error": job["error"] or None, "updated_at": job["updated_at"]}


@mcp.tool()
def get_transcript(job_id: str, preview_chars: int = 3000) -> dict:
    """완료된 작업의 전사본. 전문은 transcript_path 파일을 읽을 것 (긴 회의 전문을
    통째로 컨텍스트에 넣지 않기 위해 미리보기만 반환). speaker_map의 미확정 화자는
    사용자 확인 후 confirm_speaker로 확정할 것."""
    job = db.get_job(job_id)
    if job is None:
        raise ValueError(f"작업 없음: {job_id}")
    if job["status"] != "done":
        return {"status": job["status"], "stage": job["stage"], "error": job["error"] or None}
    md_path = Path(job["result_dir"]) / "transcript.md"
    text = md_path.read_text(encoding="utf-8")
    speaker_map = []
    for m in db.get_job_speaker_map(job_id):
        speaker_map.append({
            "cluster": m["cluster_label"],
            "confirmed": m["confirmed_name"],
            "suggested": m["suggested_name"],
            "similarity": round(m["similarity"], 3) if m["similarity"] is not None else None,
        })
    return {
        "status": "done",
        "transcript_path": str(md_path),
        "preview": text[:preview_chars] + ("\n…(이하 생략)" if len(text) > preview_chars else ""),
        "speaker_map": speaker_map,
        "uncertain_segments": text.count("[?]"),
    }


@mcp.tool()
def confirm_speaker(job_id: str, cluster_label: str, name: str,
                    consent_confirmed: bool = False, consent_note: str = "") -> dict:
    """화자 확정 + 보이스프린트 학습. 확정하면 전사본이 실명으로 재생성되고,
    다음 회의부터 이 사람을 자동 제안한다.

    ⚠️ 음성 임베딩은 개인정보보호법상 민감정보(생체인식정보)입니다. 본인 동의를 받은 뒤
    consent_confirmed=True로 호출하세요. 동의 없으면 익명 라벨(화자1)로 두면 됩니다."""
    if not consent_confirmed:
        return {"enrolled": False,
                "message": "당사자 동의 확인 후 consent_confirmed=True로 다시 호출하세요. "
                           "동의 전까지는 익명 라벨로 유지됩니다."}
    sid = db.upsert_speaker(name, consent_note or f"confirm_speaker via job {job_id}")
    emb = db.confirm_cluster(job_id, cluster_label, sid)
    if emb is None:
        raise ValueError(f"작업 {job_id}에 클러스터 '{cluster_label}'가 없습니다.")
    db.add_speaker_embedding(sid, emb, source=f"job:{job_id}/{cluster_label}")
    path = rerender_transcript(job_id)
    return {"enrolled": True, "speaker": name,
            "embeddings_total": next((s["embeddings"] for s in db.list_speakers()
                                      if s["name"] == name), 1),
            "transcript_path": str(path)}


@mcp.tool()
def list_speakers() -> list[dict]:
    """학습된 화자 목록 (임베딩 수가 많을수록 인식이 안정적)."""
    return db.list_speakers()


@mcp.tool()
def purge_speaker(name: str) -> dict:
    """화자 보이스프린트 완전 삭제 (동의 철회 대응)."""
    ok = db.purge_speaker(name)
    return {"deleted": ok}


@mcp.tool()
def save_minutes(project: str, job_id: str, title: str, minutes_md: str,
                 summary: str, meeting_date: str = "", new_terms: list[str] | None = None) -> dict:
    """작성한 회의록 저장. summary(3~5문장)는 다음 회의의 컨텍스트로 재사용된다.
    회의에서 새로 등장한 용어는 new_terms('용어 :: 설명')로 용어집에 누적할 것."""
    pid = _project_id(project)
    meeting_id = db.save_meeting(pid, job_id, title, meeting_date, minutes_md, summary)
    added = 0
    if new_terms:
        parsed = []
        for t in new_terms:
            term, _, note = t.partition("::")
            parsed.append({"term": term.strip(), "note": note.strip()})
        added = db.add_glossary_terms(pid, parsed)
    job = db.get_job(job_id)
    if job and job["result_dir"]:
        out = Path(job["result_dir"]) / "minutes.md"
        out.write_text(minutes_md, encoding="utf-8")
    return {"meeting_id": meeting_id, "new_terms_added": added}


def main() -> None:
    ensure_dirs()
    db.init_db()
    start_worker()
    mcp.run()


if __name__ == "__main__":
    main()
