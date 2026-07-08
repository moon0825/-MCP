"""작업 파이프라인: 정규화 → 화자분리 → 화자 식별(제안) → ASR → 병합 → 전사본 저장."""
import json
import os
import re
from pathlib import Path

from . import db
from .asr import get_adapter
from .audio import duration_seconds, normalize, read_wav_float32
from .config import MAX_HOTWORDS
from .diarize import DiarSegment, diarize, extract_cluster_embeddings, match_speakers
from .merge import Transcript, Utterance, merge, render_markdown

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac", ".wma", ".mp4", ".webm"}


def _stage(job_id: str, stage: str) -> None:
    db.update_job(job_id, status="running", stage=stage)


def _hotwords_and_hint(project_id: int | None) -> tuple[list[str], str]:
    if not project_id:
        return [], ""
    hotwords = db.hotwords_for_project(project_id, MAX_HOTWORDS)
    ctx = db.get_context(project_id, recent_meetings=1)
    return hotwords, ctx.get("description", "")


def name_map_for_job(job_id: str) -> dict[str, str]:
    """클러스터 → 표시 이름 (확정 > 제안(물음표) > 클러스터 라벨)."""
    out: dict[str, str] = {}
    for m in db.get_job_speaker_map(job_id):
        if m["confirmed_name"]:
            out[m["cluster_label"]] = m["confirmed_name"]
        elif m["suggested_name"]:
            out[m["cluster_label"]] = f"{m['suggested_name']}(?)"
    return out


def rerender_transcript(job_id: str) -> Path:
    """화자 확정 후 전사본 마크다운 재생성."""
    job = db.get_job(job_id)
    result_dir = Path(job["result_dir"])
    data = json.loads((result_dir / "transcript.json").read_text(encoding="utf-8"))
    transcript = Transcript.from_dict(data)
    md = render_markdown(transcript, name_map_for_job(job_id),
                         data.get("title", job_id), data.get("header_lines", []))
    out = result_dir / "transcript.md"
    out.write_text(md, encoding="utf-8")
    return out


def _save_result(job_id: str, result_dir: Path, transcript: Transcript,
                 title: str, header_lines: list[str]) -> None:
    payload = transcript.to_dict()
    payload["title"] = title
    payload["header_lines"] = header_lines
    (result_dir / "transcript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    rerender_transcript(job_id)


def run_single(job_id: str) -> None:
    """일반 녹음 파일 1개 파이프라인 (폰 녹음, 회의실 녹음 등)."""
    job = db.get_job(job_id)
    options = json.loads(job["options"])
    result_dir = Path(job["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(job["input_path"])

    _stage(job_id, "오디오 정규화")
    wav = normalize(input_path, result_dir / "audio_16k.wav")
    samples, sr = read_wav_float32(wav)
    dur = duration_seconds(wav)

    diar: list[DiarSegment] = []
    if not options.get("skip_diarization"):
        _stage(job_id, f"화자분리 (로컬 CPU, 오디오 {dur/60:.0f}분)")
        diar = diarize(samples, sr, num_speakers=options.get("num_speakers"))

        _stage(job_id, "화자 임베딩 추출·등록 화자 매칭")
        cluster_embs = extract_cluster_embeddings(samples, sr, diar)
        matches = match_speakers(cluster_embs, db.all_speaker_embeddings())
        for cluster, emb in cluster_embs.items():
            sid, _name, sim = matches.get(cluster, (None, None, 0.0))
            db.set_job_speaker_map(job_id, cluster, emb, sid, sim)

    _stage(job_id, "전사 (ASR)")
    hotwords, hint = _hotwords_and_hint(job["project_id"])
    adapter = get_adapter(options.get("asr"))
    asr_segments = adapter.transcribe(wav, options.get("language", "ko"), hotwords, hint)

    _stage(job_id, "병합·전사본 생성")
    transcript = merge(asr_segments, diar) if diar else Transcript(
        [Utterance("화자1", s.start, s.end, s.text,
                   s.avg_logprob is not None and s.avg_logprob < -0.8)
         for s in asr_segments])
    header = [
        f"- 원본: `{input_path.name}` ({dur/60:.1f}분)",
        f"- ASR 엔진: {adapter.name}" + ("" if adapter.word_timestamps else " (단어 타임스탬프 없음 — 화자 배정 정밀도 낮음)"),
        f"- 감지 화자 수: {len({d.cluster for d in diar})}" if diar else "- 화자분리 생략",
    ]
    _save_result(job_id, result_dir, transcript, input_path.stem, header)
    db.update_job(job_id, status="done", stage="완료")


_TRACK_NAME_RE = re.compile(r"^(?:audio[_\s]?(?:only[_\s]?)?)?(.+?)(?:[_\s]?\d+)?$", re.IGNORECASE)


def track_name_from_file(p: Path) -> str:
    m = _TRACK_NAME_RE.match(p.stem)
    name = (m.group(1) if m else p.stem).strip("_- ")
    return name or p.stem


def run_zoom_tracks(job_id: str) -> None:
    """Zoom '참가자별 개별 오디오' 폴더 파이프라인 — 화자분리 불필요, 트랙명이 곧 화자.
    각 트랙에서 임베딩도 추출해 (동의 후) 등록하면 대면 회의 인식의 부트스트랩이 된다."""
    job = db.get_job(job_id)
    options = json.loads(job["options"])
    result_dir = Path(job["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    track_dir = Path(job["input_path"])
    tracks = sorted(p for p in track_dir.iterdir()
                    if p.suffix.lower() in AUDIO_EXTS and p.is_file())
    if not tracks:
        raise RuntimeError(f"오디오 트랙을 찾지 못함: {track_dir}")

    hotwords, hint = _hotwords_and_hint(job["project_id"])
    adapter = get_adapter(options.get("asr"))
    enrolled = db.all_speaker_embeddings()

    all_utts: list[Utterance] = []
    total_dur = 0.0
    for i, track in enumerate(tracks):
        label = track_name_from_file(track)
        _stage(job_id, f"트랙 {i+1}/{len(tracks)}: {label}")
        wav = normalize(track, result_dir / f"track_{i:02d}.wav")
        samples, sr = read_wav_float32(wav)
        total_dur = max(total_dur, duration_seconds(wav))

        # 근접 마이크 단일 화자 트랙 → 전체를 한 화자 구간으로 보고 임베딩 추출
        segs = [DiarSegment(0.0, duration_seconds(wav), label)]
        embs = extract_cluster_embeddings(samples, sr, segs)
        if label in embs:
            matches = match_speakers({label: embs[label]}, enrolled)
            sid, _n, sim = matches[label]
            db.set_job_speaker_map(job_id, label, embs[label], sid, sim)

        for s in adapter.transcribe(wav, options.get("language", "ko"), hotwords, hint):
            if s.text.strip():
                all_utts.append(Utterance(label, s.start, s.end, s.text,
                                          s.avg_logprob is not None and s.avg_logprob < -0.8))

    transcript = Transcript(sorted(all_utts, key=lambda u: u.start))
    header = [
        f"- Zoom 참가자별 트랙 {len(tracks)}개 ({total_dur/60:.1f}분)",
        f"- ASR 엔진: {adapter.name}",
        "- 화자 = 트랙 이름 (Zoom 메타데이터 기반, 화자분리 불필요)",
    ]
    _save_result(job_id, result_dir, transcript, track_dir.name, header)
    db.update_job(job_id, status="done", stage="완료")


def run_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if job is None:
        return
    try:
        if job["kind"] == "zoom_tracks":
            run_zoom_tracks(job_id)
        else:
            run_single(job_id)
    except Exception as e:  # 실패를 DB에 남겨 get_job_status로 확인 가능하게
        db.update_job(job_id, status="error", error=f"{type(e).__name__}: {e}")


def detect_num_threads() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))
