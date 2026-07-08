"""ASR 결과(단어 타임스탬프) × 화자분리 구간 병합 → 화자별 발화록.

전략:
- 단어 타임스탬프가 있으면: 각 ASR 세그먼트의 단어들을 화자 구간에 배정하고,
  한 화자가 80% 이상이면 세그먼트 원문 유지(한국어 띄어쓰기 보존), 아니면 단어 단위 분할.
- 없으면(Gemini): 세그먼트와 겹치는 시간이 가장 긴 화자에 통째로 배정.
- avg_logprob 낮은 세그먼트는 uncertain=True → 회의록 작성 시 컨텍스트 추론 대상 표시.
"""
from dataclasses import asdict, dataclass, field

from .asr.base import AsrSegment
from .config import LOW_CONFIDENCE_LOGPROB
from .diarize import DiarSegment

UNKNOWN = "미상"


@dataclass
class Utterance:
    cluster: str
    start: float
    end: float
    text: str
    uncertain: bool = False


@dataclass
class Transcript:
    utterances: list[Utterance] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"utterances": [asdict(u) for u in self.utterances]}

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(utterances=[Utterance(**u) for u in d.get("utterances", [])])


def _speaker_at(t: float, diar: list[DiarSegment]) -> str | None:
    for s in diar:
        if s.start <= t <= s.end:
            return s.cluster
    best, gap = None, 2.0  # 2초 이내 최근접 구간까지 허용
    for s in diar:
        d = s.start - t if t < s.start else t - s.end
        if 0 < d < gap:
            best, gap = s.cluster, d
    return best


def _overlap_speaker(seg: AsrSegment, diar: list[DiarSegment]) -> str | None:
    overlaps: dict[str, float] = {}
    for d in diar:
        ov = min(seg.end, d.end) - max(seg.start, d.start)
        if ov > 0:
            overlaps[d.cluster] = overlaps.get(d.cluster, 0.0) + ov
    if not overlaps:
        return _speaker_at((seg.start + seg.end) / 2, diar)
    return max(overlaps, key=overlaps.get)


def merge(asr_segments: list[AsrSegment], diar: list[DiarSegment]) -> Transcript:
    utterances: list[Utterance] = []
    for seg in asr_segments:
        uncertain = seg.avg_logprob is not None and seg.avg_logprob < LOW_CONFIDENCE_LOGPROB
        if not seg.words:
            spk = _overlap_speaker(seg, diar) or UNKNOWN
            utterances.append(Utterance(spk, seg.start, seg.end, seg.text, uncertain))
            continue
        assignments = [(_speaker_at((w.start + w.end) / 2, diar) or UNKNOWN, w) for w in seg.words]
        counts: dict[str, int] = {}
        for spk, _ in assignments:
            counts[spk] = counts.get(spk, 0) + 1
        dominant = max(counts, key=counts.get)
        if counts[dominant] / len(assignments) >= 0.8:
            utterances.append(Utterance(dominant, seg.start, seg.end, seg.text, uncertain))
        else:  # 세그먼트 안에서 화자가 바뀜 → 단어 단위 분할
            cur_spk, cur_words = assignments[0][0], [assignments[0][1]]
            for spk, w in assignments[1:]:
                if spk == cur_spk:
                    cur_words.append(w)
                else:
                    utterances.append(Utterance(
                        cur_spk, cur_words[0].start, cur_words[-1].end,
                        " ".join(x.text for x in cur_words), uncertain))
                    cur_spk, cur_words = spk, [w]
            utterances.append(Utterance(
                cur_spk, cur_words[0].start, cur_words[-1].end,
                " ".join(x.text for x in cur_words), uncertain))
    # 같은 화자의 연속 발화(간격 2초 미만) 합치기
    merged: list[Utterance] = []
    for u in sorted(utterances, key=lambda x: x.start):
        if merged and merged[-1].cluster == u.cluster and u.start - merged[-1].end < 2.0:
            merged[-1].text = f"{merged[-1].text} {u.text}".strip()
            merged[-1].end = u.end
            merged[-1].uncertain = merged[-1].uncertain or u.uncertain
        else:
            merged.append(u)
    return Transcript(merged)


def _fmt_time(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def render_markdown(transcript: Transcript, name_map: dict[str, str],
                    title: str, header_lines: list[str]) -> str:
    lines = [f"# 전사본: {title}", ""]
    lines += header_lines
    lines += ["", "> `[?]` 표시는 인식 신뢰도가 낮은 구간 — 프로젝트 컨텍스트로 추론·보정 필요", "",
              "---", ""]
    for u in transcript.utterances:
        speaker = name_map.get(u.cluster, u.cluster)
        mark = " [?]" if u.uncertain else ""
        lines.append(f"**[{_fmt_time(u.start)}] {speaker}**:{mark} {u.text}")
        lines.append("")
    return "\n".join(lines)
