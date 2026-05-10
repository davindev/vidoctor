"use client";

import { useEffect, useRef, useState } from "react";
import type {
  AnalysisDetail,
  Dimension,
  FindingItem,
} from "@/lib/api";
import {
  DIMENSION_COLOR,
  DIMENSION_DESC,
  DIMENSION_LABEL,
  DIMENSION_ORDER,
  deleteAnalysis,
  fetchAnalysis,
  fetchVideoUrl,
} from "@/lib/api";
import { basename, fmtHMS } from "@/lib/format";
import { ResultHeader, ResultPage } from "./ResultHeader";

interface Props {
  analysisId: string;
  onDeleted: () => void;
}

const CPS_KIND_LABEL: Record<string, string> = {
  too_fast: "빠름",
  too_slow: "느림",
};

const GAZE_DIRECTION_LABEL: Record<string, string> = {
  front: "정면",
  left: "오른쪽",
  right: "왼쪽",
  up: "위",
  down: "아래",
  left_up: "오른쪽 위",
  left_down: "오른쪽 아래",
  right_up: "왼쪽 위",
  right_down: "왼쪽 아래",
};

function findingDetail(dim: Dimension, ev: FindingItem): string {
  const p = ev.payload;
  if (dim === "filler") return String(p.text ?? "");
  if (dim === "cps") return CPS_KIND_LABEL[String(p.kind ?? "")] ?? "";
  if (dim === "gaze")
    return GAZE_DIRECTION_LABEL[String(p.direction ?? "")] ?? String(p.direction ?? "");
  if (dim === "dead_zone") return `${(ev.end - ev.start).toFixed(1)}초`;
  if (dim === "content_gap") return String(p.description ?? "");
  return "";
}

export function ResultView({ analysisId, onDeleted }: Props) {
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setVideoUrl(null);
    setError(null);
    setDeleteError(null);
    setConfirmOpen(false);

    // analysis 본문은 필수, video URL은 R2 만료/네트워크로 실패해도 본문 표시는 유지.
    Promise.allSettled([
      fetchAnalysis(analysisId),
      fetchVideoUrl(analysisId),
    ]).then(([detailRes, urlRes]) => {
      if (cancelled) return;
      if (detailRes.status === "rejected") {
        const e = detailRes.reason as unknown;
        setError(e instanceof Error ? e.message : String(e));
        return;
      }
      setDetail(detailRes.value);
      setVideoUrl(urlRes.status === "fulfilled" ? urlRes.value : null);
    });

    return () => {
      cancelled = true;
    };
  }, [analysisId]);

  const seekTo = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = t;
    void v.play().catch(() => undefined);
  };

  if (error) {
    return (
      <ResultPage>
        <p className="text-danger">불러오기 실패: {error}</p>
      </ResultPage>
    );
  }
  if (!detail) {
    return (
      <ResultPage>
        <p className="text-ink-4">불러오는 중…</p>
      </ResultPage>
    );
  }

  const filename = basename(detail.storage_path) || "—";
  const latencySec =
    detail.started_at && detail.finished_at
      ? (new Date(detail.finished_at).getTime() -
          new Date(detail.started_at).getTime()) /
        1000
      : null;
  const issuesWithFindings = DIMENSION_ORDER.filter(
    (d) => (detail.findings[d]?.length ?? 0) > 0,
  );

  return (
    <ResultPage>
      <ResultHeader
        trailing="결과"
        filename={filename}
        category={detail.category}
        durationSec={detail.duration_sec}
      />

      <div className="flex flex-col gap-6">
        {/* Video card */}
        <section className="overflow-hidden rounded-[14px] border border-line bg-surface">
          {videoUrl ? (
            <video ref={videoRef} src={videoUrl} controls className="block w-full" />
          ) : (
            <div className="aspect-video grid place-items-center bg-[#1A1612] text-sm text-white/70">
              원본 영상이 R2에 없습니다.
            </div>
          )}
          <div className="flex items-center gap-3.5 border-t border-line px-[18px] py-[11px] text-xs text-ink-3">
            <MetaItem
              label="처리 시간"
              value={latencySec !== null ? fmtHMS(latencySec) : "—"}
            />
            <MetaItem
              label="LLM 비용"
              value={detail.cost_usd ? `$${detail.cost_usd.toFixed(4)}` : "—"}
            />
          </div>
        </section>

        {/* Suggestions */}
        <section className="overflow-hidden rounded-[14px] border border-line bg-surface">
          <div className="flex items-center justify-between gap-3 px-[22px] pt-[18px] pb-3.5">
            <div>
              <div className="text-[15px] font-semibold tracking-[-0.01em]">
                개선 제안
              </div>
              <div className="mt-1 text-xs text-ink-4">
                LLM이 영상 흐름을 분석해 제안한 보완 포인트입니다.
              </div>
            </div>
            <div className="text-xs text-ink-4">
              <span className="font-medium text-ink-2">
                {detail.suggestions.length}
              </span>
              개 제안
            </div>
          </div>

          {detail.suggestions.length === 0 ? (
            <div className="px-[22px] pb-5 text-sm text-ink-4">개선 제안 없음.</div>
          ) : (
            <div className="flex flex-col gap-2 px-[18px] pb-[22px] pt-1.5">
              {detail.suggestions.map((s, i) => (
                <SuggestionCard
                  key={i}
                  suggestion={s}
                  findings={detail.findings}
                  onSeek={seekTo}
                />
              ))}
            </div>
          )}
        </section>

        {/* Issues */}
        <section className="overflow-hidden rounded-[14px] border border-line bg-surface">
          <div className="px-[22px] pt-[18px] pb-3.5">
            <div className="text-[15px] font-semibold tracking-[-0.01em]">
              이슈 목록
            </div>
            <div className="mt-1 text-xs text-ink-4">
              {issuesWithFindings.length}개 차원의 자동 검출 결과 · 클릭 시 해당
              구간으로 이동합니다.
            </div>
          </div>

          <div className="flex flex-col gap-1 px-[18px] pb-[18px] pt-1.5">
            {issuesWithFindings.length === 0 ? (
              <div className="px-1 py-3.5 text-sm text-ink-4">검출된 이슈 없음.</div>
            ) : (
              issuesWithFindings.map((dim, i) => (
                <IssueRow
                  key={dim}
                  dim={dim}
                  events={detail.findings[dim] ?? []}
                  onSeek={seekTo}
                  isLast={i === issuesWithFindings.length - 1}
                />
              ))
            )}
          </div>
        </section>
      </div>

      {/* Delete card */}
      <div className="mt-7 flex items-center justify-between gap-4 rounded-xl border border-[#EAC8C2] bg-[#FBEFEB] px-[22px] py-[18px]">
        <div className="text-[13px] text-[#6E3A33]">
          이 분석을 더 이상 사용하지 않나요?{" "}
          <b className="font-semibold text-danger">영상과 분석 결과</b>가 모두
          제거됩니다.
        </div>
        <button
          type="button"
          onClick={() => setConfirmOpen(true)}
          className="flex-shrink-0 rounded-full border border-[#E5C2BD] bg-transparent px-4 py-2 text-[13px] font-medium text-danger transition-[background,border-color] duration-150 hover:border-danger hover:bg-[#FBEAE3]"
        >
          삭제
        </button>
      </div>

      {deleteError && (
        <div className="mt-3 rounded-lg border border-[#EFCBB9] bg-[#FBEAE3] px-3 py-2.5 text-[13px] text-danger">
          삭제 실패: {deleteError}
        </div>
      )}

      {confirmOpen && (
        <DeleteModal
          filename={filename}
          onCancel={() => setConfirmOpen(false)}
          onConfirm={async () => {
            try {
              await deleteAnalysis(analysisId);
              setConfirmOpen(false);
              onDeleted();
            } catch (e) {
              setDeleteError(e instanceof Error ? e.message : String(e));
              setConfirmOpen(false);
            }
          }}
        />
      )}
    </ResultPage>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-ink-4">{label}</span>
      <span
        className="font-medium text-ink-2"
        style={{ fontFeatureSettings: "'tnum'" }}
      >
        {value}
      </span>
    </div>
  );
}

function SuggestionCard({
  suggestion,
  findings,
  onSeek,
}: {
  suggestion: { text: string; finding_refs: string[] };
  findings: Record<Dimension, FindingItem[]>;
  onSeek: (t: number) => void;
}) {
  const refs = suggestion.finding_refs
    .map((r) => resolveRef(r, findings))
    .filter((x): x is { dim: Dimension; start: number } => x !== null);

  return (
    <div className="rounded-[10px] border border-line bg-[#FDFBF6] p-4">
      <div className="text-[13.5px] leading-[1.55] text-ink">{suggestion.text}</div>
      {refs.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {refs.map((r, i) => (
            <button
              key={i}
              type="button"
              onClick={() => onSeek(r.start)}
              className="inline-flex items-center gap-1.5 rounded-full border border-line-2 bg-surface px-2.5 py-[3px] text-[11.5px] font-medium text-ink-2 transition-[border-color,background] duration-[120ms] hover:border-accent hover:bg-accent-tint"
              style={{ fontFeatureSettings: "'tnum'" }}
            >
              <span className="h-[5px] w-[5px] rounded-full bg-accent" />
              {fmtHMS(r.start)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function IssueRow({
  dim,
  events,
  onSeek,
  isLast,
}: {
  dim: Dimension;
  events: FindingItem[];
  onSeek: (t: number) => void;
  isLast: boolean;
}) {
  const color = DIMENSION_COLOR[dim];
  return (
    <div
      className={`grid grid-cols-[16px_1fr] gap-3 py-3.5 px-1 ${
        isLast ? "" : "border-b border-line"
      }`}
      style={{ color }}
    >
      <span
        className="mt-[6px] h-2 w-2 rounded-full"
        style={{ background: "currentColor" }}
      />
      <div className="min-w-0">
        <div className="mb-0.5 flex items-baseline gap-2 text-[13.5px] font-medium text-ink">
          <span>{DIMENSION_LABEL[dim]}</span>
        </div>
        <div className="mb-2 text-xs leading-[1.5] text-ink-3">
          {DIMENSION_DESC[dim]}
        </div>
        {/* table style — Analysis.html ts-row[data-style="table"] */}
        <div
          className="flex flex-col border-t border-line"
          style={{ ["--c" as never]: color }}
        >
          {events.map((ev, i) => (
            <button
              key={i}
              type="button"
              onClick={() => onSeek(ev.start)}
              className="grid grid-cols-[78px_1fr] items-baseline gap-3.5 border-b border-line px-2 py-[9px] text-left transition-[background] duration-[120ms] hover:bg-[#FBF7F2]"
            >
              <span
                className="text-[11.5px] font-medium"
                style={{ color, fontFeatureSettings: "'tnum'" }}
              >
                {fmtHMS(ev.start)}
              </span>
              <span className="text-[12.5px] leading-[1.5] text-ink-2">
                {findingDetail(dim, ev) || "—"}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function DeleteModal({
  filename,
  onCancel,
  onConfirm,
}: {
  filename: string;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(31,26,23,0.42)]"
      onClick={onCancel}
    >
      <div
        className="w-[min(440px,calc(100%-32px))] rounded-[14px] border border-line bg-surface px-7 pb-[22px] pt-7 shadow-[0_10px_40px_rgba(31,26,23,0.12)]"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="mb-2 text-lg font-semibold tracking-[-0.015em]">
          분석을 삭제할까요?
        </h3>
        <p className="mb-[22px] text-[13.5px] leading-[1.6] text-ink-3">
          <b className="font-semibold text-ink">{filename}</b>의 영상 파일과 분석
          결과가 영구적으로 제거됩니다. 좌측 이전 기록에서도 함께 사라지며 복구할
          수 없습니다.
        </p>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-full border border-line-2 bg-surface px-[18px] py-[9px] text-[13px] font-medium text-ink-2 transition-[border-color] duration-150 hover:border-ink-3"
          >
            취소
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="rounded-full border border-danger bg-danger px-[18px] py-[9px] text-[13px] font-medium text-white transition-[background,border-color] duration-150 hover:border-[#9A3E34] hover:bg-[#9A3E34]"
          >
            삭제
          </button>
        </div>
      </div>
    </div>
  );
}

function resolveRef(
  ref: string,
  findings: Record<Dimension, FindingItem[]>,
): { dim: Dimension; start: number } | null {
  const colon = ref.indexOf(":");
  if (colon === -1) return null;
  const dim = ref.slice(0, colon) as Dimension;
  const idx = Number(ref.slice(colon + 1));
  if (!Number.isFinite(idx)) return null;
  const list = findings[dim];
  if (!list || idx < 0 || idx >= list.length) return null;
  return { dim, start: list[idx].start };
}

