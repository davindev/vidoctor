"use client";

import { useEffect, useState } from "react";
import type { Category } from "@/lib/api";
import type { AnalyzingPhase } from "@/lib/analyze";
import { Pipeline } from "./Pipeline";
import { ResultHeader, ResultPage } from "./ResultHeader";

interface Props {
  category: Category | null;
  filename: string | null;
  phase: AnalyzingPhase;
  completed: Set<string>;
  startedAt: string | null;
}

export function AnalyzingView({
  category,
  filename,
  phase,
  completed,
  startedAt,
}: Props) {
  // 다운로드 중에는 영상 제목을 아직 모르므로 placeholder. metadata 이벤트 도착 시 교체됨.
  const headerFilename =
    filename ?? (phase === "downloading" ? "유튜브 URL" : null);
  return (
    <ResultPage>
      <ResultHeader
        trailing="진행 중"
        filename={headerFilename}
        category={category}
      />
      {/* 전사 단계가 길어 파이프라인이 멈춘 듯 보일 수 있어, 경과 시간으로 진행을 알린다.
          정밀 ETA는 GPU 콜드스타트·큐로 부정확해 대략적 예상 범위만 안내한다. */}
      <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-ink-4">
        <ElapsedTime startedAt={startedAt} />
        <span>영상 길이에 따라 보통 1~3분 정도 걸립니다.</span>
      </div>
      <Pipeline category={category} phase={phase} completed={completed} />
    </ResultPage>
  );
}

/** 분석 시작 시각(서버 기준)부터 경과 시간을 1초마다 갱신해 표시. 시작 시각 미상이면 표시 안 함. */
function ElapsedTime({ startedAt }: { startedAt: string | null }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  if (!startedAt) return null;
  const sec = Math.max(0, Math.floor((now - new Date(startedAt).getTime()) / 1000));
  const mm = Math.floor(sec / 60);
  const ss = String(sec % 60).padStart(2, "0");
  return (
    <span className="font-medium text-ink-3" style={{ fontFeatureSettings: "'tnum'" }}>
      {mm}:{ss} 경과
    </span>
  );
}
