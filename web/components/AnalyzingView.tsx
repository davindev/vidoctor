"use client";

import type { Category } from "@/lib/api";
import type { AnalyzingPhase } from "@/lib/sse";
import { Pipeline } from "./Pipeline";
import { ResultHeader, ResultPage } from "./ResultHeader";

interface Props {
  category: Category | null;
  filename: string | null;
  phase: AnalyzingPhase;
  completed: Set<string>;
  errorMessage: string | null;
}

export function AnalyzingView({
  category,
  filename,
  phase,
  completed,
  errorMessage,
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
      <Pipeline
        category={category}
        phase={phase}
        completed={completed}
        errorMessage={errorMessage}
      />
    </ResultPage>
  );
}
