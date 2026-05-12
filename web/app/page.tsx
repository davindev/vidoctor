"use client";

import { useCallback, useEffect, useState } from "react";
import { AnalyzingView } from "@/components/AnalyzingView";
import { IdleForm } from "@/components/IdleForm";
import { ResultView } from "@/components/ResultView";
import { Sidebar } from "@/components/Sidebar";
import {
  fetchAnalyses,
  type AnalysisListItem,
  type Category,
  type CategoryChoice,
} from "@/lib/api";
import {
  postAnalyze,
  type AnalyzeEvent,
  type AnalyzeSource,
  type AnalyzingPhase,
} from "@/lib/sse";

type AppState =
  | { kind: "idle"; lastError: string | null }
  | {
      kind: "analyzing";
      category: Category | null;
      filename: string | null;
      phase: AnalyzingPhase;
      completed: Set<string>;
      errorMessage: string | null;
    }
  | { kind: "result"; analysisId: string };

export default function Home() {
  const [items, setItems] = useState<AnalysisListItem[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [state, setState] = useState<AppState>({ kind: "idle", lastError: null });

  const refreshHistory = useCallback(async () => {
    try {
      const list = await fetchAnalyses();
      setItems(list);
      setHistoryError(null);
    } catch (e) {
      console.error("failed to load history", e);
      setHistoryError(
        e instanceof Error ? e.message : "이전 분석 목록을 불러오지 못했습니다.",
      );
    }
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  const handleSelect = (id: string) => {
    if (state.kind === "analyzing") return;
    setState({ kind: "result", analysisId: id });
  };

  const handleNewAnalysis = () => {
    if (state.kind === "analyzing") return;
    setState({ kind: "idle", lastError: null });
  };

  const handleSubmit = async (
    source: AnalyzeSource,
    category: CategoryChoice,
  ) => {
    // URL 흐름은 제목·카테고리 모두 도착 이벤트로 채워지므로 초기엔 null.
    const initialFilename =
      source.kind === "file" ? source.file.name : null;
    const initialCategory: Category | null =
      category === "auto" ? null : category;
    const initialPhase: AnalyzingPhase =
      source.kind === "url"
        ? "downloading"
        : category === "auto"
          ? "classifying"
          : "uploading";

    setState({
      kind: "analyzing",
      category: initialCategory,
      filename: initialFilename,
      phase: initialPhase,
      completed: new Set(),
      errorMessage: null,
    });

    // SSE는 onEvent 콜백으로만 결과를 push하므로 종료 사유를 외부에서 추적. 객체로 감싸
    // 두는 이유는 TS가 callback 안에서의 변수 mutation을 추적하지 않아 string-literal
    // narrowing이 깨지기 때문 (let 변수만 쓰면 후행 비교가 unreachable로 잡힘).
    const status: { exit: "complete" | "error" | "stream-closed" } = {
      exit: "stream-closed",
    };

    try {
      await postAnalyze({
        source,
        category,
        onEvent: (ev: AnalyzeEvent) => {
          setState((prev) => {
            if (prev.kind !== "analyzing") return prev;
            switch (ev.type) {
              case "status":
                return prev.phase === ev.phase ? prev : { ...prev, phase: ev.phase };
              case "metadata":
                return prev.filename === ev.filename
                  ? prev
                  : { ...prev, filename: ev.filename };
              case "category":
                return prev.category === ev.category
                  ? prev
                  : { ...prev, category: ev.category };
              case "uploaded":
                return prev.phase === "running" ? prev : { ...prev, phase: "running" };
              case "node": {
                if (prev.completed.has(ev.name)) return prev;
                const next = new Set(prev.completed);
                next.add(ev.name);
                return { ...prev, completed: next };
              }
              case "complete":
                status.exit = "complete";
                return { kind: "result", analysisId: ev.analysis_id };
              case "error":
                status.exit = "error";
                return { ...prev, errorMessage: ev.message };
              default:
                return prev;
            }
          });
        },
      });
    } catch (e) {
      status.exit = "error";
      setState({
        kind: "analyzing",
        category: initialCategory,
        filename: initialFilename,
        phase: initialPhase,
        completed: new Set(),
        errorMessage: e instanceof Error ? e.message : String(e),
      });
    }

    if (status.exit === "complete") {
      void refreshHistory();
      return;
    }
    // 에러 또는 unexpected stream close — 사용자가 메시지를 읽을 수 있도록 idle로
    // 복귀하되 lastError를 들고 가서 IdleForm 상단에 배너로 노출.
    setState((prev) => {
      if (prev.kind !== "analyzing") return prev;
      const msg =
        prev.errorMessage ??
        (status.exit === "stream-closed"
          ? "분석이 완료되기 전에 연결이 끊어졌습니다."
          : "분석 중 오류가 발생했습니다.");
      return { kind: "idle", lastError: msg };
    });
  };

  const handleDeleted = async () => {
    setState({ kind: "idle", lastError: null });
    await refreshHistory();
  };

  const selectedId = state.kind === "result" ? state.analysisId : null;
  const sidebarDisabled = state.kind === "analyzing";

  return (
    <div className="grid min-h-screen grid-cols-[300px_1fr]">
      <Sidebar
        items={items}
        selectedId={selectedId}
        disabled={sidebarDisabled}
        loadError={historyError}
        onSelect={handleSelect}
        onNewAnalysis={handleNewAnalysis}
      />
      <main>
        {state.kind === "idle" && (
          <IdleForm
            disabled={false}
            lastError={state.lastError}
            onSubmit={handleSubmit}
          />
        )}
        {state.kind === "analyzing" && (
          <AnalyzingView
            category={state.category}
            filename={state.filename}
            phase={state.phase}
            completed={state.completed}
            errorMessage={state.errorMessage}
          />
        )}
        {state.kind === "result" && (
          <ResultView
            analysisId={state.analysisId}
            onDeleted={handleDeleted}
          />
        )}
      </main>
    </div>
  );
}
