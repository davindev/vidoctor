"use client";

import { useCallback, useEffect, useState } from "react";
import { AnalyzingView } from "@/components/AnalyzingView";
import { IdleForm } from "@/components/IdleForm";
import { ResultView } from "@/components/ResultView";
import { Sidebar } from "@/components/Sidebar";
import { fetchAnalyses, type AnalysisListItem, type Category } from "@/lib/api";
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
      category: Category;
      filename: string | null;
      phase: AnalyzingPhase;
      completed: Set<string>;
      errorMessage: string | null;
    }
  | { kind: "result"; analysisId: string };

export default function Home() {
  const [items, setItems] = useState<AnalysisListItem[]>([]);
  const [state, setState] = useState<AppState>({ kind: "idle", lastError: null });

  const refreshHistory = useCallback(async () => {
    try {
      const list = await fetchAnalyses();
      setItems(list);
    } catch (e) {
      console.error("failed to load history", e);
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

  const handleSubmit = async (source: AnalyzeSource, category: Category) => {
    // URL 흐름은 제목을 다운로드 완료 후에야 알 수 있어 초기 filename=null.
    // AnalyzingView가 phase==="downloading"이면 "유튜브 URL" placeholder로 보정한다.
    const initialFilename =
      source.kind === "file" ? source.file.name : null;
    const initialPhase: AnalyzingPhase =
      source.kind === "url" ? "downloading" : "uploading";

    setState({
      kind: "analyzing",
      category,
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
        category,
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
