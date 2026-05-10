"use client";

import { useCallback, useEffect, useState } from "react";
import { AnalyzingView } from "@/components/AnalyzingView";
import { IdleForm } from "@/components/IdleForm";
import { ResultView } from "@/components/ResultView";
import { Sidebar } from "@/components/Sidebar";
import { fetchAnalyses, type AnalysisListItem, type Category } from "@/lib/api";
import { postAnalyze, type AnalyzeEvent } from "@/lib/sse";

type AppState =
  | { kind: "idle" }
  | {
      kind: "analyzing";
      category: Category;
      filename: string | null;
      uploadDone: boolean;
      completed: Set<string>;
      errorMessage: string | null;
    }
  | { kind: "result"; analysisId: string };

export default function Home() {
  const [items, setItems] = useState<AnalysisListItem[]>([]);
  const [state, setState] = useState<AppState>({ kind: "idle" });

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
    setState({ kind: "idle" });
  };

  const handleSubmit = async (file: File, category: Category) => {
    setState({
      kind: "analyzing",
      category,
      filename: file.name,
      uploadDone: false,
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
        file,
        category,
        onEvent: (ev: AnalyzeEvent) => {
          setState((prev) => {
            if (prev.kind !== "analyzing") return prev;
            switch (ev.type) {
              case "uploaded":
                return { ...prev, uploadDone: true };
              case "node": {
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
        filename: file.name,
        uploadDone: false,
        completed: new Set(),
        errorMessage: e instanceof Error ? e.message : String(e),
      });
    }

    if (status.exit === "complete") {
      void refreshHistory();
      return;
    }
    // 에러 또는 unexpected stream close — 사용자 잠금 방지하기 위해 idle 복귀.
    setState((prev) => (prev.kind === "analyzing" ? { kind: "idle" } : prev));
  };

  const handleDeleted = async () => {
    setState({ kind: "idle" });
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
          <IdleForm disabled={false} onSubmit={handleSubmit} />
        )}
        {state.kind === "analyzing" && (
          <AnalyzingView
            category={state.category}
            filename={state.filename}
            uploadDone={state.uploadDone}
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
