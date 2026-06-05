"use client";

import { useCallback, useEffect, useState } from "react";
import { AnalyzingView } from "@/components/AnalyzingView";
import { IdleForm } from "@/components/IdleForm";
import { ResultView } from "@/components/ResultView";
import { Sidebar } from "@/components/Sidebar";
import {
  fetchAnalyses,
  fetchStatus,
  type AnalysisListItem,
  type Category,
  type CategoryChoice,
} from "@/lib/api";
import { basename } from "@/lib/format";
import {
  postAnalyze,
  type AnalyzeSource,
  type AnalyzingPhase,
} from "@/lib/analyze";

// 시드된 샘플 분석 2건 — 사용자가 삭제하지 못하도록 삭제 UI를 숨긴다.
// 그 외 영상은 사용자가 직접 올린 것이라 삭제 가능.
const SAMPLE_ANALYSIS_IDS = new Set([
  "615a025e-422b-464f-aa1c-74b069924c9b",
  "9b2fdee7-2959-41be-b290-14c007368d32",
]);

type AppState =
  | { kind: "idle"; lastError: string | null }
  | {
      kind: "analyzing";
      category: Category | null;
      filename: string | null;
      phase: AnalyzingPhase;
      completed: Set<string>;
      errorMessage: string | null;
      // POST 응답 전엔 null(업로드 중), 응답 후 채워지면 폴링이 시작된다.
      analysisId: string | null;
    }
  | { kind: "result"; analysisId: string };

export default function Home() {
  const [items, setItems] = useState<AnalysisListItem[]>([]);
  // 업로드(파일 전송) 동안은 서버 row가 아직 없어 사이드바가 빈다. 그 구간을 메우는
  // 낙관적 항목 — POST 응답 후 실제 row로 교체된다.
  const [pendingItem, setPendingItem] = useState<AnalysisListItem | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  // cold-start 재시도 동안 사이드바가 "기록 없음"으로 잘못 보이지 않도록 초기 로드를 추적.
  const [historyLoading, setHistoryLoading] = useState(true);
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
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  const handleSelect = (id: string) => {
    // 낙관적 항목(업로드 중, 아직 서버 row 없음)은 상세 조회가 불가하므로 업로드 진행
    // 화면으로 복귀시킨다. POST가 끝나면 폴링 useEffect가 analysisId를 채워 이어받는다.
    if (pendingItem && id === pendingItem.id) {
      setState({
        kind: "analyzing",
        category: pendingItem.category,
        filename: pendingItem.filename,
        phase: pendingItem.filename ? "uploading" : "downloading",
        completed: new Set(),
        errorMessage: null,
        analysisId: null,
      });
      return;
    }
    // 분석은 Modal이 연결과 무관하게 끝까지 돌므로, 진행 중이어도 자유롭게 이동 가능.
    // 진행 중(status='analyzing') 항목을 고르면 폴링으로 진행률을 이어본다.
    const item = items.find((it) => it.id === id);
    if (item?.status === "analyzing") {
      setState({
        kind: "analyzing",
        category: item.category,
        filename: item.filename ?? basename(item.storage_path),
        phase: "running",
        completed: new Set(),
        errorMessage: null,
        analysisId: id,
      });
    } else {
      setState({ kind: "result", analysisId: id });
    }
  };

  const handleNewAnalysis = () => {
    setState({ kind: "idle", lastError: null });
  };

  const handleSubmit = async (
    source: AnalyzeSource,
    category: CategoryChoice,
  ) => {
    // URL 흐름은 제목·카테고리가 분석 시작 시점엔 미정이므로 초기엔 null.
    const initialFilename = source.kind === "file" ? source.file.name : null;
    const initialCategory: Category | null =
      category === "auto" ? null : category;
    const initialPhase: AnalyzingPhase =
      source.kind === "url"
        ? "downloading"
        : category === "auto"
          ? "classifying"
          : "uploading";

    // 업로드 시작 시점부터 사이드바에 항목이 보이도록 낙관적 항목을 먼저 띄운다.
    const tempId = crypto.randomUUID();
    setPendingItem({
      id: tempId,
      started_at: new Date().toISOString(),
      finished_at: null,
      error: null,
      category: initialCategory,
      storage_path: null,
      status: "analyzing",
      filename: initialFilename,
    });
    setState({
      kind: "analyzing",
      category: initialCategory,
      filename: initialFilename,
      phase: initialPhase,
      completed: new Set(),
      errorMessage: null,
      analysisId: null,
    });

    try {
      // POST는 업로드·분류·Modal 위임까지만 대기하고 analysis_id를 반환한다.
      // 이후 진행은 아래 폴링 useEffect가 analysisId를 보고 이어받는다.
      const id = await postAnalyze({ source, category });
      // analysisId===null 가드: POST 대기 중 사용자가 다른 진행 중 분석으로 전환했을 수
      // 있으므로, 아직 id가 안 박힌 바로 그 analyzing 상태일 때만 채운다(엉뚱한 분석에
      // id를 덮어쓰는 race 방지). 전환했다면 이 분석은 사이드바에서 다시 열 수 있다.
      setState((prev) =>
        prev.kind === "analyzing" && prev.analysisId === null
          ? { ...prev, analysisId: id, phase: "running" }
          : prev,
      );
      // 실제 row를 먼저 받아 사이드바에 반영한 뒤 낙관적 항목을 거둔다(깜빡임 방지).
      await refreshHistory();
      setPendingItem(null);
    } catch (e) {
      setPendingItem(null);
      setState({
        kind: "idle",
        lastError: e instanceof Error ? e.message : String(e),
      });
    }
  };

  const handleDeleted = async () => {
    setState({ kind: "idle", lastError: null });
    await refreshHistory();
  };

  // 진행 중 분석을 3초 간격으로 폴링 — 직접 시작한 분석과 재접속해 연 분석 모두 처리.
  // 작업은 서버에서 끝까지 도므로, 이 훅은 진행률을 DB에서 읽어 화면에 반영할 뿐이다.
  const analyzingId = state.kind === "analyzing" ? state.analysisId : null;
  useEffect(() => {
    if (analyzingId === null) return;
    const id = analyzingId;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (!stopped) timer = setTimeout(poll, 3000);
    };

    const poll = async () => {
      if (stopped) return;
      // 탭이 백그라운드면 폴링(네트워크)을 멈추고 다음 tick에 재확인.
      if (document.visibilityState === "hidden") {
        schedule();
        return;
      }
      try {
        const s = await fetchStatus(id);
        if (stopped) return;
        const nodes = s.progress.completed_nodes ?? [];
        setState((prev) =>
          prev.kind === "analyzing" && prev.analysisId === id
            ? { ...prev, completed: new Set(nodes) }
            : prev,
        );
        if (s.status === "completed") {
          stopped = true;
          setState({ kind: "result", analysisId: id });
          void refreshHistory();
          return;
        }
        if (s.status === "failed") {
          stopped = true;
          setState((prev) =>
            prev.kind === "analyzing" && prev.analysisId === id
              ? { ...prev, errorMessage: s.error ?? "분석에 실패했습니다." }
              : prev,
          );
          void refreshHistory();
          return;
        }
      } catch {
        // 일시적 네트워크·cold-start 오류는 무시하고 다음 tick에 재시도.
      }
      schedule();
    };

    void poll(); // 진입 즉시 1회 — 재접속 시 진행률 깜빡임 최소화.
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [analyzingId, refreshHistory]);

  // 사이드바에 진행 중(analyzing) 항목이 있으면, 그 분석을 보고 있지 않아도 목록을
  // 주기적으로 갱신해 완료·실패를 반영한다. 개별 폴링은 현재 보는 분석 1건만 추적하므로
  // 백그라운드로 도는 다른 분석의 완료는 이 목록 폴링이 잡는다.
  const hasAnalyzing = items.some((it) => it.status === "analyzing");
  useEffect(() => {
    if (!hasAnalyzing) return;
    const timer = setInterval(() => {
      if (document.visibilityState !== "hidden") void refreshHistory();
    }, 4000);
    return () => clearInterval(timer);
  }, [hasAnalyzing, refreshHistory]);

  const selectedId =
    state.kind === "result"
      ? state.analysisId
      : state.kind === "analyzing"
        ? (state.analysisId ?? pendingItem?.id ?? null)
        : null;
  // 업로드 중(서버 row 생성 전)에는 낙관적 항목을 사이드바 맨 위에 끼워 보여준다.
  const sidebarItems = pendingItem ? [pendingItem, ...items] : items;
  // 분석이 연결과 분리돼 백그라운드에서 도므로 진행 중에도 사이드바 이동을 막지 않는다.
  const sidebarDisabled = false;

  return (
    <div className="grid min-h-screen grid-cols-[300px_1fr]">
      <Sidebar
        items={sidebarItems}
        selectedId={selectedId}
        disabled={sidebarDisabled}
        loadError={historyError}
        loading={historyLoading}
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
            deletable={!SAMPLE_ANALYSIS_IDS.has(state.analysisId)}
            onDeleted={handleDeleted}
          />
        )}
      </main>
    </div>
  );
}
