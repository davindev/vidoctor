"use client";

import { useState } from "react";
import { deleteAnalysis, type Category } from "@/lib/api";
import { DeleteModal } from "./DeleteModal";
import { ErrorBanner } from "./ErrorBanner";
import { ResultHeader, ResultPage } from "./ResultHeader";

interface Props {
  category: Category | null;
  filename: string | null;
  errorMessage: string;
  analysisId: string | null;
  onNewAnalysis: () => void;
  onDeleted: () => void;
}

/** 분석 실패 전용 화면 — 진행률 UI 없이 실패 사유와 다음 행동(새 분석·삭제)만 보여준다. */
export function FailedView({
  category,
  filename,
  errorMessage,
  analysisId,
  onNewAnalysis,
  onDeleted,
}: Props) {
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  return (
    <ResultPage>
      <ResultHeader trailing="실패" filename={filename} category={category} />
      <section className="rounded-[14px] border border-[#EAC8C2] bg-[#FBEFEB] px-8 py-12 text-center">
        <div className="mx-auto grid h-11 w-11 place-items-center rounded-full bg-[#F2D5CD] text-[20px] font-semibold text-danger">
          !
        </div>
        <h2 className="mt-4 text-[16px] font-semibold tracking-[-0.01em] text-[#6E3A33]">
          분석을 완료하지 못했습니다
        </h2>
        <p className="mx-auto mt-2 max-w-[460px] text-[13.5px] leading-[1.65] text-[#8A5249]">
          {errorMessage}
        </p>
        <div className="mt-6 flex items-center justify-center gap-2">
          <button
            type="button"
            onClick={onNewAnalysis}
            className="rounded-full bg-accent px-5 py-2.5 text-[13px] font-medium text-white transition-[background] duration-150 hover:bg-[#C26344]"
          >
            새 영상 분석
          </button>
          {/* 실패한 분석도 영상이 R2에 업로드돼 있으므로 정리할 수 있게 한다. */}
          {analysisId && (
            <button
              type="button"
              onClick={() => setConfirmOpen(true)}
              className="rounded-full border border-[#E5C2BD] bg-transparent px-5 py-2.5 text-[13px] font-medium text-danger transition-[background,border-color] duration-150 hover:border-danger hover:bg-danger-tint"
            >
              삭제
            </button>
          )}
        </div>
      </section>

      {deleteError && (
        <ErrorBanner
          icon={false}
          message={`삭제 실패: ${deleteError}`}
          className="mt-3"
        />
      )}

      {confirmOpen && analysisId && (
        <DeleteModal
          filename={filename ?? "이 분석"}
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
