"use client";

/** 분석 삭제 확인 모달 — 영상·분석 결과가 영구 제거되므로 한 번 확인받는다. */
export function DeleteModal({
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
