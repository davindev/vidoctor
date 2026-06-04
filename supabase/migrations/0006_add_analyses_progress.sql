-- analyses.progress: 분석 진행 상태(노드별 완료 목록 + 단계).
-- 적용: Supabase Dashboard → SQL Editor에 붙여넣고 실행
-- (CLI 사용 시: supabase db push)

-- ============================================================
-- 진행률 폴링용 컬럼
-- ============================================================
-- 분석을 클라이언트 연결에서 분리하면서, 진행 상태를 영속화해 재접속 시 복원한다.
-- 형식: {"completed_nodes": ["transcribe", ...], "phase": "running"}
-- 기존 row는 default '{}'로 채워지고, 코드는 .get(...)으로 안전 접근.

alter table public.analyses
    add column progress jsonb not null default '{}'::jsonb;
