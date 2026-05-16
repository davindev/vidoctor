-- DB-level 데이터 무결성 가드.
-- 적용: Supabase Dashboard → SQL Editor에 붙여넣고 실행
-- (CLI 사용 시: supabase db push)

-- ============================================================
-- 시간·금액 invariant
-- ============================================================
-- Python 검증과 belt+suspenders. constraint 이름 명시로 위반 시 메시지에서 즉시 식별.

alter table public.findings
    add constraint findings_time_order_check
    check (end_sec >= start_sec);

alter table public.analyses
    add constraint analyses_time_order_check
    check (finished_at is null or finished_at >= started_at);

alter table public.analyses
    add constraint analyses_cost_nonneg_check
    check (cost_usd is null or cost_usd >= 0);
