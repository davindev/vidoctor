-- Vidoctor 초기 스키마
-- 적용: Supabase Dashboard → SQL Editor에 붙여넣고 실행
-- (CLI 사용 시: supabase db push)

-- ============================================================
-- videos: 업로드된 영상 메타
-- ============================================================
create table public.videos (
    id uuid primary key default gen_random_uuid(),
    storage_path text not null,
    category text not null check (category in ('lecture', 'vlog', 'other')),
    duration_sec numeric,
    status text not null default 'pending'
        check (status in ('pending', 'analyzing', 'completed', 'failed')),
    created_at timestamptz not null default now()
);

-- ============================================================
-- analyses: 분석 실행 한 건 (한 영상이 여러 번 재분석될 수 있음)
-- ============================================================
create table public.analyses (
    id uuid primary key default gen_random_uuid(),
    video_id uuid not null references public.videos(id) on delete cascade,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    cost_usd numeric,
    error text,
    metadata jsonb not null default '{}'::jsonb
);
create index analyses_video_id_idx on public.analyses (video_id);

-- ============================================================
-- findings: 5차원 분석 결과 통합
-- payload에는 차원별 디테일 (filler 텍스트, CPS 값, frame index 등)
-- ============================================================
create table public.findings (
    id uuid primary key default gen_random_uuid(),
    analysis_id uuid not null references public.analyses(id) on delete cascade,
    dimension text not null
        check (dimension in ('filler', 'cps', 'dead_zone', 'gaze', 'content_gap')),
    start_sec numeric not null,
    end_sec numeric not null,
    severity text check (severity in ('low', 'mid', 'high')),
    payload jsonb not null
);
create index findings_analysis_dimension_idx
    on public.findings (analysis_id, dimension);
create index findings_analysis_start_idx
    on public.findings (analysis_id, start_sec);

-- ============================================================
-- suggestions: 개선 제안 (LLM 생성)
-- finding_ids로 어떤 finding(들)을 근거로 한 제안인지 연결
-- ============================================================
create table public.suggestions (
    id uuid primary key default gen_random_uuid(),
    analysis_id uuid not null references public.analyses(id) on delete cascade,
    finding_ids uuid[] not null default '{}'::uuid[],
    text text not null,
    priority int not null default 0
);
create index suggestions_analysis_id_idx on public.suggestions (analysis_id);

-- ============================================================
-- Storage bucket: 영상 파일
-- ============================================================
insert into storage.buckets (id, name, public)
values ('videos', 'videos', false)
on conflict (id) do nothing;

-- 참고: RLS 비활성. service_role key로만 접근 (단일 사용자 데모 가정).
