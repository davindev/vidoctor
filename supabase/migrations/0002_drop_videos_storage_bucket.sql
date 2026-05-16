-- Supabase Storage 'videos' bucket 정리. 영상 storage는 Cloudflare R2(S3 호환)로 분리.
-- 적용: Supabase Dashboard → SQL Editor에 붙여넣고 실행
-- (CLI 사용 시: supabase db push)
--
-- bucket 삭제 전에 객체부터 비워야 FK 제약을 안 만난다 (storage.objects → storage.buckets).
-- 이미 비어 있거나 bucket이 없으면 두 statement 모두 0건 처리되어 멱등.

delete from storage.objects where bucket_id = 'videos';
delete from storage.buckets where id = 'videos';
