-- Persistent sketch metadata. Image bytes live in the private Storage bucket.
create table if not exists public.sketches (
    sketch_id text primary key,
    metadata jsonb not null,
    updated_at timestamptz not null default now()
);

alter table public.sketches enable row level security;
revoke all on table public.sketches from anon, authenticated;
grant all on table public.sketches to service_role;
