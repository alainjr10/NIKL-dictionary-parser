-- Run once in the Supabase SQL editor (Dashboard → SQL).
-- Mirrors topik_master_v1.json / db/topik_grammar.sqlite so the app can
-- fetch the same grammar records remotely.

create table if not exists public.topik_grammar (
  "grammarId" integer primary key,
  name text not null,
  "englishMeaning" text,
  "briefDescription" text,
  "featuredExample" text,
  "featuredTranslation" text,
  "longerExplanation" text,
  level text not null,
  "sentenceExamples" jsonb not null default '[]'::jsonb,
  "similarToIds" integer[] not null default '{}'::integer[]
);

create index if not exists idx_topik_grammar_level on public.topik_grammar (level);
create index if not exists idx_topik_grammar_name on public.topik_grammar (name);

alter table public.topik_grammar enable row level security;

drop policy if exists "Public read topik_grammar" on public.topik_grammar;
create policy "Public read topik_grammar"
  on public.topik_grammar
  for select
  to anon, authenticated
  using (true);

-- Writes go through the service role (bypasses RLS). No public insert/update/delete.
