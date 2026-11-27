-- LOOP Personal Archive & Reminders — Supabase PostgreSQL Schema
-- Run this in Supabase Dashboard -> SQL Editor -> New Query
-- Then create a PRIVATE Storage bucket named: loop-vault

create extension if not exists "pgcrypto";

-- 1. USER PROFILES, PERMISSIONS & SUBSCRIPTION TIER
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null,
  full_name text,
  plan_tier text not null default 'free_pilot' check (plan_tier in ('free_pilot', 'plus', 'pro', 'family')),
  max_active_records integer not null default 25,
  razorpay_customer_id text,
  razorpay_subscription_id text,
  permissions_granted boolean not null default false,
  permissions jsonb not null default '{
    "documentAccess": true,
    "notifications": true,
    "microphone": false,
    "leadTime": "2 days before"
  }'::jsonb,
  channels jsonb not null default '{
    "emailDigest": true,
    "pushNotifications": true,
    "smsFallback": false
  }'::jsonb,
  consents jsonb not null default '{
    "localParsing": true,
    "patternInsights": true
  }'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Auto-create a profile row when a user signs up via Supabase Auth
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, email, full_name)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'full_name', split_part(new.email, '@', 1))
  );
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- 2. ARCHIVED VAULT RECORDS
create table if not exists public.records (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  title text not null,
  issuer text not null,
  amount text not null default '₹0',
  numeric_amount numeric not null default 0,
  due_date date not null,
  category text not null check (category in ('Bill', 'Booking', 'Warranty')),
  status text not null default 'Confirmed' check (status in ('Confirmed', 'Paid')),
  reminder_timing text not null default '2 days before',
  source_type text not null default 'PDF Document',
  file_path text, -- Private path inside Supabase Storage bucket 'loop-vault'
  doc_header text,
  doc_subheader text,
  reference_code text,
  sections jsonb not null default '[]'::jsonb,
  snippet text not null default '',
  reminder_sent_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists idx_records_user_due on public.records (user_id, status, due_date);

-- 3. PRICING WAITLIST (Plus / Pro / Family)
create table if not exists public.waitlist (
  id uuid primary key default gen_random_uuid(),
  email text not null,
  tier text not null check (tier in ('LOOP Plus', 'LOOP Pro', 'LOOP Family')),
  user_id uuid references public.profiles(id) on delete set null,
  created_at timestamptz not null default now()
);

-- 4. PILOT TESTER FEEDBACK
create table if not exists public.pilot_feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.profiles(id) on delete set null,
  email text,
  screen text,
  rating integer check (rating between 1 and 5),
  message text not null,
  created_at timestamptz not null default now()
);

-- 5. ROW LEVEL SECURITY (Strict per-user privacy)
alter table public.profiles enable row level security;
alter table public.records enable row level security;
alter table public.waitlist enable row level security;
alter table public.pilot_feedback enable row level security;

create policy "Users view own profile" on public.profiles
  for select using (auth.uid() = id);
create policy "Users update own profile" on public.profiles
  for update using (auth.uid() = id);

create policy "Users manage own records" on public.records
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "Anyone can join waitlist" on public.waitlist
  for insert with check (true);
create policy "Users submit own feedback" on public.pilot_feedback
  for insert with check (true);
