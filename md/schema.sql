-- MotherDuck helper schema for the streamlit UI + worker

CREATE TABLE IF NOT EXISTS secrets.yahoo_oauth_tokens (
  id TEXT,
  league_key TEXT,
  token_json TEXT,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ops.import_status (
  job_id TEXT,
  league_key TEXT,
  league_name TEXT,
  season TEXT,
  status TEXT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

