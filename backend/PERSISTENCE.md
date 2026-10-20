# Demo sketch history

## Default demo behavior

The Vercel frontend saves generated versions, image data URLs, descriptions, witness ages, refinement instructions, timestamps, and age-analysis results in browser `localStorage` under `ai-police-sketch-history-v1`. It loads this browser history before requesting optional backend history. Matching entries are deduplicated by sketch ID and version, with the local image data preferred.

No external database or storage service is required. The backend defaults to `SKETCH_STORAGE_BACKEND=local`; if Supabase is selected without its URL or server key, it logs a warning and falls back to local storage so FastAPI still starts.

The existing backend filesystem store remains useful during an active Render instance. Browser-local image data is also sent to the existing age-analysis route if the backend cannot find the requested version.

Browser history is per browser profile and origin. It survives refreshes, browser restarts, and Render restarts/redeploys, but it does not sync between browsers/devices and can be removed by clearing site data. Browser storage has finite quota; if it fills or is disabled, the page warns that new history may not persist. Backend-local files can disappear during Render restarts, so they are not a durable backup.

## Optional Supabase backend persistence

Supabase support remains available for a future setup. It is not needed for the AME534 demo. To enable it later, create a Supabase project, run `backend/sql/supabase_schema.sql`, create a private `sketch-images` Storage bucket, then set these backend-only Render variables:

```text
SKETCH_STORAGE_BACKEND=supabase
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<server-only service role key>
SUPABASE_STORAGE_BUCKET=sketch-images
```

Never put `SUPABASE_SERVICE_ROLE_KEY` in Vercel or a `NEXT_PUBLIC_*` variable. Images remain behind FastAPI; the frontend does not need a Supabase credential.

To copy existing Render/local files after configuring Supabase, put the variables in `backend/.env` and run from the repository root:

```sh
python backend/scripts/migrate_local_sketches_to_supabase.py
```

The migration preserves version metadata and age-analysis results and does not delete the local source.
