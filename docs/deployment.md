# Deployment

This guide covers running DevRAG in production. The recommended path is Docker
Compose; a bare-metal/systemd path is documented for hosts without Docker.

## Prerequisites

- Python 3.13+ (bare metal) or Docker with Compose (recommended)
- PostgreSQL 13+ with the `pgvector` extension (or the bundled Compose file)
- An OpenAI API key for embeddings/generation, **or** local embedding +
  offline generation providers (see [Provider configuration](#provider-configuration))

## Docker Compose (recommended)

The repository ships a `Dockerfile` and `docker-compose.yml` that run the API
against a `pgvector/pgvector:pg16` database.

```bash
# Set your provider configuration (optional; defaults shown)
export OPENAI_API_KEY=sk-...

# Build and start the API + database
docker compose up --build -d

# Check the health probes
curl http://127.0.0.1:8000/api/v1/health/live
curl http://127.0.0.1:8000/api/v1/health/ready
```

What the compose file does:

- Starts PostgreSQL with pgvector, a healthcheck, and a named volume for data
  persistence.
- Builds the API image, wires `DATABASE_URL` to the `db` service, and starts it
  only once the database is healthy.
- Runs `alembic upgrade head` on container start via
  `docker-entrypoint.sh` (set `SKIP_MIGRATIONS=1` to disable).
- Mounts a named volume at `/home/devrag/.cache` so downloaded embedding and
  reranker models persist across restarts.

### Offline / local providers (no API key)

```bash
export EMBEDDING_PROVIDER=local
export GENERATION_PROVIDER=none
export RERANKER_PROVIDER=none
docker compose up --build -d
```

Note: with `EMBEDDING_PROVIDER=local` the vector dimension is 384; drop and
recreate the database (or adjust the migration) so `EMBEDDING_DIMENSION` and
the schema match.

### Logs

The API writes structured JSON logs to stdout; view them with:

```bash
docker compose logs -f api
```

## Provider configuration

Provider selection is driven by environment variables read by
`app/core/settings.py`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `OPENAI_API_KEY` | — | Required for OpenAI embeddings and generation |
| `EMBEDDING_DIMENSION` | `1536` | Must match the DB vector column (1536 OpenAI / 384 local) |
| `RERANKER_PROVIDER` | `cross_encoder` | `cross_encoder`, `llm`, or `none` |
| `GENERATION_PROVIDER` | `openai` | `openai` or `none` (canned/offline) |
| `DATABASE_URL` | local dev URL | SQLAlchemy connection string |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | Structured logging |

Never bake secrets into the image; pass them at runtime via the environment or
an orchestrator secret store.

## Bare-metal deployment

```bash
# 1. Install
python -m venv /opt/devrag/venv
/opt/devrag/venv/bin/pip install .

# 2. Configure (systemd EnvironmentFile or shell env)
export DATABASE_URL=postgresql+psycopg://user:pass@db-host:5432/devrag
export OPENAI_API_KEY=sk-...

# 3. Migrate
/opt/devrag/venv/bin/alembic upgrade head

# 4. Serve
/opt/devrag/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### systemd unit

```ini
[Unit]
Description=DevRAG API
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=exec
User=devrag
WorkingDirectory=/opt/devrag
EnvironmentFile=/etc/devrag.env
ExecStart=/opt/devrag/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

## Database migrations

Migrations are managed with Alembic (`alembic/versions/`). Apply them on every
deploy before starting new code:

```bash
alembic upgrade head
```

`docker-compose` and the container entrypoint do this automatically; bare-metal
deployments should run it explicitly (e.g. as a deploy step or `ExecStartPre`).

## Health checks and orchestration

- **Liveness:** `GET /api/v1/health/live` — returns 200 whenever the process is
  up; use for restart decisions.
- **Readiness:** `GET /api/v1/health/ready` — returns 200 only when all
  dependency probes (database, required API keys) pass, 503 otherwise with
  per-dependency details; use for load-balancer/rollout gating.
- **Version:** `GET /api/v1/version` — returns the running service version; use
  for release verification after a deploy.

Example Kubernetes probes:

```yaml
livenessProbe:
  httpGet: { path: /api/v1/health/live, port: 8000 }
  initialDelaySeconds: 10
readinessProbe:
  httpGet: { path: /api/v1/health/ready, port: 8000 }
  initialDelaySeconds: 15
```

## Reverse proxy

Terminate TLS in a reverse proxy and forward to the API. Example with nginx:

```nginx
server {
    listen 443 ssl;
    server_name devrag.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

The API binds to `0.0.0.0:8000` by default (`python -m app.main`).

## Versioning and releases

- The service version is reported by `/api/v1/version` and comes from
  `app_version` in settings (kept in sync with `pyproject.toml`).
- Releases are tagged `vX.Y.Z` and summarized in `CHANGELOG.md`.
- After deploying a tagged release, confirm with
  `curl /api/v1/version` that the expected version is running.
