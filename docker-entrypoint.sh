#!/bin/sh
set -e

# Apply migrations before serving unless explicitly skipped.
if [ "${SKIP_MIGRATIONS:-0}" != "1" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

exec "$@"