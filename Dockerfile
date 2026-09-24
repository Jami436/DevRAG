FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/devrag

RUN useradd --create-home --uid 1000 devrag

COPY pyproject.toml README.md LICENSE ./
COPY app ./app
COPY devrag ./devrag
RUN pip install --upgrade pip && pip install .

COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts
COPY data ./data
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER devrag
EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "app.main"]