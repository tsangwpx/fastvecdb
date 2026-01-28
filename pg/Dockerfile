FROM postgres:18

RUN set -eux; \
    apt-get update; \
    apt-get install --yes --no-install-recommends \
        postgresql-18-pgvector

RUN echo "CREATE EXTENSION IF NOT EXISTS vector;" > /docker-entrypoint-initdb.d/init_pgvector.sql
