#!/bin/sh

TAG="fastvecdb:latest"
NAME="fastvecdb"

docker build -t "$TAG" .
docker run \
    -e POSTGRES_PASSWORD=postgres \
    -p 5432:5432 \
    --rm \
    --name "$NAME" \
    "$TAG"
