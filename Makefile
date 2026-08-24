.DEFAULT_GOAL := dev

.PHONY: dev up down build logs clean

# Runs both servers locally (no Docker) — the docker-based `up` below has a
# few unresolved build issues on this machine (nnsight's C extension needs a
# full build toolchain the slim image doesn't have) and isn't the fast path
# right now.
dev:
	./dev.sh

up:
	docker compose up --build

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

clean:
	docker compose down -v
