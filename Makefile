# mechlens dev orchestration
#
#   make dev    start the backend (uvicorn) and frontend (vite) together, in the background
#   make down   stop whatever `make dev` started
#
# Logs and PIDs live in .dev/ (gitignored). Backend serves on :8000, the port
# the frontend's VITE_API_BASE_URL points at.
#
# `dev` refuses to start on top of a live run: a second vite cannot have 5173,
# so it silently takes 5174 and you end up with the browser on one port and a
# stale build serving the other.

BACKEND_PORT ?= 8000

# Prefer the repo-root venv if it exists, else fall back to whatever's on PATH.
# The venv path is made absolute because the backend starts from backend/.
VENV := venv
UVICORN := $(if $(wildcard $(VENV)/bin/uvicorn),$(abspath $(VENV)/bin/uvicorn),uvicorn)

RUN := .dev
BACKEND_PID := $(RUN)/backend.pid
FRONTEND_PID := $(RUN)/frontend.pid

.PHONY: dev down check-stale

dev: check-stale $(RUN)
	@if [ ! -f frontend/.env ]; then cp frontend/.env.example frontend/.env; fi
	@echo "starting backend on :$(BACKEND_PORT) ..."
	@cd backend && setsid $(UVICORN) app.service.app:app --port $(BACKEND_PORT) \
		> ../$(RUN)/backend.log 2>&1 & echo $$! > $(BACKEND_PID)
	@echo "starting frontend ..."
	@cd frontend && setsid npm run dev > ../$(RUN)/frontend.log 2>&1 & echo $$! > $(FRONTEND_PID)
	@printf "waiting for the backend to answer on :$(BACKEND_PORT) "
	@for i in $$(seq 1 40); do \
		if curl -sf -m 2 http://localhost:$(BACKEND_PORT)/health > $(RUN)/health.json 2>/dev/null; then \
			echo " up"; break; \
		fi; \
		if ! kill -0 $$(cat $(BACKEND_PID)) 2>/dev/null; then \
			echo " backend exited -- see $(RUN)/backend.log"; exit 1; \
		fi; \
		printf "."; sleep 0.5; \
		if [ $$i = 40 ]; then echo " no answer yet -- see $(RUN)/backend.log"; fi; \
	done
	@echo ""
	@echo "  backend  -> http://localhost:$(BACKEND_PORT)   (log: $(RUN)/backend.log)"
	@echo "             model: $$(sed -n 's/.*\"status\":\"\([a-z]*\)\".*/\1/p' $(RUN)/health.json 2>/dev/null || echo unknown)"
	@echo "  frontend -> $$(grep -o 'http://localhost:[0-9]*' $(RUN)/frontend.log | head -1 || echo 'starting, see the log')   (log: $(RUN)/frontend.log)"
	@echo ""
	@echo "  the model loads in the background: /trace answers 503 until it is"
	@echo "  ready, and the UI retries on its own -- watch GET /health for it."
	@echo ""
	@echo "  tail -f $(RUN)/*.log   to follow output"
	@echo "  make down              to stop both"

# A live pid from an earlier `make dev` means ports are already taken.
check-stale:
	@for f in $(BACKEND_PID) $(FRONTEND_PID); do \
		if [ -f $$f ] && kill -0 $$(cat $$f) 2>/dev/null; then \
			echo "already running: $$f (pid $$(cat $$f)). run 'make down' first."; \
			exit 1; \
		fi; \
	done

# `setsid` in `dev` gives each service its own process group, so `kill -- -pid`
# takes the whole tree down. Signalling only the recorded pid used to leave
# npm's grandchild vite alive and still holding :5173, so the next `make dev`
# quietly landed on 5174 with a stale server on the port you had open.
down:
	@for f in $(BACKEND_PID) $(FRONTEND_PID); do \
		if [ -f $$f ]; then \
			pid=$$(cat $$f); \
			if kill -0 $$pid 2>/dev/null; then \
				echo "stopping $$f (pid $$pid)"; \
				kill -- -$$pid 2>/dev/null || kill $$pid 2>/dev/null || true; \
			fi; \
			rm -f $$f; \
		fi; \
	done
	@echo "down."

$(RUN):
	@mkdir -p $(RUN)
