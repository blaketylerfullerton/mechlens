FROM python:3.11-slim

WORKDIR /app

# nnsight compiles a small C extension at install time — python:3.11-slim
# has no compiler by default.
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
# torch's default PyPI wheel drags in the full CUDA/cuDNN stack even on a
# CPU-only container. A separate `pip install torch --index-url .../cpu`
# before this doesn't stick — the very next `pip install -r requirements.txt`
# re-resolves torch from the default index (since requirements.txt lists it
# too) and reinstalls the CUDA build over it. Making the CPU index the
# *primary* index for this single install call (with PyPI as a fallback for
# everything else) is what actually sticks.
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
