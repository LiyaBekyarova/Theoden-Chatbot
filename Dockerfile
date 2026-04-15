# ── Stage 1: install dependencies ──────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the NLTK tokenizer data that nltk_utils.py needs at runtime
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"


# ── Stage 2: runtime image ──────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/nltk_data /root/nltk_data

# Copy only what the app needs to run
COPY src/ ./src/
COPY data/ ./data/

# Run uvicorn from inside src/ so Python can find main, chat, model, etc.
WORKDIR /app/src

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
