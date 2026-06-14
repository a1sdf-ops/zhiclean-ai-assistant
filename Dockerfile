# ── 阶段 1: 编译 Go Weather MCP Server ──
FROM golang:1.21-alpine AS go-builder
WORKDIR /go-app
COPY go-weather-server/go.mod go-weather-server/main.go ./
COPY go-weather-server/internal/ ./internal/
RUN go build -o weather-mcp-server .

# ── 阶段 2: Python 运行时 ──
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 从 Go 构建阶段复制编译好的 binary
COPY --from=go-builder /go-app/weather-mcp-server ./go-weather-server/weather-mcp-server

RUN mkdir -p /app/data/chroma_db /app/data/chat_history /app/data/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
