FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod ./
COPY . .
RUN go build -o weather-mcp-server .

FROM alpine:3.19
WORKDIR /app
COPY --from=builder /app/weather-mcp-server .
CMD ["./weather-mcp-server"]
