# Go Weather MCP Server

基于 Go 实现的 MCP (Model Context Protocol) 天气服务，通过 JSON-RPC over stdio 与主 Agent 通信。

## 构建

```bash
# 需要 Go 1.21+
go build -o weather-mcp-server .

# Windows
go build -o weather-mcp-server.exe .
```

## 运行

```bash
# 设置 API Key（可选，未设置时返回演示数据）
export QWEATHER_API_KEY=your-key

# 启动（stdio 模式，由 MCP Client 自动拉起）
./weather-mcp-server
```

## MCP 协议

实现 `2024-11-05` 版本 MCP 协议：
- `initialize` — 握手+能力协商
- `tools/list` — 返回可用工具列表
- `tools/call` — 执行工具调用

## 工具

| 工具 | 说明 |
|------|------|
| `get_weather` | 获取指定城市的实时天气 |

## 项目结构

```
go-weather-server/
├── main.go                    # 入口，注册 MCP tools
├── internal/
│   ├── mcp/
│   │   ├── protocol.go        # JSON-RPC 类型定义
│   │   └── server.go          # Stdio server 实现
│   └── weather/
│       └── client.go          # 和风天气 API 封装
├── go.mod
└── Dockerfile
```
