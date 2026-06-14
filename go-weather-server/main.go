// go-weather-server: MCP stdio server providing real-time weather data.
//
// Build:  go build -o weather-mcp-server .
// Usage:  Configure as an MCP server in the agent's config, or run
//         python agent部分/mcp_client.py to connect to it.
//
// Protocol: JSON-RPC over stdio (Model Context Protocol 2024-11-05).
package main

import (
	"log"
	"os"

	"go-weather-server/internal/mcp"
	"go-weather-server/internal/weather"
)

func main() {
	// Log to stderr so stdout stays clean for JSON-RPC
	log.SetOutput(os.Stderr)
	log.SetPrefix("[weather-mcp] ")

	server := mcp.NewServer("weather-mcp-server", "1.0.0")

	server.RegisterTool(
		mcp.Tool{
			Name:        "get_weather",
			Description: "获取指定城市的实时天气（温度、湿度、风向风速等）。参数为中文城市名如'深圳'、'北京'。",
			InputSchema: mcp.InputSchema{
				Type: "object",
				Properties: map[string]mcp.Property{
					"city": {
						Type:        "string",
						Description: "中文城市名称，如'北京'、'深圳'",
					},
				},
				Required: []string{"city"},
			},
		},
		func(args map[string]any) (string, error) {
			city, ok := args["city"].(string)
			if !ok || city == "" {
				city = "北京"
			}
			log.Printf("查询天气: city=%s", city)
			result := weather.GetWeather(city)
			return result, nil
		},
	)

	log.Println("MCP Weather Server 启动 (stdio)")

	if err := server.Run(); err != nil {
		log.Fatalf("服务异常退出: %v", err)
	}
}
