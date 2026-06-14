"""
MCP 多服务器客户端 — 管理多个 MCP Server 的 JSON-RPC over stdio 连接

支持的 MCP Server:
  - knowledge-base: RAG部分/mcp_server.py（7 个知识库工具）
  - weather:       go-weather-server（Go 实现，1 个天气工具）
"""

import json
import os
import subprocess
import sys
import threading
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.logger_handler import logger


class MCPServerConnection:
    """单个 MCP Server 的连接管理"""

    def __init__(self, name: str, command: list[str], cwd: str = None):
        self.name = name
        self.command = command
        self.cwd = cwd or config.PROJECT_ROOT
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._request_id = 0

    def connect(self) -> None:
        logger.info("MCP[%s] 连接中: %s", self.name, " ".join(self.command))
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=self.cwd,
        )
        # 后台消费 stderr
        threading.Thread(target=self._consume_stderr, daemon=True).start()

        # MCP 初始化握手
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "zhiclean-agent", "version": "2.0.0"},
            },
        )
        self._send_notification("notifications/initialized", {})
        logger.info("MCP[%s] 握手完成", self.name)

    def _consume_stderr(self) -> None:
        try:
            for line in self.process.stderr:
                if line.strip():
                    logger.debug("MCP[%s] stderr: %s", self.name, line.strip())
        except Exception:
            pass

    def _send_notification(self, method: str, params: dict) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self.process.stdin.write(json.dumps(msg) + "\n")
        self.process.stdin.flush()

    def _request(self, method: str, params: dict) -> dict:
        self._request_id += 1
        request = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            raise ConnectionError(f"MCP[{self.name}] 连接已断开")
        return json.loads(response_line)

    def _ensure_connected(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.connect()

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        with self._lock:
            self._ensure_connected()
            try:
                result = self._request("tools/call", {"name": tool_name, "arguments": arguments})
            except Exception as e:
                return f"[错误] MCP[{self.name}] 调用失败: {e}"

        if "error" in result:
            return f"[MCP错误] {result['error']}"

        content = result.get("result", {}).get("content", [])
        return content[0].get("text", "") if content else "[MCP] 空结果"

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("MCP[%s] 已关闭", self.name)


class MCPClientManager:
    """多 MCP Server 管理器 —— 应用级单例"""

    _instance: Optional["MCPClientManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._connections: dict[str, MCPServerConnection] = {}
        self._register_servers()

    def _register_servers(self) -> None:
        # Knowledge MCP Server (Python)
        self._connections["knowledge"] = MCPServerConnection(
            name="knowledge",
            command=["python", config.MCP_SERVER_PATH],
            cwd=os.path.dirname(config.MCP_SERVER_PATH),
        )

        # Weather MCP Server (Go binary)
        go_binary = os.path.join(config.PROJECT_ROOT, "go-weather-server", "weather-mcp-server")
        if os.name == "nt":
            go_binary += ".exe"

        if os.path.exists(go_binary):
            self._connections["weather"] = MCPServerConnection(
                name="weather",
                command=[go_binary],
            )
        else:
            # Fallback: run via go run
            go_main = os.path.join(config.PROJECT_ROOT, "go-weather-server", "main.go")
            if os.path.exists(go_main):
                self._connections["weather"] = MCPServerConnection(
                    name="weather",
                    command=["go", "run", go_main],
                )
            else:
                logger.warning("Go weather server 未找到，天气工具将通过 Python fallback 调用")

    def get_connection(self, name: str) -> MCPServerConnection | None:
        return self._connections.get(name)

    def call_tool(self, server: str, tool: str, args: dict) -> str:
        conn = self._connections.get(server)
        if conn is None:
            return f"[错误] MCP Server '{server}' 未注册"
        return conn.call_tool(tool, args)

    def connect_all(self) -> None:
        for name, conn in self._connections.items():
            try:
                conn.connect()
            except Exception as e:
                logger.error("MCP[%s] 连接失败: %s", name, e)

    def close_all(self) -> None:
        for conn in self._connections.values():
            conn.close()


# 全局单例
_mcp_manager: MCPClientManager | None = None


def get_mcp_manager() -> MCPClientManager:
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager()
    return _mcp_manager
