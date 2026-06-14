package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// ToolHandler is a function that handles a tool call.
type ToolHandler func(args map[string]any) (string, error)

// Server represents an MCP stdio server.
type Server struct {
	name     string
	version  string
	tools    []Tool
	handlers map[string]ToolHandler
	reader   *bufio.Reader
	writer   io.Writer
}

// NewServer creates a new MCP server.
func NewServer(name, version string) *Server {
	return &Server{
		name:     name,
		version:  version,
		handlers: make(map[string]ToolHandler),
		reader:   bufio.NewReader(os.Stdin),
		writer:   os.Stdout,
	}
}

// RegisterTool registers a tool and its handler.
func (s *Server) RegisterTool(tool Tool, handler ToolHandler) {
	s.tools = append(s.tools, tool)
	s.handlers[tool.Name] = handler
}

// Run starts the stdio server loop. Blocks until EOF.
func (s *Server) Run() error {
	for {
		line, err := s.reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("read error: %w", err)
		}

		var req Request
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			continue // skip malformed lines
		}

		if req.Method != "" && req.ID == 0 {
			// Notification — no response needed
			continue
		}

		resp := s.handleRequest(&req)
		if resp != nil {
			data, _ := json.Marshal(resp)
			fmt.Fprintf(s.writer, "%s\n", data)
		}
	}
}

func (s *Server) handleRequest(req *Request) *Response {
	switch req.Method {
	case "initialize":
		return s.handleInitialize(req)
	case "tools/list":
		return s.handleToolsList(req)
	case "tools/call":
		return s.handleToolsCall(req)
	default:
		return &Response{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &Error{Code: -32601, Message: fmt.Sprintf("unknown method: %s", req.Method)},
		}
	}
}

func (s *Server) handleInitialize(req *Request) *Response {
	return &Response{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: InitializeResult{
			ProtocolVersion: "2024-11-05",
			Capabilities: ServerCapabilities{
				Tools: &ToolsCapability{ListChanged: false},
			},
			ServerInfo: ServerInfo{
				Name:    s.name,
				Version: s.version,
			},
		},
	}
}

func (s *Server) handleToolsList(req *Request) *Response {
	if s.tools == nil {
		s.tools = []Tool{} // ensure JSON "tools":[] not "tools":null
	}
	return &Response{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result:  map[string]any{"tools": s.tools},
	}
}

func (s *Server) handleToolsCall(req *Request) *Response {
	params := &ToolCallParams{}
	// Re-marshal params from the raw map
	raw, _ := json.Marshal(req.Params)
	if err := json.Unmarshal(raw, params); err != nil {
		return &Response{
			JSONRPC: "2.0", ID: req.ID,
			Error: &Error{Code: -32602, Message: fmt.Sprintf("invalid params: %v", err)},
		}
	}

	handler, ok := s.handlers[params.Name]
	if !ok {
		return &Response{
			JSONRPC: "2.0", ID: req.ID,
			Error: &Error{Code: -32602, Message: fmt.Sprintf("unknown tool: %s", params.Name)},
		}
	}

	result, err := handler(params.Arguments)
	if err != nil {
		return &Response{
			JSONRPC: "2.0", ID: req.ID,
			Result: ToolCallResult{
				Content: []TextContent{{Type: "text", Text: fmt.Sprintf("[错误] %v", err)}},
				IsError: true,
			},
		}
	}

	return &Response{
		JSONRPC: "2.0", ID: req.ID,
		Result: ToolCallResult{
			Content: []TextContent{{Type: "text", Text: result}},
		},
	}
}
