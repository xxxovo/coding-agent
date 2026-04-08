import asyncio
import json
import threading
from typing import List, Dict, Any, Callable
from langchain_core.tools import StructuredTool
from pydantic import create_model

try:
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.session import ClientSession
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

class MCPToolAdapter:
    """
    Model Context Protocol (MCP) Client Adapter for LangChain.
    实现了 跨进程工具调用的完美解耦。允许从外部 Node.js/Rust MCP Server 动态抓取工具，
    并转换成当前 Agent 能懂的 LangChain @tool (StructuredTool)。
    """
    def __init__(self):
        self._servers: Dict[str, StdioServerParameters] = {}
        self.loaded_tools: Dict[str, StructuredTool] = {}

    def register_server(self, name: str, command: str, args: List[str]):
        """注册一个外部 MCP 服务器（如 sqlite-mcp, github-mcp）"""
        if not MCP_AVAILABLE:
            print("⚠️ MCP SDK 未安装，请执行 `pip install mcp` 启用跨平台工具协议。")
            return
        
        self._servers[name] = StdioServerParameters(command=command, args=args)
        print(f"🔌 MCP Server [{name}] 已注册: {command} {' '.join(args)}")

    async def _fetch_and_wrap_tools(self, server_name: str) -> List[StructuredTool]:
        """连接 MCP 服务器，握手，拉取工具列表，将其热包装为 LangChain Tool"""
        server_params = self._servers[server_name]
        
        langchain_tools = []
        try:
            # 建立跨进程标准输入输出流 (stdio 通信)
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # 拉取该服务器暴露的所有 Tools
                    mcp_tools_response = await session.list_tools()
                    
                    for mcp_tool in mcp_tools_response.tools:
                        # 解析 MCP 的 JSON Schema，动态生成 Pydantic Model
                        tool_args_schema = self._create_pydantic_model_from_json_schema(
                            mcp_tool.name, mcp_tool.inputSchema
                        )
                        
                        # 制造一个闭包包装器，用于执行真正的跨进程 RPC
                        async def _tool_caller(*args, _mcp_session=session, _tool_name=mcp_tool.name, **kwargs) -> str:
                            result = await _mcp_session.call_tool(_tool_name, kwargs)
                            return json.dumps(result.content, ensure_ascii=False)

                        # 构建 LangChain 规范工具
                        lc_tool = StructuredTool.from_function(
                            func=_tool_caller,
                            name=f"{server_name}_{mcp_tool.name}",
                            description=mcp_tool.description or f"Tool {mcp_tool.name} hosted on MCP {server_name}",
                            args_schema=tool_args_schema
                        )
                        langchain_tools.append(lc_tool)
                        self.loaded_tools[lc_tool.name] = lc_tool
                        
            print(f"✅ 从 MCP Server [{server_name}] 成功热加载 {len(langchain_tools)} 个工具。")
        except Exception as e:
            print(f"❌ 连接 MCP Server [{server_name}] 失败: {e}")
            
        return langchain_tools

    def _create_pydantic_model_from_json_schema(self, tool_name: str, json_schema: Dict[str, Any]):
        """将 MCP 标准的 JSON Schema 转为 Pydantic 强类型"""
        fields = {}
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])
        
        for field_name, attr in properties.items():
            field_type = str
            if attr.get("type") == "integer": field_type = int
            elif attr.get("type") == "boolean": field_type = bool
            elif attr.get("type") == "array": field_type = list
            
            default_val = ... if field_name in required else None
            fields[field_name] = (field_type, default_val)
            
        return create_model(f"MCP_{tool_name}_Args", **fields)

# ================= 使用示例 =================
# mcp_adapter = MCPToolAdapter()
# mcp_adapter.register_server("sqlite", "npx", ["-y", "@modelcontextprotocol/server-sqlite", "database.db"])
# mcp_adapter.register_server("github", "npx", ["-y", "@modelcontextprotocol/server-github"])
#
# 在你的 graph.py 初始化前加载：
# loop = asyncio.get_event_loop()
# new_tools = loop.run_until_complete(mcp_adapter._fetch_and_wrap_tools("sqlite"))
# ALL_TOOLS_DICT.update({t.name: t for t in new_tools})
