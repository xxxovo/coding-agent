import os
import sys
import subprocess
import uuid
import json
from pathlib import Path
from typing import Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 将根目录放入 Python 搜索路径，以便导入 rag 模块
current_dir = Path(__file__).resolve().parent
agent_root = current_dir.parent  # 这是 coding-agent 目录，用于导包和索引

# 对于代码写入和执行，应该定位到目标测试项目（例如 fastapi-realworld-example-app）
target_repo_path = agent_root / "test_repo" / "fastapi-realworld-example-app"
project_root = target_repo_path if target_repo_path.exists() else agent_root

sys.path.append(str(agent_root))

from rag.code_indexer import CodeIndexer
from rag.graph_builder import CodeGraphBuilder
from rag.retriever import HybridRetriever

# ================= 全局初始化 RAG 引擎 =================
rag_persist_dir = agent_root / "rag" / "data" / "demo_index"
retriever = None

try:
    if rag_persist_dir.exists():
        code_indexer = CodeIndexer.load(rag_persist_dir)
        graph_builder = CodeGraphBuilder(code_indexer.code_units)
        graph_builder.build()
        retriever = HybridRetriever(code_indexer.code_units, graph_builder, code_indexer)
        print("✅ 工具链 RAG 引擎加载成功！")
    else:
        print(f"⚠️ 工具链 RAG 索引未找到: {rag_persist_dir}，请确保已构建索引。")
except Exception as e:
    print(f"❌ 工具链 RAG 索引加载失败 - {e}")

# ================= 工具输出超长截断与持久化 (ToolResultStorage) =================
TOOL_RESULTS_DIR = project_root / ".memory" / "tool_results"

def store_long_result(content: str, max_chars: int = 3000) -> str:
    """
    如果工具的输出结果过长，截断它并把全量日志写到硬盘，
    给大模型返回一个包含截断信息的预览及查阅指针。
    """
    if not isinstance(content, str):
        content = str(content)
        
    if len(content) <= max_chars:
        return content
        
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())[:8]
    filepath = TOOL_RESULTS_DIR / f"result_{file_id}.txt"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    preview = content[:max_chars]
    marker = (
        f"\n\n...[OUTPUT TRUNCATED]...\n"
        f"⚠️ 工具输出超过 {max_chars} 字符，全量完整输出已保存至:\n"
        f"➡️ {filepath}\n"
        f"如果上面的预览信息不够，请使用 `read_file` 工具读取该文件以获取完整堆栈/日志。"
    )
    return preview + marker

def _format_results(results):
    res_text = ""
    for i, res in enumerate(results):
        res_text += f"\n--- 检索命中 {i+1} ---\n"
        res_text += f"名称 (Name): {res.get('name')} | 类型 (Type): {res.get('type')}\n"
        res_text += f"文件路径 (File): {res.get('file')}\n"
        res_text += f"源代码:\n```python\n{res.get('code')}\n```\n"
    return res_text

# ================= Pydantic Schemas 强类型校验 =================

class RagSearchArgs(BaseModel):
    query: str = Field(..., description="Natural language query to search code logic.")
    top_k: int = Field(3, description="Number of results to return.")

class ExpandGraphArgs(BaseModel):
    query: str = Field(..., description="Query to find the center dependencies.")
    top_k: int = Field(3, description="Seed nodes.")
    hops: int = Field(1, description="Graph expansion hops.")

class FileWriteArgs(BaseModel):
    filepath: str = Field(..., description="Absolute path or relative to project root to create/overwrite a file.")
    content: str = Field(..., description="The entire new content of the file.")

class FileEditArgs(BaseModel):
    filepath: str = Field(..., description="File path to edit.")
    old_string: str = Field(..., description="EXACT existing string to replace, including whitespaces/indentation.")
    new_string: str = Field(..., description="The replacement string.")

class FileReadArgs(BaseModel):
    filepath: str = Field(..., description="File to read.")
    start_line: int = Field(1, description="Line number to start reading from (1-based index).")
    end_line: int = Field(None, description="Line number to end reading at (inclusive). Leave empty to read to the end.")

class BashArgs(BaseModel):
    command: str = Field(..., description="Bash command to execute (e.g., pytest, npm run build).")

class ListDirArgs(BaseModel):
    directory: str = Field(".", description="Directory path to list content.")

class SaveMemoryArgs(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    memory_type: Literal['user', 'feedback', 'project', 'reference'] = Field(...)
    content: str = Field(...)

class ReadMemoryArgs(BaseModel):
    filename: str = Field(...)

class WebFetchArgs(BaseModel):
    url: str = Field(..., description="URL to fetch content from using curl/python.")

class ToolSearchArgs(BaseModel):
    query: str = Field(..., description="Keywords to search for tools.")
    max_results: int = Field(3, description="Max tool results to return.")

# ================= 权限管控与安全沙盒 (Permission & Guardrails) =================

DANGEROUS_BASH_KEYWORDS = [
    "rm -", "sudo ", "docker ", "git push", "drop table", "truncate", 
    "mkfs", "chown", "chmod", "systemctl", "kill ", "reboot"
]

SENSITIVE_FILES = [
    ".env", ".git", "credentials", "secrets", "config.py"
]

def verify_action(action_type: str, detail: str) -> bool:
    """HITL (Human-in-the-loop) 人工授权拦截器"""
    print(f"\n" + "="*55)
    print(f"🛡️  [Security Guardrail] 拦截到高权限/高危操作！")
    print(f"   [操作类型]: {action_type}")
    print(f"   [操作详情]: {detail}")
    print("="*55)
    while True:
        try:
            ans = input("❓ 是否授权 AI 执行此操作？[y/N]: ").strip().lower()
            if ans in ['y', 'yes']:
                print("✅ 用户已授权，继续执行...\n")
                return True
            elif ans in ['n', 'no', '']:
                print("🚫 用户已拒绝，操作中止。\n")
                return False
            else:
                print("请输入 y 或 n。")
        except (KeyboardInterrupt, EOFError):
            print("\n🚫 异常中断，默认拒绝。")
            return False

def check_bash_safety(command: str) -> bool:
    cmd_lower = command.lower()
    return any(kw in cmd_lower for kw in DANGEROUS_BASH_KEYWORDS)

def check_file_safety(filepath: str) -> bool:
    path_lower = str(filepath).lower()
    return any(sf in path_lower for sf in SENSITIVE_FILES)


# ================= LangChain 规范 Tools =================

@tool(args_schema=RagSearchArgs)
def retrieve_code(query: str, top_k: int = 3) -> str:
    """负责 RAG 检索: Search for relevant code blocks and their semantics based on a natural language query."""
    if not retriever: return "Error: RAG index not initialized."
    results = retriever.search(query, top_k=top_k, expand_hops=0)
    return store_long_result("🔥 RAG 纯搜查结果:\n" + _format_results(results))

@tool(args_schema=ExpandGraphArgs)
def expand_code_graph(query: str, top_k: int = 3, hops: int = 1) -> str:
    """需要时加入 codegraph 扩展的结果: 仅输出根据中心节点进行向外发散跳跃扩散出的新节点。"""
    if not retriever: return "Error: RAG index not initialized."
    seed_results = retriever.search(query, top_k=top_k)
    expanded_results = retriever.expand_context(seed_results, expand_hops=hops)
    if not expanded_results: return f"🕸️ Code Graph (Hops={hops}) 深度扩展失败。"
    return store_long_result(f"🕸️ Code Graph (Hops={hops}) 深度扩展结果:\n" + _format_results(expanded_results))

@tool(args_schema=FileWriteArgs)
def file_write(filepath: str, content: str) -> str:
    """Write entirely new contents to a specific local file."""
    if check_file_safety(filepath):
        if not verify_action("Sensitive File Write", f"Target: {filepath}"):
            return f"❌ Permission Denied: User rejected modifications to sensitive system file `{filepath}`."
            
    path = project_root / filepath
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote new file to {filepath}"

@tool(args_schema=FileEditArgs)
def file_edit(filepath: str, old_string: str, new_string: str) -> str:
    """
    修改代码文件 (Replace EXACT string). 
    取代旧的 apply_patch，采用更稳妥的局部替换逻辑。
    要求 old_string 必须能完全精确匹配文件中的某一段文本。
    """
    if check_file_safety(filepath):
        if not verify_action("Sensitive File Edit", f"Target: {filepath}\nOld string:\n{old_string}"):
            return f"❌ Permission Denied: User rejected modifications to sensitive system file `{filepath}`."
            
    path = project_root / filepath
    if not path.exists():
        return f"Error: Target patch file {filepath} does not exist."
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old_string not in content:
        return f"Error: file_edit FAILED. 'old_string' not found inside {filepath}. Please check whitespace."
    new_content = content.replace(old_string, new_string, 1) # Only replace the first occurrence safely
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return f"Successfully applied exact string replacement on {filepath}"

@tool(args_schema=FileReadArgs)
def read_file(filepath: str, start_line: int = 1, end_line: int = None) -> str:
    """按行读取文件内容 (支持分页，自动进行长度截断)"""
    path = project_root / filepath
    if not path.exists():
        return f"Error: File {filepath} does not exist."
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line) if end_line else len(lines)
    
    # 附带行号输出，这对大模型精准匹配极其重要
    content_with_lines = "".join(f"{i+1:4d} | {lines[i]}" for i in range(start_idx, end_idx))
    
    header = f"--- File Overview: {filepath} ({len(lines)} lines total) ---\n"
    if start_line > 1 or end_idx < len(lines):
        header = f"--- File Segment: {filepath} (Lines {start_idx+1}-{end_idx} of {len(lines)}) ---\n"
        
    return store_long_result(header + content_with_lines, max_chars=4000)

@tool(args_schema=BashArgs)
def run_bash(command: str) -> str:
    """运行代码或bash命令并返回执行结果 (带有输出截断和安全性护栏)"""
    if check_bash_safety(command):
        if not verify_action("Destructive Bash Command", command):
            return f"❌ Permission Denied: User rejected the dangerous command `{command}`. Ask the user for guidance or perform safe reads instead."
            
    wrapped_command = f'eval "$(conda shell.bash hook)" && conda activate agent && {command}'
    try:
        result = subprocess.run(
            wrapped_command, shell=True, executable="/bin/bash",
            cwd=project_root, capture_output=True, text=True, timeout=45
        )
        out = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT CODE: {result.returncode}"
        return store_long_result(out, max_chars=3000)
    except subprocess.TimeoutExpired:
        return "Error: Shell command execution timed out after 45 seconds."
    except Exception as e:
        return f"System Error executing Bash Shell command: {e}"

@tool(args_schema=ListDirArgs)
def list_dir(directory: str = ".") -> str:
    """列出项目指定目录的内容"""
    target_path = project_root / directory
    if not target_path.exists() or not target_path.is_dir():
        return f"Error: Directory '{directory}' does not exist."
    try:
        items = os.listdir(target_path)
        items.sort()
        res = f"Contents of {directory}:\n" + "\n".join([f"📁 {i}/" if (target_path/i).is_dir() else f"📄 {i}" for i in items])
        return store_long_result(res)
    except Exception as e:
        return f"Error reading directory: {e}"

@tool(args_schema=WebFetchArgs)
def web_fetch(url: str) -> str:
    """简单发起一个网页或接口请求，获取正文 (使用 curl 降级处理)"""
    try:
        result = subprocess.run(["curl", "-s", "-L", url], capture_output=True, text=True, timeout=15)
        return store_long_result(f"Response from {url}:\n{result.stdout}", max_chars=3000)
    except Exception as e:
        return f"Error fetching web url: {e}"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.memory_manager import save_memory as _save_memory_fn
from utils.memory_manager import read_memory as _read_memory_fn

@tool(args_schema=SaveMemoryArgs)
def save_memory(name: str, description: str, memory_type: str, content: str) -> str:
    """Save persistent memory snippet into the local database."""
    return _save_memory_fn(name, description, memory_type, content)

@tool(args_schema=ReadMemoryArgs)
def read_memory(filename: str) -> str:
    """Read a specific persistent memory file by name."""
    return _read_memory_fn(filename)

# ================= 动态工具发现机制 (Tool Search) =================

CORE_TOOL_NAMES = {"search_tools", "run_bash", "list_dir", "read_file"} 

ALL_TOOLS_DICT = {
    "retrieve_code": retrieve_code,
    "expand_code_graph": expand_code_graph,
    "file_write": file_write,
    "file_edit": file_edit,
    "read_file": read_file,
    "run_bash": run_bash,
    "list_dir": list_dir,
    "save_memory": save_memory,
    "read_memory": read_memory,
    "web_fetch": web_fetch,
}

TOOL_METADATA = {
    "retrieve_code": {"hints": "rag search index query context code snippet semantic"},
    "expand_code_graph": {"hints": "graph dependencies hops relationships topology"},
    "file_write": {"hints": "create new write touch make save filewrite"},
    "file_edit": {"hints": "edit replace modify change patch update fix fileedit"},
    "read_file": {"hints": "cat read view show content open"},
    "run_bash": {"hints": "shell command terminal run run_code execute sh bash"},
    "list_dir": {"hints": "ls tree dir folder list contents glob"},
    "save_memory": {"hints": "remember store long-term preference memory save"},
    "read_memory": {"hints": "recall fetch configuration index check memory load"},
    "web_fetch": {"hints": "curl http get post web websearch scrape browser fetch url"},
}

@tool(args_schema=ToolSearchArgs)
def search_tools(query: str, max_results: int = 3) -> str:
    """搜寻缺失的专业工具并动态挂载 (e.g. 'file_edit', 'rag search', 'web_fetch')"""
    query_terms = [term.lower() for term in query.split()]
    scored_tools = []
    
    for name, tool_func in ALL_TOOLS_DICT.items():
        if name in CORE_TOOL_NAMES: continue
            
        score = 0
        desc = (tool_func.description or "").lower()
        hints = TOOL_METADATA.get(name, {}).get("hints", "").lower()
        
        for term in query_terms:
            if term in name.lower(): score += 10
            elif term in hints: score += 5
            elif term in desc: score += 2
                
        if score > 0:
            scored_tools.append({"name": name, "score": score, "desc": tool_func.description})

    scored_tools.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = scored_tools[:max_results]
    
    if not top_candidates:
        return "⚠️ 未找到匹配的额外工具。你可以尝试使用 run_bash 执行原始命令解决。"
    
    matched_names = [t["name"] for t in top_candidates]
    return (
        f"✅ 已成功搜索并临时激活以下工具，现在你可以直接调用它们了！\n\n"
        f"TOOL_MOUNT_SIGNAL: {json.dumps(matched_names)}\n\n"
        "工具详情介绍：\n" + "\n".join([f"- {t['name']}: {t['desc']}" for t in top_candidates])
    )

ALL_TOOLS_DICT["search_tools"] = search_tools
