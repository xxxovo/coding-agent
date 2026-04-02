import os
import sys
import subprocess
from pathlib import Path
from langchain_core.tools import tool

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

# ================= 私有格式化方法 =================
def _format_results(results):
    res_text = ""
    for i, res in enumerate(results):
        res_text += f"\n--- 检索命中 {i+1} ---\n"
        res_text += f"名称 (Name): {res.get('name')} | 类型 (Type): {res.get('type')}\n"
        res_text += f"文件路径 (File): {res.get('file')}\n"
        res_text += f"置信度 (Score): {res.get('retrieval_score')} | 决策归因: {res.get('retrieval_reason')}\n"
        res_text += f"源代码:\n```python\n{res.get('code')}\n```\n"
    return res_text


# ================= LangChain 规范 Tools =================

@tool
def retrieve_code(query: str, top_k: int = 3) -> str:
    """负责 RAG 检索: Search for relevant code blocks and their semantics based on a natural language query."""
    if not retriever:
        return "Error: RAG index not initialized. Ask user to run build_index.py first."
    
    results = retriever.search(query, top_k=top_k, expand_hops=0)
    return "🔥 RAG 纯搜查结果:\n" + _format_results(results)

@tool
def expand_code_graph(query: str, top_k: int = 3, hops: int = 1) -> str:
    """需要时加入 codegraph 扩展的结果: 仅输出根据中心节点进行向外发散跳跃扩散出的新节点（滤除主召回节点）。"""
    if not retriever:
        return "Error: RAG index not initialized. Ask user to run build_index.py first."
        
    # 1. 先拿到主召回节点作为种子 (Seed Nodes)
    seed_results = retriever.search(query, top_k=top_k)
    
    # 2. 独立调用 expand_context 进行星状跳跃扩张
    # retriever.expand_context 内部逻辑会自动过滤掉原来就存在的 seed 节点，只返回纯纯的拓展增量
    expanded_results = retriever.expand_context(seed_results, expand_hops=hops)
    
    if not expanded_results:
        return f"🕸️ Code Graph (Hops={hops}) 深度扩展失败：没有发现新的关联网状上下文。"
        
    return f"🕸️ Code Graph (Hops={hops}) 深度扩展【单独新增】上下文结果:\n" + _format_results(expanded_results)

@tool
def write_file(filepath: str, content: str) -> str:
    """将新建代码文件: Write entirely new contents to a specific local file. 'filepath' is relative to project root."""
    path = project_root / filepath
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote new file to {filepath}"

@tool
def apply_patch(filepath: str, old_string: str, new_string: str) -> str:
    """修改代码文件: Replace EXACT 'old_string' with 'new_string' in an existing file. 'old_string' must exist line-by-line."""
    path = project_root / filepath
    if not path.exists():
        return f"Error: Target patch file {filepath} does not exist."
        
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    if old_string not in content:
        return f"Error: apply_patch FAILED. 'old_string' not found perfectly inside {filepath}. Please verify your old_string whitespace."
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.replace(old_string, new_string))
        
    return f"Successfully applied Patch format logic on {filepath}"

@tool
def run_code(command: str) -> str:
    """运行代码并返回执行结果: Execute an arbitrary bash/terminal command (E.g. pytest, python script) at project root."""
    
    # 包装命令：先初始化 conda bash hook，激活 agent 环境，再执行原命令
    wrapped_command = f'eval "$(conda shell.bash hook)" && conda activate agent && {command}'
    
    try:
        result = subprocess.run(
            wrapped_command,
            shell=True,
            executable="/bin/bash",
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=45
        )
        return (f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}\n"
                f"EXIT CODE: {result.returncode}")
    except subprocess.TimeoutExpired:
        return "Error: Shell command execution timed out after 45 seconds."
    except Exception as e:
        return f"System Error executing Bash Shell command: {e}"

@tool
def list_dir(directory: str = ".") -> str:
    """列出项目指定目录的内容: List the contents of a directory relative to the project root."""
    target_path = project_root / directory
    if not target_path.exists() or not target_path.is_dir():
        return f"Error: Directory '{directory}' does not exist."
    try:
        items = os.listdir(target_path)
        items.sort()
        result = [f"Contents of {directory}:"]
        for item in items:
            item_path = target_path / item
            if item_path.is_dir():
                result.append(f"📁 {item}/")
            else:
                result.append(f"📄 {item}")
        return "\n".join(result)
    except Exception as e:
        return f"Error reading directory: {e}"