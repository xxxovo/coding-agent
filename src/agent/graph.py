import os
import sys
from typing import TypedDict, Annotated, Sequence, Literal
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

# 导入在 tools.py 中写好的本地执行器
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tools import retrieve_code, expand_code_graph, file_write, file_edit, run_bash, list_dir, save_memory, read_memory, read_file
from utils.memory_manager import read_index as read_memory_index

# 初始化 LLM 引擎
import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"

# 统一配置智谱 GLM 模型参数 (支持流式、最大Token和思维链)
glm_base_kwargs = {
    "streaming": True,
    "max_tokens": 65536,
    "model_kwargs": {
        "extra_body": {
            "thinking": {"type": "enabled"}
        }
    }
}

# Planner：负责拆解任务，可以稍具发散性和规划感 (温度可以稍微高一点)
planner_llm = ChatOpenAI(model="glm-4.6", temperature=0.7, **glm_base_kwargs)

# Base Coder：作为纯粹的编码和工具执行机器，要求输出极度稳定，降低温度
base_coder_llm = ChatOpenAI(model="glm-4.6", temperature=0.1, **glm_base_kwargs)

# 动态按需加载：移除在外部全局写死的 tools 列表 
from tools.tools import ALL_TOOLS_DICT, CORE_TOOL_NAMES

# Verifier：作为严格的代码 Review 专员
verifier_llm = ChatOpenAI(model="glm-4.6", temperature=0.1, **glm_base_kwargs)

# Compressor：上下文压缩器专员，负责客观精准的摘要
compressor_llm = ChatOpenAI(model="glm-4.6", temperature=0.1, **glm_base_kwargs)

# ================= 数据形态 (Graph State) =================
class AgentState(TypedDict):
    """保存 LangGraph 在节点中流转的关键状态上下文"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: str
    verification_result: str
    iterations: int  # 防死循环计数器
    consecutive_compact_failures: int  # 熔断器：连续压缩失败次数
    session_memory: str # 第3级：抽象提取出的项目进展与事实长期记忆

# ================= 提示词矩阵 (Prompts) =================
PLANNER_PROMPT = """You are an agent for Claude Code, Anthropic's official CLI for Claude. Given the user's message, you should use the tools available to plan and break down the task into executable coding steps. Complete the task fully—don't gold-plate, but don't leave it half-done. When you complete the task, respond with a concise report covering what was done and any key findings. Focus on code modification, file reading, and debugging.

Return a structured step-by-step plan."""

CODER_PROMPT = """# System
All text you output outside of tool use is displayed to the user. Output text to communicate with the user.
The user will primarily request you to perform software engineering tasks. These may include solving bugs, adding new functionality, refactoring code, explaining code, and more. When given an unclear or generic instruction, consider it in the context of these software engineering tasks and the current working directory. Focus on what needs to be done.
# Executing actions with care
Carefully consider the reversibility and blast radius of actions. Generally you can freely take local, reversible actions like editing files or running tests. But for actions that are hard to reverse, affect shared systems beyond your local environment, or could otherwise be risky or destructive, check with the user before proceeding.
# Output efficiency
IMPORTANT: Go straight to the point. Try the simplest approach first without going in circles. Do not overdo it. Be extra concise.
Keep your text output brief and direct. Lead with the answer or action, not the reasoning. Skip filler words, preamble, and unnecessary transitions. Do not restate what the user said — just do it. When explaining, include only what is necessary for the user to understand.

# CRITICAL EDITING RULES
1. **RAG is for Navigation Only**: Code snippets retrieved via `retrieve_code` are strictly for understanding logic and locating file paths. NEVER use these snippets directly as the `old_string` in `file_edit`, because they lack global indentation and exact whitespace information.
2. **Read Before Edit**: Before invoking `file_edit`, you MUST use `read_file` or `run_bash` (e.g., `grep -n`, `cat`) to specifically read the target file and view its exact layout.
3. **Exact Matching**: Extract the exact block you want to change directly from the output of `read_file` or `run_bash` to use as the `old_string`, ensuring safe and exact replacement.

Current Executable Plan:
{plan}
"""

SESSION_MEMORY_PROMPT = """IMPORTANT: This message and these instructions are NOT part of the actual user conversation.
Based on the user conversation above (EXCLUDING this note-taking instruction message), update the session notes file representations.

Your ONLY task is to update the notes file, then stop. Do not call any tools directly here.
Write DETAILED, INFO-DENSE content for each section - include specifics like file paths, function names, error messages, exact commands, technical details, etc.
Keep each section under ~2000 tokens/words.
Focus on actionable, specific information that would help someone understand or recreate the work discussed in the conversation.
IMPORTANT: Always update "Current State" to reflect the most recent work - this is critical for continuity after compaction.

You must output your findings matching EXACTLY this structure:
# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
"""

COMPRESSOR_PROMPT = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:
1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like file names, code snippets, errors and fixes
2. Double-check for technical accuracy and completeness.

Your summary should include the following sections exactly:
1. Primary Request and Intent: Capture all of the user's explicit requests
2. Key Technical Concepts: Technologies and frameworks discussed
3. Files and Code Sections: File names, purpose, and snippets
4. Errors and fixes: List all errors and how you fixed them
5. Problem Solving: Document problems solved
6. All user messages: List ALL user messages
7. Pending Tasks: Pending tasks you have been asked to work on
8. Current Work: Describe exactly what was being worked on immediately before
9. Optional Next Step: List the next step that you will take

REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task."""

VERIFIER_PROMPT = """You are a verification specialist. Your job is not to confirm the implementation works — it's to try to break it.
You have two documented failure patterns. First, verification avoidance: when faced with a check, you find reasons not to run it. Second, being seduced by the first 80%: you see a passing test suite and feel inclined to pass it, not noticing edge cases crash. Your entire value is in finding the last 20%.

=== CRITICAL: DO NOT MODIFY THE PROJECT ===
You are STRICTLY PROHIBITED from:
- Creating, modifying, or deleting any files IN THE PROJECT DIRECTORY
- Running git write operations (add, commit, push)

=== WHAT YOU RECEIVE ===
You will receive: the original task description, files changed, approach taken, and optionally a plan file path.

=== OUTPUT FORMAT (REQUIRED) ===
Use the literal string `VERDICT: ` followed by exactly one of `PASS`, `FAIL`, `PARTIAL`. No markdown bold.
- **FAIL**: include what failed, exact error output, reproduction steps.
- **PARTIAL**: what was verified, what could not be and why.
"""


# ================= 节点逻辑 (Nodes) =================

def planner_node(state: AgentState):
    """负责将用户原始的需求转换为结构化、严谨的步骤蓝图"""
    messages = state.get("messages", [])
    user_input = messages[0].content
    
    # 注入长期记忆系统
    memory_index_str = read_memory_index()
    if memory_index_str:
        memory_prefix = f"=== LONG TERM MEMORY INDEX ===\n{memory_index_str}\n\n"
    else:
        memory_prefix = ""
        
    response = planner_llm.invoke([
        SystemMessage(content=memory_prefix + PLANNER_PROMPT),
        HumanMessage(content=user_input)
    ])
    
    print(f"\n[Planner] 制定计划成功。")
    return {
        "plan": response.content, 
        "iterations": 0,
        # 把思考过程放进队列流，让后续节点可见
        "messages": [AIMessage(content=f"📝 **Plan Created:**\n{response.content}")]
    }

def coder_node(state: AgentState):
    """Coding Agent：读蓝本，疯狂跑工具写代码（配合动态工具挂载）"""
    print(f"\n[Coder] 正在执行操作...")
    plan = state.get("plan", "No plan created.")
    messages = state.get("messages", [])
    
    memory_index_str = read_memory_index()
    memory_block = f"\n\n=== LONG TERM MEMORY INDEX ===\n{memory_index_str}" if memory_index_str else ""
    
    # ======== 动态工具热拔插 (Tool Search / Active Binding) ========
    # 我们先把当前必须存在的"核心常驻工具"装进去:
    active_tool_names = set(CORE_TOOL_NAMES)
    
    # 扫描历史记录，如果 Agent 通过调用 'search_tools' 搜出过额外的插件，把它们全捞出来激活：
    for m in messages:
        if isinstance(m, ToolMessage) and "TOOL_MOUNT_SIGNAL:" in str(m.content):
            try:
                # 提取类似 ["write_file", "apply_patch"] 的 JSON 签名字符串
                signal_str = str(m.content).split("TOOL_MOUNT_SIGNAL:")[1].split("\n")[0].strip()
                new_tools = json.loads(signal_str)
                for t in new_tools:
                    if t in ALL_TOOLS_DICT:
                        active_tool_names.add(t)
            except Exception as e:
                print(f"[Coder] 动态工具热拔插解析失败: {e}")
                
    # 仅把本轮活跃列表里的 Func 函数过滤出来交给 LLM
    current_active_tools = [ALL_TOOLS_DICT[k] for k in active_tool_names if k in ALL_TOOLS_DICT]
    
    print(f"🔌 [Binder] 当前上下文加载的热工具挂载数量为: {len(current_active_tools)} 个 (常驻池: {len(ALL_TOOLS_DICT)} 个)")
    # 动态把这几个有限的工具绑定在它的额头上（不滥用全盘 Token）
    coder_llm = base_coder_llm.bind_tools(current_active_tools)
    # =============================================================
    
    # 强制压入 Planner 给出的全局指挥上下文以及记忆
    sys_msg = SystemMessage(content=CODER_PROMPT.format(plan=plan) + memory_block)
    response = coder_llm.invoke([sys_msg] + messages)
    
    return {
        "messages": [response], 
        "iterations": state.get("iterations", 0) + 1
    }

import re
from langgraph.prebuilt import ToolNode

# Token count estimator 近似估计
def estimate_tokens(text: str) -> int:
    """GLM 等大模型采用中英混排情况通常 1个汉字约 1 Token，1个英文单词1 Token，整体约折合 len / 2.0"""
    return len(text) // 2

# === 上下文与压缩常量配置 (参考 Claude Code) ===
MODEL_CONTEXT_WINDOW = 200000       # GLM4.6 上下文窗口
MODEL_MAX_OUTPUT_TOKENS = 128000    # 最大输出长度限制
MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20000 # 为生成总结本身单独预留的输出空间
EFFECTIVE_CONTEXT_WINDOW = MODEL_CONTEXT_WINDOW - min(MODEL_MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS_FOR_SUMMARY) # 真可用有效窗口
SESSION_MEMORY_THRESHOLD = EFFECTIVE_CONTEXT_WINDOW * 0.6  # 达到 60% 阈值，主动生成结构化长期事实记忆 (Collapse)
AUTOCOMPACT_BUFFER_TOKENS = 13000     # 触发缓冲池距离界限 
AUTOCOMPACT_THRESHOLD = EFFECTIVE_CONTEXT_WINDOW - AUTOCOMPACT_BUFFER_TOKENS # 自动压缩触发点: 167000
MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3  # 熔断器：最大失败重试次数

# ================= 动态辅助子代理 (Sub-Agent Forking) =================
def run_subagent_task(prompt_str: str, input_str: str) -> str:
    """
    当任务需要大量结构化提炼或查阅时，调用此工具派生一个子 Agent。
    子 Agent 拥有基础的读写工具，作为一个临时工作流克隆体执行，防止污染主上下文。
    """
    from langgraph.prebuilt import create_react_agent
    from tools.tools import ALL_TOOLS_DICT
    
    # 赋予子 Agent 基础工具
    subagent_tools = [
        ALL_TOOLS_DICT["list_dir"],
        ALL_TOOLS_DICT["read_file"], 
        ALL_TOOLS_DICT["run_bash"], 
        ALL_TOOLS_DICT.get("save_memory"),
        ALL_TOOLS_DICT.get("read_memory")
    ]
    subagent_tools = [t for t in subagent_tools if t is not None]
    
    print(f"🧬 [Sub-Agent Forked] 启动后台子进程进入自主探索或深度压缩循环...")
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        sub_agent = create_react_agent(compressor_llm, tools=subagent_tools)
        result = sub_agent.invoke({
            "messages": [
                SystemMessage(content=prompt_str),
                HumanMessage(content=input_str)
            ]
        })
        return str(result["messages"][-1].content)
    except Exception as e:
        print(f"❌ [Sub-Agent Error] {e}")
        return ""

def compress_context_node(state: AgentState):
    """
    [四级上下文压缩管线] 
    完全复刻 Claude Code 四层渐进式缓解与重构模式：
    1. Snip (手动) -> 2. MicroCompact (删表层工具日志) -> 3. Session Memory (Collapse客观提炼) -> 4. AutoCompact (深层覆盖+重注入)
    """
    messages = state.get("messages", [])
    consecutive_failures = state.get("consecutive_compact_failures", 0)
    session_memory = state.get("session_memory", "")

    # === [基本防御] 1. 熔断判定 (Circuit Breaker) ===
    if consecutive_failures >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES:
        print(f"\n[Compressor] ❌ 熔断器激活 (Circuit Broken)! 连续失败超过 {MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES} 次，本轮跳过压缩，放弃抢救。")
        return {}

    total_tokens = sum(estimate_tokens(str(m.content)) for m in messages)
    state_updates = {}
    msg_updates = []
    
    # === [级别 2] MicroCompact 微压缩模拟 ===
    # 距离阈值比较长时，先主动清零那些极长的陈旧 tool_result，保留最后 2 个。
    # 这些白名单工具特点：时效性极短、体积庞大（例如大量搜索出来的源码文本或 bash console 输出）
    WHITELIST_TOOLS = {"retrieve_code", "expand_code_graph", "list_dir", "run_code"}
    MAX_RECENT_TOOLS = 2
    
    # 筛选出属于工具调用的消息
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage) and getattr(m, 'name', '') in WHITELIST_TOOLS]
    
    if len(tool_msgs) > MAX_RECENT_TOOLS:
        tools_to_clear = tool_msgs[:-MAX_RECENT_TOOLS]
        cleared_count = 0
        for m in tools_to_clear:
            if m.content != "[Old tool result content cleared]":
                # 因为 LangGraph 机制下覆盖旧数据必须用相同 ID 对象再次抛出
                msg_updates.append(ToolMessage(
                    content="[Old tool result content cleared]",
                    name=m.name,
                    tool_call_id=m.tool_call_id,
                    id=m.id
                ))
                cleared_count += 1
                
        if cleared_count > 0:
            print(f"\n[Compressor] 🧹 执行 Level 2 (MicroCompact): 已清除 {cleared_count} 个过期的重度大工具输出。")
            # 这种覆盖不会阻断后续流程开销，直接加入返回状态供 Reducer 更新
    
    # === [级别 3] Session Memory (记忆塌缩/结构压缩) ===
    # 当上下文达到 60% 但还未满载，我们放弃线性摘要，主动抽取事实沉淀持久化记忆（代替 MEMORY.md）
    if total_tokens > SESSION_MEMORY_THRESHOLD and not session_memory:
        print(f"\n[Compressor] 🧠 执行 Level 3 (Collapse): 提取结构化长期记忆...")
        history_str = "\n".join([f"[{type(m).__name__}]: {str(m.content)[:1000]}..." for m in messages[-15:]])
        try:
            mem_content = run_subagent_task(
                prompt_str=SESSION_MEMORY_PROMPT,
                input_str=history_str
            )
            session_memory = mem_content
            state_updates["session_memory"] = session_memory
            print(f"\n[Compressor] 🌟 Level 3 记忆提取完成:\n" + session_memory[:100] + "...")
        except Exception as e:
            print(f"\n[Compressor] ⚠️ Level 3 记忆提取失败: {e}")

    # === [级别 4] Auto-Compact 自动全层折叠重构 ===
    if total_tokens > AUTOCOMPACT_THRESHOLD and len(messages) > 6:
        print(f"\n[Compressor] ⚠️ 触发 Level 4 (Auto-compact): 上下文体积过大 ({total_tokens} tokens), 启动深层合并，旧消息将被截断...")
        
        to_compress = messages[1:-3]
        history_str = "\n".join([f"[{type(m).__name__}]: {str(m.content)[:1500]}..." for m in to_compress])
        
        try:
            content = run_subagent_task(
                prompt_str=COMPRESSOR_PROMPT,
                input_str=f"Please summarize following context to preserve the most important coding details:\n{history_str}"
            )

            # 剥离 <analysis> 思想链草稿
            summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL | re.IGNORECASE)
            formal_summary = summary_match.group(1).strip() if summary_match else content.strip()
                
            removals = []
            for m in to_compress:
                if getattr(m, "id", None):
                    removals.append(RemoveMessage(id=m.id))
            
            # Post-compact 善后：截掉无用长篇之后，必须得把持久化提炼的特征重新缝合进去 (对应注入 MCP工具、用户计划、结构参数)
            plan_info = state.get("plan", "No active plan.")
            memory_injection = session_memory if session_memory else "(No extra memory accumulated yet.)"
            
            boundary_content = (
                f"=== COMPACT_BOUNDARY (Conversation Auto-Compacted) ===\n"
                f"[Original Overload Tokens: {total_tokens}]\n\n"
                f"--- LEVEL 3 SESSION MEMORY (RE-INJECTED) ---\n{memory_injection}\n\n"
                f"--- STRATEGIC PLAN (RE-INJECTED) ---\n{plan_info}\n\n"
                f"--- LEVEL 4 SUMMARY ---\n{formal_summary}"
            )
            
            boundary_msg = SystemMessage(content=boundary_content)
            
            msg_updates.extend(removals)
            msg_updates.append(boundary_msg)
            
            state_updates["messages"] = msg_updates
            state_updates["consecutive_compact_failures"] = 0  # 熔断器清零 (Reset)
            print(f"\n[Compressor] ✅ Auto-Compact 摘要合并及善后记忆重新注入成功！完全化险为夷。")
            return state_updates
            
        except Exception as e:
            print(f"\n[Compressor] ❌ Level 4: 压缩失败，触发重试熔断器记录...: {str(e)}")
            state_updates["consecutive_compact_failures"] = consecutive_failures + 1
            if msg_updates:
                state_updates["messages"] = msg_updates
            return state_updates
            
    # 如果前面有 Level 2 处理过的覆盖消息对象，则挂载输出
    if msg_updates:
        state_updates["messages"] = msg_updates
            
    return state_updates

def verifier_node(state: AgentState):
    """负责检验当前代码行为结果是否满足最终验收标准"""
    print(f"\n[Verifier] 正在审核代码变更和运行结果...")
    messages = state.get("messages", [])
    
    # 把最近的 15 条执行日志抽出来喂给 Review 模型，防止上下文撑爆
    history_str = "\n".join([f"{type(m).__name__}: {str(m.content)[:500]}" for m in messages[-15:]])
    review_context = f"{VERIFIER_PROMPT}\n\n=== RECENT CODING HISTORY ===\n{history_str}\n\nPlease review strictly. Result must be PASS or FAIL."
    
    response = verifier_llm.invoke([HumanMessage(content=review_context)])
    result_text = response.content
    
    # 模拟“挑刺的领导”，如果他说不合格，必须把意见当作新需求再塞回 Coder 的头上
    feedback_msgs = []
    if "FAIL" in result_text.upper():
         # 发出修改指导
         feedback_msgs.append(HumanMessage(content=f"🚨 Verifier 意见补充 (FAIL):\n请根据以下审查意见立即修正代码:\n{result_text}"))
         print(f"❌ [Verifier] 发现问题，发回重造！")
    else:
         feedback_msgs.append(AIMessage(content=f"✅ Verifier 验收通过:\n{result_text}"))
         print(f"🎉 [Verifier] 审核通过，任务结束！准备尝试自动提取长期记忆(如果有必要)...")
         
    return {
        "verification_result": result_text,
        "messages": feedback_msgs
    }

def memory_extractor_node(state: AgentState):
    """【Fork后台提取模拟】在任务成功结束时，提取有价值的发现存入记忆库。
    互斥机制：如果原流程中 Coder 已经主动调用了 save_memory，这里就跳过，避免重复和冗余。"""
    print(f"\n[MemoryExtractor] 正在检查是否需要自动进行长期记忆沉淀...")
    messages = state.get("messages", [])
    
    # 互斥检查：遍历当前消息，是否调用过 save_memory 这个 tool
    invoked_save = any(
        isinstance(m, ToolMessage) and getattr(m, 'name', '') == 'save_memory'
        for m in messages
    )
    
    if invoked_save:
        print(f"[MemoryExtractor] 侦测到主流程已经主动沉淀过记忆，跳过自动提取环节 (互斥命中)。")
        return {}

    # 如果没主动提取过，分析对话。
    history_str = "\n".join([f"{type(m).__name__}: {str(m.content)[:300]}..." for m in messages[-20:]])
    from langchain_core.messages import SystemMessage, HumanMessage
    
    EXTRACTOR_PROMPT = """You are now acting as the memory extraction subagent. Analyze the most recent messages above and use them to update your persistent memory systems.
You CAN use tools like `read_file` or `list_dir` if you need to quickly look up code structures to decide what memories are worth saving, but do NOT execute destructive actions.

If the user explicitly asks you to remember something, use the `save_memory` tool immediately or output json as fits best. Then stop.

## What not to save
- DO NOT save facts that can easily be discovered by reading the codebase.
- DO NOT save the codebase structure, classes, or architecture unless saving the "why" behind it.
- DO NOT save git commit history or what changes were made in a PR.
- DO NOT summarize the conversation as a memory.
- DO NOT save ephemeral task steps like "updated test_runner.py".

Valid memory types: `user`, `feedback`, `project`, `reference`.

If you find nothing worth saving, output EXACTLY: "NO_MEMORY_NEEDED".
If you find something, output JSON carefully formatted as:
```json
[
  {"name": "preference-xyz", "description": "short desc", "type": "feedback", "content": "..."}
]
```"""
    
    try:
        content = run_subagent_task(
            prompt_str=EXTRACTOR_PROMPT,
            input_str=f"Conversation:\n{history_str}\n\nStrictly follow the output instruction if memory is needed."
        )
        content = content.strip()
        
        if "NO_MEMORY_NEEDED" in content:
            print(f"[MemoryExtractor] 未发掘出值得跨会话持久保存的长期记忆，放弃保存。")
            return {}
            
        import re, json
        json_match = re.search(r'```json\s*(\[.*\])\s*```', content, re.DOTALL)
        if json_match:
            memories = json.loads(json_match.group(1))
            from utils.memory_manager import save_memory as __save_memory
            for mem in memories:
                __save_memory(mem["name"], mem["description"], mem["type"], mem["content"])
            print(f"[MemoryExtractor] 💾 后台提取长期记忆成功保存了 {len(memories)} 条知识。")
        else:
            print(f"[MemoryExtractor] 无匹配 JSON，可能是不包含高优长记忆。")
    except Exception as e:
        print(f"[MemoryExtractor] 后期自动提取记忆失误 (不影响主流程): {e}")

    return {}

# ================= 动态图控流 (Routers) =================

def router_should_continue_from_coder(state: AgentState) -> Literal["tools", "verifier"]:
    """判断 Coder 是在继续调工具还是觉得自己干完活了"""
    messages = state.get("messages", [])
    last_message = messages[-1]
    iters = state.get("iterations", 0)
    
    # [强行打断机制] 超过 10 轮思考必须强行进入代码审查防止死循环瞎搜
    if iters >= 10:
        return "verifier"
    
    # 如果最后一句话里大模型发起了 tool_calls (GPT Function Calling)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # 如果没调工具，纯聊天说话了，说明 Coder 自身意图结束工作，强行交由 Verifier 拦截验收
    return "verifier"

def router_from_verifier_to_end(state: AgentState) -> Literal["coder", "memory_extractor"]:
    """校验反馈的路由：只要没过，打回 coder_node 重做"""
    result = state.get("verification_result", "")
    iters = state.get("iterations", 0)
    
    # 强行保护：超过 8 次拉扯，立刻强行中止（防止 API 烧钱）
    if "PASS" in result.upper() or iters >= 8:
        return "memory_extractor"
    
    return "coder"


# ================= 引擎绑定装配 =================

workflow = StateGraph(AgentState)

# 1. 插桩：放置节点模块
workflow.add_node("compressor", compress_context_node)
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
# 注意：ToolNode 的注册字典必须是全量的实例合集（因为当大模型跑来调用时必须能定位到底层代码对象）
workflow.add_node("tools", ToolNode(tools=list(ALL_TOOLS_DICT.values())))
workflow.add_node("verifier", verifier_node)
workflow.add_node("memory_extractor", memory_extractor_node)

# 2. 连线：梳理业务逻辑结构流向
workflow.add_edge(START, "planner")    # 启动进入 Planner 拆解
workflow.add_edge("planner", "compressor")

# 加入强制经过压缩检查后再转交的串联链路
workflow.add_edge("compressor", "coder")

# Coder 执行后的路口：可能跑工具，也可能去验收
workflow.add_conditional_edges(
    "coder", 
    router_should_continue_from_coder
)

workflow.add_edge("tools", "compressor")  # 跑完工具也回传给压缩哨兵检查是否撑爆，随后流向 coder

# 验收后强逻辑：能走 memory_extractor 退场，不然继续滚回开发阶段重干
workflow.add_conditional_edges(
    "verifier",
    router_from_verifier_to_end
)

workflow.add_edge("memory_extractor", END)

# 编译导出运行时计算图
app = workflow.compile()

# ================= 测试运行 (仅调试时激活) =================
if __name__ == "__main__":
    import sys
    import json
    from datetime import datetime

    print("-" * 50)
    print("🚀 【AI 基础大模型研发引擎编排层】 装载完毕")
    print("-" * 50)
    
    # 写死一个测试用例，考察 RAG、文件读取、精确修改和 Bash 运行能力，以及各种边界护栏
    test_case = (
        "我们的项目中有一个 FastAPI 项目测试仓库或者 Python 脚本仓库对吧？\n"
        "请帮我在当前工作区（或者你发现的 text2sql 目录下的某个测试文件如 testapi.py 中）"
        "寻找一个合适的地方，添加一个叫做 `def mcp_agent_test(): print('Hello from MCP Agent')` 的测试函数。\n"
        "根据你的规则，请先用搜索找出位置，然后必须调用 read_file 读取，再用 file_edit 去修改，最后用 run_bash 运行它验证。"
    )
    
    print("\n🤖 测试用例输入: ")
    print(test_case)
    print("\n" + "="*50)
    print("🧠 Agent 特工团队开始思考与作业...")
    print("="*50)
    
    try:
        initial_state = {"messages": [HumanMessage(content=test_case)], "iterations": 0}
        
        # 记录本轮对话落盘的数据结构 (存入专门的运行记录文件夹)
        log_filename = f"/Users/zrj/Documents/项目/coding-agent/runs_log/agent_run_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        conversation_log = {
            "timestamp": datetime.now().isoformat(),
            "user_prompt": test_case,
            "history": []
        }

        def persist_log():
            with open(log_filename, "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, ensure_ascii=False, indent=2)

        persist_log()

        # 使用流式输出监听整个 Graph 的节点状态推进
        for event in app.stream(initial_state, stream_mode="updates"):
            for node_name, node_state in event.items():
                if node_state is None: continue
                messages = node_state.get("messages", [])
                if messages:
                    # 获取每一个节点刚吐出的最新消息
                    last_msg = messages[-1]
                    print(f"\n⚡ [{node_name.upper()}] 进度流报告:")
                    
                    log_entry = {
                        "node": node_name.upper(),
                        "role": type(last_msg).__name__,
                        "content": ""
                    }

                    # 把模型试图调用工具的参数优雅打印出来
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        tool_info = f"🔧 准备调用工具: {[t['name'] for t in last_msg.tool_calls]}\n参数: " + str([t['args'] for t in last_msg.tool_calls])
                        print(f"  {tool_info}")
                        log_entry["content"] = tool_info
                        log_entry["tool_calls"] = last_msg.tool_calls
                    # 判断是否为 Tool 返回结果
                    elif type(last_msg).__name__ == "ToolMessage":
                        tool_res = f"🛠️ 工具执行完毕。返回了 {len(str(last_msg.content))} 字符。"
                        print(f"  {tool_res}")
                        log_entry["content"] = str(last_msg.content)
                    else:
                        # 正常文本打印大模型的话语
                        print(f"  {last_msg.content}")
                        log_entry["content"] = last_msg.content
                        
                    conversation_log["history"].append(log_entry)
                    persist_log()  # 实时将日志落盘！
                    
        print(f"\n💾 本次自动化测试协同对话已保存至: {log_filename}")

    except KeyboardInterrupt:
        print("\n🚨 用户强行中断了思考流。")

