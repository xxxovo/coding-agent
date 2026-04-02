import os
import sys
from typing import TypedDict, Annotated, Sequence, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

# 导入在 tools.py 中写好的本地执行器
from tools import retrieve_code, expand_code_graph, write_file, apply_patch, run_code, list_dir

# 初始化 LLM 引擎
import os
os.environ["OPENAI_API_KEY"] = "sk-qdglzganpplxlmretijydhxxcvygqflqiclqnxhzvhlphoxr"
# 如果使用的是硅基流动中转（因为你需要调 Qwen和DeepSeek模型），必须填写对应的网关地址
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"

# Planner：负责拆解任务，可以稍具发散性和规划感
planner_llm = ChatOpenAI(model="Qwen/Qwen3-8B", temperature=0.7)

# Base Coder：作为纯粹的编码和工具执行机器，要求输出极度稳定，降低温度
base_coder_llm = ChatOpenAI(model="Qwen/Qwen3-8B", temperature=0.1)

# Verifier：作为严格的代码 Review 专员，使用专有大模型
verifier_llm = ChatOpenAI(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", temperature=0.1)

# ================= 数据形态 (Graph State) =================
class AgentState(TypedDict):
    """保存 LangGraph 在节点中流转的关键状态上下文"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: str
    verification_result: str
    iterations: int  # 防死循环计数器

# ================= 提示词矩阵 (Prompts) =================
PLANNER_PROMPT = """You are a planning agent.

Break the user request into executable coding steps.

Focus on:
- code search (retrieve_code / expand_code_graph)
- code modification (write_file / apply_patch)
- debugging
- verification (run_code)

Return a structured step-by-step plan."""

CODER_PROMPT = """You are a coding agent.

You have access to tools. 
When searching code:
1. Try `retrieve_code` first.
2. Only use `expand_code_graph` ONE TIME if the original results lack necessary context. DO NOT endlessly call `expand_code_graph` with the same parameters!
3. If search fails or repeats uselessly, stop searching and try to fulfill the task with existing context.

Follow the plan step by step.

Current Executable Plan:
{plan}
"""

VERIFIER_PROMPT = """You are a strict code reviewer.

Check the recent interaction history:
1. Does the code satisfy the original user request?
2. Are there logical errors?
3. Are there potential bugs?
4. Does execution output match expectation?

Return:
PASS or FAIL

If FAIL:
- explain reason
- suggest fix direction"""


# ================= 节点逻辑 (Nodes) =================

def planner_node(state: AgentState):
    """负责将用户原始的需求转换为结构化、严谨的步骤蓝图"""
    messages = state.get("messages", [])
    user_input = messages[0].content
    
    response = planner_llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=user_input)
    ])
    
    print(f"\n[Planner] 制定计划成功。")
    return {
        "plan": response.content, 
        "iterations": 0,
        # 把思考过程放进队列流，让后续节点可见
        "messages": [AIMessage(content=f"📝 **Plan Created:**\n{response.content}")]
    }

# 绑定工具到语言模型作为 Function Calling 支持
tools = [retrieve_code, expand_code_graph, write_file, apply_patch, run_code, list_dir]
coder_llm = base_coder_llm.bind_tools(tools)

def coder_node(state: AgentState):
    """Coding Agent：读蓝本，疯狂跑工具写代码"""
    print(f"\n[Coder] 正在执行操作...")
    plan = state.get("plan", "No plan created.")
    messages = state.get("messages", [])
    
    # 强制压入 Planner 给出的全局指挥上下文
    sys_msg = SystemMessage(content=CODER_PROMPT.format(plan=plan))
    response = coder_llm.invoke([sys_msg] + messages)
    
    return {
        "messages": [response], 
        "iterations": state.get("iterations", 0) + 1
    }

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
         print(f"🎉 [Verifier] 审核通过，任务结束！")
         
    return {
        "verification_result": result_text,
        "messages": feedback_msgs
    }

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

def router_from_verifier_to_end(state: AgentState) -> Literal["coder", "__end__"]:
    """校验反馈的路由：只要没过，打回 coder_node 重做"""
    result = state.get("verification_result", "")
    iters = state.get("iterations", 0)
    
    # 强行保护：超过 8 次拉扯，立刻强行中止（防止 API 烧钱）
    if "PASS" in result.upper() or iters >= 8:
        return "__end__"
    
    return "coder"


# ================= 引擎绑定装配 =================

workflow = StateGraph(AgentState)

# 1. 插桩：放置节点模块
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("tools", ToolNode(tools=tools))
workflow.add_node("verifier", verifier_node)

# 2. 连线：梳理业务逻辑结构流向
workflow.add_edge(START, "planner")    # 启动进入 Planner 拆解
workflow.add_edge("planner", "coder")  # 计划出炉，转交 Coder

# Coder 执行后的路口：可能跑工具，也可能去验收
workflow.add_conditional_edges(
    "coder", 
    router_should_continue_from_coder
)

workflow.add_edge("tools", "coder")    # 跑完工具拿到 stdout 回到大脑进一步思索

# 验收后强逻辑：能走 __end__ 退场，不然继续滚回开发阶段重干
workflow.add_conditional_edges(
    "verifier",
    router_from_verifier_to_end
)

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
    
    # 增加交互式命令行循环，让用户能输入指令
    while True:
        try:
            user_input = input("\n🤖 请输入指令 (输入 q 退出): ")
            if user_input.strip().lower() in ['q', 'quit', 'exit']:
                break
            if not user_input.strip():
                continue

            print("\n" + "="*50)
            print("🧠 Agent 特工团队开始思考与作业...")
            print("="*50)
            
            initial_state = {"messages": [HumanMessage(content=user_input)], "iterations": 0}
            
            # 记录本轮对话落盘的数据结构
            log_filename = f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            conversation_log = {
                "timestamp": datetime.now().isoformat(),
                "user_prompt": user_input,
                "history": []
            }

            def persist_log():
                # 辅助函数：每次写入都覆盖保存
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(conversation_log, f, ensure_ascii=False, indent=2)

            # 先创建文件结构
            persist_log()

            # 使用流式输出监听整个 Graph 的节点状态推进
            for event in app.stream(initial_state, stream_mode="updates"):
                for node_name, node_state in event.items():
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
                            tool_info = f"🔧 准备调用工具: {[t['name'] for t in last_msg.tool_calls]}\n参数: {[t['args'] for t in last_msg.tool_calls]}"
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
                        
            print(f"\n💾 本次群智协同多轮对话已保存至: {log_filename}")

        except KeyboardInterrupt:
            print("\n🚨 用户强行中断了思考流。")
            break