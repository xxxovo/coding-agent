import pytest

# 假设这是我们 RAG 系统里的一个核心工具函数，它负责判断是否需要重新获取上下文 
# (你把它想象成是被大模型改坏的代码，或者你本来写好的健康代码)
def agent_should_expand_graph(score: float, current_hops: int) -> bool:
    """如果召回的分数低于 0.5 且扩展跳数小于 3，则继续图网拓展寻找上下文"""
    if score < 0.5 and current_hops < 3:
        return True
    return False

# ================= 下面这一部分就是 PyTest 单元测试 =================
# PyTest 的精髓在于以 test_ 开头的函数命名
# 当我们运行 pytest 命令时，它会自动来这里执行这三道题：

def test_should_expand_when_score_is_low():
    # 场景1: 分数极低，且刚开始拓展 (hops=1)。我们期望返回 True。
    result = agent_should_expand_graph(score=0.2, current_hops=1)
    # assert（断言）是灵魂：如果你写错了底层代码导致返回了 False，这行就会立刻"爆红"报错
    assert result == True

def test_should_stop_expanding_when_score_is_high():
    # 场景2: 已经被精确召回了(score=0.9)，不需要再扩大干扰图节点了。期望返回 False。
    result = agent_should_expand_graph(score=0.9, current_hops=1)
    assert result == False

def test_should_stop_expanding_when_hops_reach_limit():
    # 场景3: 分数很低，但是跳数已经用光了(hops=3)，为了防止死循环必须停下！期望返回 False。
    result = agent_should_expand_graph(score=0.1, current_hops=3)
    assert result == False

def test_mcp_agent():
    print('Hello from MCP Agent')
