import sys
import time
from pathlib import Path

# 我们强制屏蔽标准输出里的无用信息以显得清爽
import warnings
warnings.filterwarnings('ignore')

from code_indexer import CodeIndexer
from graph_builder import CodeGraphBuilder
from retriever import HybridRetriever

"""
模拟在线检索服务 (Online Serving)
这个脚本展示了我们在生产环境中启动服务的极速体感。
不需要扫文件、不需要调 Embedding大模型编码原始代码！
"""

CURRENT_DIR = Path(__file__).resolve().parent
PERSIST_DIR = CURRENT_DIR / "data" / "demo_index"

if not PERSIST_DIR.exists():
    print("❌ 错误：找不到索引数据，请先运行 python build_index.py")
    sys.exit(1)

print("====================================")
print("🚀 启动【在线检索服务】...")
print("====================================")

start_load = time.time()

# 1. 毫秒级直接从硬盘反弹起大模型计算了很久的结果
print("⚡ 正在从磁盘加载 FAISS 向量索引与代码元数据...")
code_indexer = CodeIndexer.load(PERSIST_DIR)

# 为了演示我们用持久化的元数据极速构建出关系图
graph_builder = CodeGraphBuilder(code_indexer.code_units)
graph_builder.build()

retriever = HybridRetriever(code_indexer.code_units, graph_builder, code_indexer)

end_load = time.time()
print(f"✅ 服务启动完成！冷启动仅耗时: {end_load - start_load:.4f} 秒\n")

# 现在模拟用户的搜索行为
while True:
    try:
        query = input("\n🔍 请输入检索词 (输入 q 退出): ")
        if query.strip().lower() == 'q':
            break
        if not query.strip():
            continue
            
        print("-" * 50)
        results = retriever.search(query, top_k=3, expand_hops=1)
        
        for i, res in enumerate(results):
            print(f"[{i+1}] 命中: {res['name']} | 类型: {res['type']} | 文件: {res['file']}")
            print(f"    ⭐ 得分: {res['retrieval_score']} | 归因: {res['retrieval_reason']}")
        print("-" * 50)
        
    except KeyboardInterrupt:
        break
