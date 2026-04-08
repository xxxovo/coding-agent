import time
from pathlib import Path
from parser import RepoParser
from code_indexer import CodeIndexer

"""
模拟离线建库流程 (Offline Pipeline)
这个脚本只有在代码库发生变动、或者系统初次初始化时才需要运行。
"""

CURRENT_DIR = Path(__file__).resolve().parent
# 我们用你的测试库作为目标
REPO_PATH = CURRENT_DIR.parent / "test_repo" / "fastapi-realworld-example-app"
PERSIST_DIR = CURRENT_DIR / "data" / "demo_index"

print("====================================")
print("开始执行【离线建库】任务...")
print("====================================")

start_time = time.time()

# 1. 解析抽象语法树 (最耗时的 CPU 密集密操作之一)
print("[1/3] 正在扫全盘代码、解析 AST...")
parser = RepoParser(str(REPO_PATH))
units = parser.parse()

# 2. 调用大模型计算 Embedding 并建立 FAISS (重度耗时操作)
print(f"[2/3] 正在对 {len(units)} 个单位进行 Chunk 分块并计算向量（首次可能需要较久）...")
code_indexer = CodeIndexer(units, enable_chunking=True)
code_indexer.build()

# 3. 将结果固化到磁盘
print(f"[3/4] 正在将 FAISS 向量索引落盘到 {PERSIST_DIR} ...")
code_indexer.save(PERSIST_DIR)

# 4. 把代码数据同时灌入 Elasticsearch
print(f"[4/4] 正在将数据灌入 Elasticsearch (建立倒排索引和亚词分词)...")
from es_indexer import ESIndexer
es = ESIndexer()
es.build_index(units)

end_time = time.time()
print("====================================")
print(f"✅ 离线建库完成！总耗时: {end_time - start_time:.2f} 秒")
print("====================================")
