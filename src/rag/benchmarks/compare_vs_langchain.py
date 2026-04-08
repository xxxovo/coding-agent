import json
import time
from pathlib import Path
from typing import List, Dict

# 加入环境路径
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# LangChain 相关依赖 (用于 Baseline)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 你的项目依赖 (用于 Hybrid RAG)
from retriever import HybridRetriever
from code_indexer import CodeIndexer
from graph_builder import CodeGraphBuilder

# ================= 1. 配置路径 =================
CURRENT_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = CURRENT_DIR / "benchmark.jsonl"
INDEX_DIR = CURRENT_DIR.parent / "data" / "demo_index"
LOCAL_EMBEDDING_MODEL = '/Users/zrj/Documents/项目/coding-agent/.cache/huggingface/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'

# ================= 2. 准备评测数据集 =================
def load_benchmarks() -> List[Dict]:
    benchmarks = []
    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                b = json.loads(line)
                if b.get("repo") == "fastapi-realworld-example-app":
                    benchmarks.append(b)
    return benchmarks

# ================= 3. 构建 LangChain Baseline =================
def build_langchain_baseline(code_units: List[Dict]):
    print("⏳ [Baseline] 正在构建普通 LangChain 文本切片索引...")
    # 提取完整的文件/函数级原始文本，模拟无脑切分
    raw_docs = []
    for unit in code_units:
        content = f"File: {unit.get('file', '')}\n{unit.get('code', '')}"
        raw_docs.append(Document(page_content=content, metadata={"file": unit.get("file", ""), "id": unit.get("id")}))

    # 模拟传统：固定长度 1000 字符切割，重叠 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_docs)
    print(f"   -> 生成了 {len(chunks)} 个基于字符长度的文本切片")

    embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ [Baseline] LangChain 索引构建完毕！\n")
    return vectorstore

# ================= 4. 初始化你的混合检索架构 =================
def build_hybrid_system():
    print("⏳ [Hybrid] 正在加载你的高级混合检索(AST+双塔+重排)架构...")
    indexer = CodeIndexer.load(INDEX_DIR, embedding_model_name=LOCAL_EMBEDDING_MODEL)
    graph_builder = CodeGraphBuilder(indexer.code_units) 
    # 解析 graph，加载边等，可根据实际图构建器的实现修改
    graph_builder.build() 
    
    retriever = HybridRetriever(indexer.code_units, graph_builder, indexer)
    # 取消掉因为尚未配置 ES 而引发的报错，安全回退到 BM25
    retriever.use_es = False
    # 初始化 BM25 模型（模拟 Fallback）
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [retriever._tokenize(doc) for doc in retriever.documents]
    retriever.bm25_model = BM25Okapi(tokenized_corpus)
    print("✅ [Hybrid] 架构装载完毕！\n")
    return retriever, indexer.code_units
    
# ================= 5. 开始打擂台 (评测函数) =================
def evaluate_hit_rate(query: str, expected_files: List[str], baseline_store, hybrid_retriever) -> Dict:
    # 评测标准：检索出来的前 3 个结果中，只要有一个对应文件在我们预期的 primary_files 列表里，就算命中
    
    # 1. 测试 Baseline
    base_docs = baseline_store.similarity_search(query, k=5)
    # 提取 Top-3 里的独立文件路径
    base_top3_files = []
    for doc in base_docs:
        file_path = doc.metadata.get("file")
        if file_path not in base_top3_files:
            base_top3_files.append(file_path)
            if len(base_top3_files) == 3: break
            
    base_hit = any(f in expected_files for f in base_top3_files)

    # 2. 测试 Hybrid
    hybrid_results = hybrid_retriever.search(query, top_k=3, expand_hops=0)
    hybrid_top3_files = []
    for res in hybrid_results:
        f = res.get("file")
        if f not in hybrid_top3_files:
            hybrid_top3_files.append(f)
            
    hybrid_hit = any(f in expected_files for f in hybrid_top3_files)
    
    return {"base_hit": base_hit, "hybrid_hit": hybrid_hit}


if __name__ == "__main__":
    benchmarks = load_benchmarks()
    # 为保证对比环境一致，共享你 index_dir 里保存的原文件
    hybrid_retriever, code_units = build_hybrid_system()
    baseline_store = build_langchain_baseline(code_units)
    
    print("\n" + "="*50)
    print("🚀 开始 1v1 Top-3 准确率对比测试")
    print("="*50)
    
    base_success = 0
    hybrid_success = 0
    total = len(benchmarks)
    
    for i, b in enumerate(benchmarks):
        q = b["query"]
        expected = b.get("primary_files", []) 
        res = evaluate_hit_rate(q, expected, baseline_store, hybrid_retriever)
        
        if res["base_hit"]: base_success += 1
        if res["hybrid_hit"]: hybrid_success += 1
            
        print(f"[{i+1}/{total}] Query: '{q}'")
        print(f"   -> Baseline (LangChain): {'✅ 命中' if res['base_hit'] else '❌ 丢失'}")
        print(f"   -> Hybrid   (Your RAG): {'✅ 命中' if res['hybrid_hit'] else '❌ 丢失'}")
        print("-" * 30)
        
    base_rate = (base_success / total) * 100
    hybrid_rate = (hybrid_success / total) * 100
    uplift = hybrid_rate - base_rate
    
    print("\n" + "🔥 最终成绩出炉 🔥".center(48))
    print(f"📚 测试查询总数: {total}")
    print(f"🐢 LangChain 暴力分块 RAG Top-3 召回率: {base_rate:.1f}%")
    print(f"🐉 你的 AST 语义混合 RAG Top-3 召回率: {hybrid_rate:.1f}%")
    
    improvement_text = f"🚀🚀 性能较基础路线绝对值提升了 {uplift:.1f}% 🚀🚀"
    relative_uplift = (uplift / base_rate * 100) if base_rate > 0 else float('inf')
    
    print("="*50)
    print(improvement_text)
    print(f"💡 相对提升率达到了 {relative_uplift:.1f}%！可以将该数据直接写进简历中。")
    print("="*50)
