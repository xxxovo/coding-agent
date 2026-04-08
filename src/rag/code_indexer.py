from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "huggingface"


@dataclass(slots=True)
class VectorSearchResult:
    unit_id: str
    score: float


class CodeIndexer:
    """负责把结构化 code units 编码成向量并建立 FAISS 索引。"""

    def __init__(
        self,
        code_units: Iterable[dict] | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        enable_chunking: bool = True  # 新增控制
    ):
        self.code_units = list(code_units) if code_units else []
        self.embedding_model_name = embedding_model_name
        self.cache_dir = Path(cache_dir)
        self.enable_chunking = enable_chunking
        self.chunk_mapping = []  # 保存每个 chunk 到原 unit 的映射

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 用绝对路径加载本地 SentenceTransformer 模型，并且强制断网
        local_model_path = '/Users/zrj/Documents/项目/coding-agent/.cache/huggingface/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
        
        self.embedding_model = SentenceTransformer(
            local_model_path,
            local_files_only=True
        )

        if self.enable_chunking:
            from chunker import TreeSitterChunker
            chunker = TreeSitterChunker()
            self.documents = []
            for unit in self.code_units:
                chunks = chunker.chunk_code_unit(unit)
                for chunk in chunks:
                    self.documents.append(chunk["document"])
                    self.chunk_mapping.append(chunk["unit_id"])
        else:
            self.documents = [self._build_document(unit) for unit in self.code_units]
            self.chunk_mapping = [unit["id"] for unit in self.code_units]

        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None

    def build(self) -> None:
        """生成所有 code unit 的向量并建立 FAISS 索引。"""
        embeddings = self.embedding_model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = embeddings.astype("float32")
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.embeddings)

    def save(self, persist_dir: str | Path) -> None:
        """离线建库持久化：将生成的索引与元数据落盘保存。"""
        import json
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存 FAISS 索引数据
        if self.index is not None:
            faiss.write_index(self.index, str(persist_dir / "vector.index"))
            
        # 2. 保存原始代码元信息 (Code Units) 以及 Chunk Map
        with open(persist_dir / "code_units.json", "w", encoding="utf-8") as f:
            json.dump(self.code_units, f, ensure_ascii=False, indent=2)
            
        with open(persist_dir / "chunk_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_mapping, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls, 
        persist_dir: str | Path, 
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR
    ) -> CodeIndexer:
        """在线查询加载：从磁盘反序列化读取数据结构。"""
        import json
        persist_dir = Path(persist_dir)
        
        # 1. 读取元数据重建实例
        with open(persist_dir / "code_units.json", "r", encoding="utf-8") as f:
            code_units = json.load(f)
            
        instance = cls(
            code_units=code_units, 
            embedding_model_name=embedding_model_name, 
            cache_dir=cache_dir,
            enable_chunking=False # 加载时无需再次 Chunking
        )
        
        with open(persist_dir / "chunk_mapping.json", "r", encoding="utf-8") as f:
            instance.chunk_mapping = json.load(f)
        
        # 2. 加载 FAISS 二进制索引图
        index_path = persist_dir / "vector.index"
        if index_path.exists():
            instance.index = faiss.read_index(str(index_path))
            
        return instance

    def search(self, query: str, top_k: int = 5) -> List[VectorSearchResult]:
        """使用 query embedding 在 FAISS 中召回最相似的 code units。"""
        if self.index is None:
            self.build()

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k * 3) # 查询更多以便聚合
        
        unit_scores: dict[str, float] = {}
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            unit_id = self.chunk_mapping[index]
            # Max-Pooling: 同一个 unit 如果命中多个 chunk，取最高分
            if unit_id not in unit_scores or score > unit_scores[unit_id]:
                unit_scores[unit_id] = float(score)

        results: List[VectorSearchResult] = [
            VectorSearchResult(unit_id=uid, score=s) 
            for uid, s in sorted(unit_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]
        return results

    def _build_document(self, unit: dict) -> str:
        """把 symbol 元数据拼成适合 embedding 的检索文档。"""
        fields = [
            unit.get("name", ""),
            unit.get("type", ""),
            unit.get("module", ""),
            unit.get("file", ""),
            unit.get("signature", ""),
            unit.get("docstring", "") or "",
            " ".join(unit.get("imports") or []),
            " ".join(unit.get("calls") or []),
            unit.get("code", ""),
        ]
        return "\n".join(field for field in fields if field)
