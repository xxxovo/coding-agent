from __future__ import annotations

import tree_sitter
try:
    from tree_sitter_python import language as ts_python
except ImportError:
    ts_python = None

from typing import List, Dict

class TreeSitterChunker:
    """基于 Tree-Sitter 语法树做代码语义切分（Chunking），防止长代码块截断。"""

    def __init__(self, max_chunk_tokens: int = 400):
        # 400个单词约等于512 BPE tokens。防止越出。
        self.max_tokens = max_chunk_tokens
        
        if ts_python is None:
            raise ImportError("Please install tree-sitter-python")
        
        lang = tree_sitter.Language(ts_python())
        try:
            self.parser = tree_sitter.Parser()
            self.parser.set_language(lang)
        except AttributeError:
            # 兼容 tree-sitter 0.22+ 最新版本 API
            self.parser = tree_sitter.Parser(lang)

    def chunk_code_unit(self, unit: dict) -> List[Dict]:
        """将完整的 CodeUnit 长代码块打碎，保留上下文。"""
        code = unit.get("code", "")
        if not code:
            return []

        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        chunks = []
        current_chunk = []
        current_length = 0

        # 获取父级函数/类的签名头作为每个 Chunk 的上下文前缀
        signature = unit.get("signature", "")
        # 加入 imports 和 Docstring 作为所有 chunk 共享的元数据
        metadata_header = "\n".join(filter(None, [
            unit.get("docstring"),
            " ".join(unit.get("imports") or []),
            signature
        ]))
        
        # 将头部长度纳入估算 (一个词粗略当做 token)
        meta_base_len = len(metadata_header.split())
        
        for statement in root_node.children:
            text = statement.text.decode("utf8")
            stmt_len = len(text.split())

            # 超过阈值就换块
            if current_length + stmt_len + meta_base_len > self.max_tokens and current_chunk:
                chunk_code = "\n".join(current_chunk)
                chunks.append(self._build_chunk(unit, chunk_code, metadata_header, len(chunks)))
                current_chunk = []
                current_length = 0

            current_chunk.append(text)
            current_length += stmt_len

        if current_chunk:
            chunk_code = "\n".join(current_chunk)
            chunks.append(self._build_chunk(unit, chunk_code, metadata_header, len(chunks)))

        return chunks

    def _build_chunk(self, unit: dict, chunk_code: str, meta: str, index: int) -> dict:
        """生成独立于全文本的向量化文档。"""
        return {
            "chunk_id": f"{unit['id']}#chunk{index}",
            "unit_id": unit["id"],
            "document": f"{meta}\n\n{chunk_code}"
        }
