from __future__ import annotations

from typing import Iterable, List
from elasticsearch import Elasticsearch, helpers

class ESIndexer:
    """使用 Elasticsearch 替代内存中的 BM25Okapi。
    
    主要特性：
    1. 可持久化、分布式的倒排索引。
    2. 针对代码设计的亚词分词器：支持 camelCase 和 snake_case。
    """

    def __init__(self, es_url: str = "http://localhost:9200", index_name: str = "code_search"):
        # 禁用默认的 sniff 及一些复杂的连接维持机制，对单机版容错更好
        self.es = Elasticsearch(es_url, request_timeout=30, max_retries=3, retry_on_timeout=True)
        self.index_name = index_name

    def setup_index(self) -> None:
        """配置索引，加入支持驼峰和下划线的分词器。"""
        try:
            if self.es.indices.exists(index=self.index_name):
                return
        except Exception as e:
            # handle potential routing or empty cluster status
            if getattr(e, 'status_code', 0) == 404:
                pass
            elif getattr(e, 'meta', None) and getattr(e.meta, 'status', 0) == 404:
                pass
            else:
                try: 
                    # fallback check
                    self.es.indices.get(index=self.index_name)
                    return
                except:
                    pass

        settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "code_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["code_word_delimiter", "lowercase", "stop"]
                        }
                    },
                    "filter": {
                        "code_word_delimiter": {
                            "type": "word_delimiter_graph",
                            "split_on_case_change": True,  # 匹配 camelCase -> camel, Case
                            "generate_word_parts": True,   # 匹配 snake_case -> snake, case
                            "generate_number_parts": True,
                            "preserve_original": True      # 保留原词 UserAccount
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "name": {"type": "text", "analyzer": "code_analyzer"},
                    "content": {"type": "text", "analyzer": "code_analyzer"}
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=settings)

    def build_index(self, code_units: Iterable[dict]) -> None:
        """将从 Parser 提取的 Code Units 离线写入 ES。"""
        self.setup_index()
        actions = []
        for unit in code_units:
            fields = [
                unit.get("name", ""),
                unit.get("type", ""),
                unit.get("signature", ""),
                unit.get("docstring", "") or "",
                unit.get("code", ""),
            ]
            content = "\n".join(f for f in fields if f)
            
            actions.append({
                "_index": self.index_name,
                "_id": unit["id"],
                "_source": {
                    "id": unit["id"],
                    "name": unit.get("name", ""),
                    "content": content
                }
            })
        
        helpers.bulk(self.es, actions)
        self.es.indices.refresh(index=self.index_name)

    def search(self, query: str, top_k: int = 5) -> dict[str, float]:
        """按 BM25 打分召回。"""
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["name^2", "content"], # name 权重 * 2
                    "analyzer": "code_analyzer"
                }
            },
            "size": top_k,
            "_source": False
        }
        resp = self.es.search(index=self.index_name, body=body)
        
        scores: dict[str, float] = {}
        for hit in resp["hits"]["hits"]:
            scores[hit["_id"]] = hit["_score"]
        return scores
