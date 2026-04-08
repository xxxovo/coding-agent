from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass(slots=True)
class GraphEdge:
    """图中的一条有向边。"""

    source: str
    target: str
    edge_type: str


class CodeGraphBuilder:
    """根据 code units 构建轻量代码图。

    当前版本构建的是 symbol-level graph，适合作为 Graph-RAG 的基础图。
    """

    def __init__(self, code_units: Iterable[dict]):
        self.code_units = list(code_units)
        self.symbol_by_id: Dict[str, dict] = {unit["id"]: unit for unit in self.code_units}
        self.symbols_by_name: Dict[str, List[dict]] = {}
        for unit in self.code_units:
            self.symbols_by_name.setdefault(unit["name"], []).append(unit)
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, List[GraphEdge]] = {}
        self.reverse_adjacency: Dict[str, List[GraphEdge]] = {}

    def build(self) -> dict:
        """构建完整代码图并返回可序列化结果。"""
        self._add_contains_edges()
        self._add_calls_edges()
        self._add_import_edges()
        self._add_inherits_edges()
        self._build_adjacency()
        return {
            "nodes": self.code_units,
            "edges": [asdict(edge) for edge in self.edges],
            "adjacency": {
                node_id: [asdict(edge) for edge in edges]
                for node_id, edges in self.adjacency.items()
            },
            "reverse_adjacency": {
                node_id: [asdict(edge) for edge in edges]
                for node_id, edges in self.reverse_adjacency.items()
            },
        }

    def expand_neighbors(self, node_ids: Iterable[str], max_hops: int = 1) -> List[dict]:
        """从命中节点出发做有限 hop 的图扩展。

        这是 Graph-RAG 的关键步骤之一：
        初始召回只负责找到入口节点，图扩展负责补齐局部上下文。
        """
        visited = set(node_ids)
        frontier = set(node_ids)
        
        # We also keep track of node distances to help with scoring later
        self.node_distances = {node_id: 0 for node_id in node_ids}

        for hop in range(1, max_hops + 1):
            next_frontier = set()
            for node_id in frontier:
                # Iterate over both direct and reverse edges
                all_edges = self.adjacency.get(node_id, []) + self.reverse_adjacency.get(node_id, [])
                for edge in all_edges:
                    if edge.target not in visited:
                        visited.add(edge.target)
                        next_frontier.add(edge.target)
                        self.node_distances[edge.target] = hop
            frontier = next_frontier
            if not frontier:
                break

        return [self.symbol_by_id[node_id] for node_id in visited if node_id in self.symbol_by_id]

    def _add_contains_edges(self) -> None:
        """建立父子作用域关系，例如 class -> method。"""
        qualified_index = {
            self._qualified_name(unit): unit["id"]
            for unit in self.code_units
        }
        for unit in self.code_units:
            parent = unit.get("parent")
            if not parent:
                continue
            parent_key = ".".join(part for part in [unit["module"], parent] if part)
            parent_id = qualified_index.get(parent_key)
            if parent_id:
                self.edges.append(GraphEdge(source=parent_id, target=unit["id"], edge_type="contains"))

    def _add_calls_edges(self) -> None:
        """根据精确的 LSP 跳转信息或近似匹配建立 symbol 间调用关系。

        当前版本支持解析 'JEDI:{rel_module}:{symbol_name}' 格式的高精度调用追踪，
        从而避免简单的 string matching 所造成的 graph explosion （图爆炸）和错误边覆盖。
        """
        # 预先构建按 (模块相对路径, 对象名称) 的精确索引
        exact_index = {}
        for unit in self.code_units:
            # CodeUnit 的 file_path 含有相对路径信息，例如 'app/main.py'
            # module 则是 'app.main'
            rel_file = unit.get("file_path", "")
            if rel_file.endswith(".py"):
                mod_path = rel_file[:-3].replace("/", ".")
            else:
                mod_path = unit.get("module", "")
                
            ident_key = f"{mod_path}:{unit['name']}"
            exact_index[ident_key] = unit["id"]
            
        # 常见弱区分度的泛型名称黑名单，在非 Jedi 模式下屏蔽，防范 O(N^2) 图爆炸
        GENERIC_NAMES = {"__init__", "run", "get", "post", "main", "config", "items", "append", "add"}

        for unit in self.code_units:
            for call_name in unit.get("calls") or []:
                # 1. 拦截 JEDI 精确追踪标记
                if call_name.startswith("JEDI:"):
                    # 格式: JEDI:fastapi-realworld-example-app/app/main.py:create_app
                    parts = call_name.split(":", 2)
                    if len(parts) == 3:
                        _, rel_path, symbol_name = parts
                        # 转换相对路径为 module path 模式
                        # e.g., 'app/main.py' -> 'app.main'
                        mod_base = rel_path.split("/")
                        if mod_base and mod_base[-1].endswith(".py"):
                            mod_base[-1] = mod_base[-1][:-3]
                        
                        target_mod_path = ".".join(mod_base)
                        target_key = f"{target_mod_path}:{symbol_name}"
                        
                        target_id = exact_index.get(target_key)
                        if target_id and target_id != unit["id"]:
                            self.edges.append(GraphEdge(source=unit["id"], target=target_id, edge_type="calls"))
                    continue

                # 2. 传统兜底匹配 (降级模式)
                short_name = call_name.split(".")[-1]
                if short_name in GENERIC_NAMES:
                    continue
                    
                for target in self.symbols_by_name.get(short_name, []):
                    if target["id"] != unit["id"]:
                        self.edges.append(GraphEdge(source=unit["id"], target=target["id"], edge_type="calls_fuzzy"))

    def _add_import_edges(self) -> None:
        """根据导入信息补充模块/符号依赖关系。"""
        modules = {unit["module"]: unit["id"] for unit in self.code_units if unit["type"] == "class"}
        for unit in self.code_units:
            for imported_symbol in unit.get("imports") or []:
                target_id = modules.get(imported_symbol)
                if target_id:
                    self.edges.append(GraphEdge(source=unit["id"], target=target_id, edge_type="imports"))

    def _add_inherits_edges(self) -> None:
        """根据类基类信息建立继承关系。"""
        for unit in self.code_units:
            for base_name in unit.get("bases") or []:
                for target in self.symbols_by_name.get(base_name.split(".")[-1], []):
                    self.edges.append(GraphEdge(source=unit["id"], target=target["id"], edge_type="inherits"))

    def _build_adjacency(self) -> None:
        """将边表转换成邻接表，便于后续图遍历。"""
        for edge in self.edges:
            self.adjacency.setdefault(edge.source, []).append(edge)
            
        for source, edges in self.adjacency.items():
            for edge in edges:
                self.reverse_adjacency.setdefault(edge.target, []).append(
                    GraphEdge(source=edge.target, target=source, edge_type=f"{edge.edge_type}_by")
                )

    def _qualified_name(self, unit: dict) -> str:
        """生成 symbol 的限定名，用于作用域内匹配。"""
        return ".".join(part for part in [unit["module"], unit.get("parent"), unit["name"]] if part)