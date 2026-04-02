from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(slots=True)
class CodeUnit:
    """结构化表示一个可索引的代码单元。

    这里的 code unit 是后续向量索引、图构建和检索的基本粒度。
    当前主要覆盖 function / async function / class。
    """

    id: str
    name: str
    type: str
    file: str
    code: str
    start_line: int
    end_line: int
    module: str
    signature: str
    docstring: Optional[str]
    parent: Optional[str] = None
    imports: Optional[list[str]] = None
    calls: Optional[list[str]] = None
    decorators: Optional[list[str]] = None
    bases: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return asdict(self)


class RepoParser:
    def __init__(self, repo_path: str, file_extensions: Optional[Iterable[str]] = None):
        self.repo_path = Path(repo_path).resolve()
        self.file_extensions = tuple(file_extensions or (".py",))
        # 引入大厂级 IDE 内核技术：启动 Jedi LSP Server 项目级分析基座
        try:
            import jedi
            self.jedi_project = jedi.Project(str(self.repo_path))
        except ImportError:
            self.jedi_project = None

    def parse(self) -> List[dict]:
        """扫描整个仓库并返回所有结构化 code units。"""
        code_units: List[dict] = []

        for file_path in self.scan_files():
            code_units.extend(self.parse_file(file_path))

        return code_units

    def scan_files(self) -> List[Path]:
        """递归扫描仓库中的目标文件。

        当前默认只扫描 `.py` 文件，并跳过 `__pycache__`。
        后续如果要支持多语言，可以在这里扩展文件后缀策略。
        """
        files: List[Path] = []

        for extension in self.file_extensions:
            files.extend(self.repo_path.rglob(f"*{extension}"))

        return sorted(
            file_path
            for file_path in files
            if file_path.is_file() and "__pycache__" not in file_path.parts
        )

    def parse_file(self, file_path: str | Path) -> List[dict]:
        """解析单个 Python 文件并提取 code units。"""
        path = Path(file_path)

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            print(f"Error parsing {path}: {error}")
            return []
            
        jedi_script = None
        if getattr(self, 'jedi_project', None):
            import jedi
            # 为每个文件激活一个带有全项目上下文感知的 Script
            jedi_script = jedi.Script(source, path=str(path), project=self.jedi_project)

        source_lines = source.splitlines()
        visitor = _CodeUnitVisitor(
            source_lines=source_lines,
            repo_root=self.repo_path,
            file_path=path,
            jedi_script=jedi_script, # 赋能 AST 访问器
        )
        visitor.visit(tree)
        return [unit.to_dict() for unit in visitor.units]


class _CodeUnitVisitor(ast.NodeVisitor):
    """基于 AST 的访问器，结合 Jedi LSP引擎提取高维语义信息。"""

    def __init__(self, source_lines: List[str], repo_root: Path, file_path: Path, jedi_script=None):
        self.source_lines = source_lines
        self.repo_root = repo_root
        self.file_path = file_path
        self.module_name = self._build_module_name(file_path)
        self.units: List[CodeUnit] = []
        self.parent_stack: List[str] = []
        self.file_imports: set[str] = set()
        self.jedi_script = jedi_script

    def visit_Import(self, node: ast.Import) -> None:
        """收集文件级 import 信息，供每个 symbol 复用。"""
        for alias in node.names:
            self.file_imports.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """收集 `from x import y` 形式的导入。"""
        module = node.module or ""
        for alias in node.names:
            imported_name = f"{module}.{alias.name}".strip(".")
            self.file_imports.add(imported_name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """提取普通函数，并继续递归处理内部嵌套定义。"""
        self._add_code_unit(node=node, unit_type="function")
        self.parent_stack.append(node.name)
        self.generic_visit(node)
        self.parent_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """提取异步函数，统一视为 function 类型。"""
        self._add_code_unit(node=node, unit_type="function")
        self.parent_stack.append(node.name)
        self.generic_visit(node)
        self.parent_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """提取类定义，并继续处理类中的方法。"""
        self._add_code_unit(node=node, unit_type="class")
        self.parent_stack.append(node.name)
        self.generic_visit(node)
        self.parent_stack.pop()

    def _add_code_unit(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        unit_type: str,
    ) -> None:
        """把 AST 节点转换为结构化 CodeUnit。

        这里会同时提取：
        - 基础定位信息：文件、行号、代码片段
        - 语义描述信息：signature、docstring
        - 结构信息：parent、imports、calls、decorators、bases
        """
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)

        if start_line is None or end_line is None:
            return

        code = "\n".join(self.source_lines[start_line - 1 : end_line])
        relative_file = str(self.file_path.relative_to(self.repo_root))
        parent = ".".join(self.parent_stack) if self.parent_stack else None
        qualified_name = ".".join(part for part in [self.module_name, parent, node.name] if part)

        self.units.append(
            CodeUnit(
                id=f"{relative_file}:{node.name}:{start_line}",
                name=node.name,
                type=unit_type,
                file=relative_file,
                code=code,
                start_line=start_line,
                end_line=end_line,
                module=self.module_name,
                signature=self._build_signature(node),
                docstring=ast.get_docstring(node),
                parent=parent,
                imports=sorted(self.file_imports),
                calls=self._collect_calls(node),
                decorators=self._collect_decorators(node),
                bases=self._collect_bases(node) if isinstance(node, ast.ClassDef) else [],
            )
        )

    def _build_module_name(self, file_path: Path) -> str:
        """将文件路径转成 Python 模块路径。"""
        relative_path = file_path.relative_to(self.repo_root)
        module_parts = list(relative_path.with_suffix("").parts)
        if module_parts and module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        return ".".join(module_parts)

    def _build_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
        """构造更适合检索的简化签名文本。"""
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(self._collect_bases(node))
            return f"class {node.name}({bases})" if bases else f"class {node.name}"

        args = [argument.arg for argument in node.args.args]
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        args.extend(argument.arg for argument in node.args.kwonlyargs)
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)})"

    def _collect_calls(self, node: ast.AST) -> list[str]:
        """探针级提取代码调用特征，结合 Jedi LSP 执行多文件跳跃追踪。"""
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                resolved_by_jedi = False
                
                # 1. 优先采用 IDE 级语法树类型推演
                if self.jedi_script and hasattr(child.func, 'end_lineno') and hasattr(child.func, 'end_col_offset'):
                    try:
                        line = child.func.end_lineno
                        col = max(0, child.func.end_col_offset - 1)
                        
                        defs = self.jedi_script.infer(line, col)
                        for d in defs:
                            if d.module_path and d.name:
                                try:
                                    rel_path = Path(d.module_path).relative_to(self.repo_root)
                                    # 提取出极度精确的目标坐标签名，打入 JEDI 防伪印记
                                    names.add(f"JEDI:{rel_path}:{d.name}")
                                    resolved_by_jedi = True
                                except ValueError:
                                    # 跑出项目沙盒的外调（比如引用官方的 os 或者 requests），忽略
                                    pass
                    except Exception:
                        pass
                        
                # 2. 如果 Jedi 处理部分动态代码失败，优雅降维回到我们的启发式字符推理
                if not resolved_by_jedi:
                    call_name = self._resolve_name(child.func)
                    if call_name:
                        names.add(call_name)
                        
        return sorted(list(names))

    def _collect_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
        """收集装饰器名称，后续可用于路由/框架语义增强。"""
        decorators: list[str] = []
        for decorator in node.decorator_list:
            decorator_name = self._resolve_name(decorator)
            if decorator_name:
                decorators.append(decorator_name)
        return decorators

    def _collect_bases(self, node: ast.ClassDef) -> list[str]:
        """收集类继承的父类名称。"""
        bases: list[str] = []
        for base in node.bases:
            base_name = self._resolve_name(base)
            if base_name:
                bases.append(base_name)
        return bases

    def _resolve_name(self, node: ast.AST) -> Optional[str]:
        """把 AST 表达式尽量还原成可读名称。

        例如：
        - `foo` -> `foo`
        - `repo.get_user` -> `repo.get_user`
        - `super().__call__` -> `super.__call__`
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            value_name = self._resolve_name(node.value)
            return f"{value_name}.{node.attr}" if value_name else node.attr
        if isinstance(node, ast.Call):
            return self._resolve_name(node.func)
        if isinstance(node, ast.Subscript):
            return self._resolve_name(node.value)
        return None