"""Small, auditable tools exposed to the LangGraph research workflow."""

import ast
import operator
from typing import Callable, Dict, List


class AgentTools:
    def __init__(self, service):
        self.service = service
        self.registry: Dict[str, Callable] = {
            "local_search": self.local_search,
            "web_search": self.web_search,
            "library_stats": self.library_stats,
            "calculator": self.calculator,
        }

    def local_search(self, question: str) -> List[dict]:
        return self.service.search(question, limit=self.service.settings.context_top_k)

    def web_search(self, question: str) -> List[dict]:
        return self.service.web_search.search(question, limit=5, force=True)

    def library_stats(self) -> dict:
        return self.service.stats()

    @staticmethod
    def calculator(expression: str) -> str:
        """Evaluate basic arithmetic without eval or attribute access."""
        allowed = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def visit(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
                return allowed[type(node.op)](visit(node.operand))
            if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                return allowed[type(node.op)](visit(node.left), visit(node.right))
            raise ValueError("仅支持基础算术表达式")

        tree = ast.parse(expression, mode="eval")
        return str(visit(tree.body))
