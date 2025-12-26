import argparse
import ast
import json


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        # 簡易的な複雑度計算: 分岐の数 + 1
        complexity = 1
        for child in ast.walk(node):
            if isinstance(
                child, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler)
            ):
                complexity += 1

        self.functions.append(
            {
                "name": node.name,
                "lineno": node.lineno,
                "complexity": complexity,
                "args": [a.arg for a in node.args.args],
            }
        )
        self.generic_visit(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        visitor = ComplexityVisitor()
        visitor.visit(tree)

        result = {
            "status": "success",
            "file": args.file,
            "functions": sorted(
                visitor.functions, key=lambda x: x["complexity"], reverse=True
            ),
        }

        # 評価コメントの付与
        for func in result["functions"]:
            score = func["complexity"]
            if score > 20:
                func["risk"] = "CRITICAL (Needs Refactor)"
            elif score > 10:
                func["risk"] = "WARNING"
            else:
                func["risk"] = "SAFE"

    except Exception as e:
        result = {"status": "error", "message": str(e)}

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
