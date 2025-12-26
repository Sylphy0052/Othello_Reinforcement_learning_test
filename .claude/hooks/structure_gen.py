import os
import sys


def generate_tree(startpath):
    tree_str = "Project Structure:\n"
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        tree_str += "{}{}/\n".format(indent, os.path.basename(root))
        subindent = " " * 4 * (level + 1)
        for f in files:
            if f.endswith(".py") or f.endswith(".md"):  # 重要なファイルのみ
                tree_str += "{}{}\n".format(subindent, f)
    return tree_str


def main():
    # docsフォルダがない場合は作る
    os.makedirs("docs", exist_ok=True)

    with open("docs/structure.txt", "w", encoding="utf-8") as f:
        f.write(generate_tree("."))

    print("🏗️  Updated docs/structure.txt with latest file tree.")
    sys.exit(0)


if __name__ == "__main__":
    main()
