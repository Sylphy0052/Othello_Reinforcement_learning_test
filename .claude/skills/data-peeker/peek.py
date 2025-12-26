import argparse
import csv
import json
import os
import pickle


def peek_file(filepath, lines=5):
    ext = os.path.splitext(filepath)[1].lower()
    info = {"file": filepath, "size_bytes": os.path.getsize(filepath), "type": ext}

    try:
        if ext == ".json":
            with open(filepath, "r") as f:
                # 巨大JSON対策: 先頭だけ読んで構造推測は難しいので、一度ロードしてキーのみ表示などの工夫
                # ここでは簡易的にロードし、トップレベルのキーまたは先頭要素のみ返す
                data = json.load(f)
                if isinstance(data, list):
                    info["structure"] = "List"
                    info["length"] = len(data)
                    info["preview"] = data[:lines]
                elif isinstance(data, dict):
                    info["structure"] = "Dict"
                    info["keys"] = list(data.keys())
                    info["preview"] = {
                        k: str(v)[:100] for k, v in list(data.items())[:lines]
                    }

        elif ext == ".csv":
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = [row for _, row in zip(range(lines), reader)]
                info["header"] = header
                info["preview"] = rows

        elif ext == ".pkl":
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                info["type_loaded"] = str(type(data))
                if hasattr(data, "__len__"):
                    info["length"] = len(data)
                info["preview"] = str(data)[:500] + "..."  # 文字列化して先頭のみ

        else:  # Text fallback
            with open(filepath, "r") as f:
                info["preview"] = [next(f).strip() for _ in range(lines)]

    except Exception as e:
        info["error"] = str(e)

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--lines", type=int, default=5)
    args = parser.parse_args()

    print(json.dumps(peek_file(args.file, args.lines), indent=2, default=str))


if __name__ == "__main__":
    main()
