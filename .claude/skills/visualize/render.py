import argparse
import json


def render_board(data_str):
    try:
        # 文字列として渡されたリストをパース
        # 例: "[0,0,1,-1...]" -> list
        data = json.loads(data_str)

        if len(data) == 64:
            board = [data[i : i + 8] for i in range(0, 64, 8)]
        elif len(data) == 8 and isinstance(data[0], list):
            board = data
        else:
            return "Error: Invalid board size (must be 64 or 8x8)."

        output = []
        output.append("   0 1 2 3 4 5 6 7")
        output.append("  +-----------------+")

        # 1=黒(●), -1=白(○), 0=空(.)
        chars = {0: ".", 1: "●", -1: "○", 2: "●", -2: "○"}

        for y in range(8):
            row_str = f"{y} |"
            for x in range(8):
                val = board[y][x]
                row_str += f" {chars.get(val, '?')}"
            row_str += " |"
            output.append(row_str)

        output.append("  +-----------------+")
        return "\n".join(output)

    except Exception as e:
        return f"Error parsing data: {e}\nInput was: {data_str}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Board data (JSON string)")
    args = parser.parse_args()
    print(render_board(args.data))


if __name__ == "__main__":
    main()
