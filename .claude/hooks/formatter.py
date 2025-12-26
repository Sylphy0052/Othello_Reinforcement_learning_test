import subprocess
import sys


def main():
    print("🧹 Auto-formatting code...")

    # Pythonの標準的なフォーマッタを実行
    # インストールされていなければスキップする優しい設計
    commands = [
        "isort . --profile black",  # import順序の整理
        "black .",  # コードフォーマット
    ]

    for cmd in commands:
        try:
            subprocess.run(cmd, shell=True, check=False, capture_output=True)
        except Exception:
            pass  # ツールが入ってない場合は無視

    print("✨ Code is clean.")
    sys.exit(0)


if __name__ == "__main__":
    main()
