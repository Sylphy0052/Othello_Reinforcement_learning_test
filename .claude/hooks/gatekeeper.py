import subprocess
import sys


def run_check(command, name):
    print(f"🛡️  Running {name}...", end=" ", flush=True)
    try:
        # capture_output=True で出力を隠し、エラー時のみ表示する
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print("✅ Passed")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed")
        print(f"\n--- {name} Error Logs ---")
        print(e.stderr.decode() or e.stdout.decode())
        print("-------------------------")
        return False


def main():
    print("\n🚢 Pre-ship Inspection Started...\n")

    checks = [
        # 1. Unit Tests (Pytest)
        ("python3 -m pytest", "Unit Tests"),
        # 2. Type Check (Mypy) - 厳しすぎるなら外してもOK
        # ("python3 -m mypy src", "Type Check"),
        # 3. Syntax Check (Compile)
        ("python3 -m py_compile src/**/*.py", "Syntax Check"),
    ]

    all_passed = True
    for cmd, name in checks:
        if not run_check(cmd, name):
            all_passed = False
            break  # 1つでも失敗したら即終了

    if not all_passed:
        print("\n🚫 Ship Aborted: Fix the errors above first.")
        sys.exit(1)  # Exit 1 を返すと Claude Code はコマンドを中断する

    print("\n✨ All checks passed. Ready to ship!")
    sys.exit(0)


if __name__ == "__main__":
    main()
