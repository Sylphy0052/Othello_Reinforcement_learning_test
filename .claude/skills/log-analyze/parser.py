import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./logs")
    args = parser.parse_args()

    # ダミー分析ロジック
    # 本来は pandas で CSV を読み込んで移動平均などを計算する

    current_loss = random.uniform(0.01, 0.05)

    analysis = {
        "status": "success",
        "latest_epoch": 1500,
        "metrics": {
            "loss": {
                "current": round(current_loss, 4),
                "trend": "decreasing" if random.random() > 0.3 else "stagnated",
            },
            "epsilon": 0.15,
            "win_rate_moving_avg": 0.58,
        },
        "health_check": "HEALTHY",
    }

    if analysis["metrics"]["loss"]["trend"] == "stagnated":
        analysis["health_check"] = "WARNING"
        analysis["advice"] = "Loss is not decreasing. Try reducing the Learning Rate."

    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
