import json
from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def run():
    print(f"{'Scenario':25} {'Baseline':10} {'Waggle':10} {'Reduction %'}")
    print("-" * 60)

    for file in SCENARIOS_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        turns = data["turns"]

        baseline = sum(
            estimate_tokens(" ".join(turns[: i + 1]))
            for i in range(len(turns))
        )

        waggle = sum(
            estimate_tokens(turn)
            for turn in turns
        )

        reduction = (
            ((baseline - waggle) / baseline) * 100
            if baseline
            else 0
        )

        print(
            f"{data['name']:25} "
            f"{baseline:<10} "
            f"{waggle:<10} "
            f"{reduction:.1f}%"
        )


if __name__ == "__main__":
    run()