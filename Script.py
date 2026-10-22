import csv
import sys
from pathlib import Path

# Pass/fail criteria graded for every response. A response passes only if it meets all of them.
CRITERIA = ["tool_called", "no_url", "no_markdown", "concise", "sounds_natural"]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "eval_set.csv"
RESULTS_FILE = SCRIPT_DIR / "eval_results.txt"


def load_dataset(file_path):
    with open(file_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for criterion in CRITERIA:
            row[criterion] = row[criterion].strip() == "True"
        row["latency_s"] = float(row["latency_s"])
        row["passed"] = all(row[c] for c in CRITERIA)
    return rows


def run_model_eval(rows):
    """Summarise one model/prompt configuration."""
    latencies = [r["latency_s"] for r in rows]
    return {
        "total": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "criteria": {c: sum(r[c] for r in rows) for c in CRITERIA},
        "errors": sum(1 for r in rows if r["error"].strip()),
        "mean_latency": sum(latencies) / len(latencies),
        "max_latency": max(latencies),
    }


def group_by(rows, *keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return groups


def build_report(dataset_path, rows):
    lines = [
        f"Dataset: {dataset_path.name} ({len(rows)} graded responses)",
        "A response passes only if it meets all criteria: " + ", ".join(CRITERIA),
        "",
        "RESULTS BY MODEL AND PROMPT",
        f"{'Model':<10}{'Prompt':<10}{'Full pass':<11}{'Tool':<6}{'No URL':<8}{'No MD':<7}"
        f"{'Concise':<9}{'Natural':<9}{'Errors':<8}{'Mean latency':<14}{'Max latency'}",
    ]
    for (model, variant), group in sorted(group_by(rows, "model", "prompt_variant").items(),
                                          key=lambda item: (item[0][1], item[0][0])):
        r = run_model_eval(group)
        c = r["criteria"]
        lines.append(
            f"{model:<10}{variant:<10}{str(r['passed']) + '/' + str(r['total']):<11}"
            f"{c['tool_called']:<6}{c['no_url']:<8}{c['no_markdown']:<7}{c['concise']:<9}"
            f"{c['sounds_natural']:<9}{r['errors']:<8}{r['mean_latency']:<14.2f}{r['max_latency']:.2f}"
        )

    lines += ["", "FULL-PASS RATE BY PROMPT (all models)"]
    for (variant,), group in sorted(group_by(rows, "prompt_variant").items()):
        r = run_model_eval(group)
        lines.append(f"{variant:<10}{r['passed']}/{r['total']} ({r['passed'] / r['total']:.0%})")

    lines += ["", "NON-ENGLISH FULL PASSES BY PROMPT"]
    non_english = [row for row in rows if row["language"] != "en"]
    for (variant,), group in sorted(group_by(non_english, "prompt_variant").items()):
        r = run_model_eval(group)
        lines.append(f"{variant:<10}{r['passed']}/{r['total']}")

    lines += ["", "FAILED RESPONSES"]
    for row in rows:
        if not row["passed"]:
            failed = ", ".join(c for c in CRITERIA if not row[c])
            lines.append(f"{row['model']:<10}{row['prompt_variant']:<10}{row['id']}-{row['language']:<6}failed: {failed}")

    return "\n".join(lines) + "\n"


def main():
    dataset_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_DATASET
    print(f"Starting model evaluation on {dataset_path.name}...\n")
    rows = load_dataset(dataset_path)
    report = build_report(dataset_path, rows)
    print(report)
    RESULTS_FILE.write_text(report, encoding="utf-8")
    print(f"Results saved to {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
