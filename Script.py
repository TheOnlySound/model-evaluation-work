import csv
import sys

CRITERIA = ["tool_called", "no_url", "no_markdown", "concise", "sounds_natural"]


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for c in CRITERIA:
            row[c] = row[c] == "True"
        row["latency_s"] = float(row["latency_s"])
        row["passed"] = all(row[c] for c in CRITERIA) and not row["error"]
    return rows


def group_rows(rows):
    groups = {}
    for row in rows:
        key = (row["model"], row["prompt_variant"])
        groups.setdefault(key, []).append(row)
    return groups


def print_model_table(groups):
    print("RESULTS BY MODEL AND PROMPT")
    print(f"{'Model':<10}{'Prompt':<10}{'Full pass':<11}{'Tool':<6}{'No URL':<8}"
          f"{'No MD':<7}{'Concise':<9}{'Natural':<9}{'Errors':<8}{'Mean latency':<14}Max latency")
    for (model, prompt), g in groups.items():
        counts = [sum(r[c] for r in g) for c in CRITERIA]
        passes = sum(r["passed"] for r in g)
        full = f"{passes}/{len(g)}"
        errors = sum(1 for r in g if r["error"])
        latencies = [r["latency_s"] for r in g]
        mean = sum(latencies) / len(latencies)
        print(f"{model:<10}{prompt:<10}{full:<11}{counts[0]:<6}{counts[1]:<8}"
              f"{counts[2]:<7}{counts[3]:<9}{counts[4]:<9}{errors:<8}"
              f"{mean:<14.2f}{max(latencies):.2f}")


def print_prompt_rates(rows, variants):
    print("FULL-PASS RATE BY PROMPT (all models)")
    for v in variants:
        vr = [r for r in rows if r["prompt_variant"] == v]
        passes = sum(r["passed"] for r in vr)
        print(f"{v:<10}{passes}/{len(vr)} ({passes / len(vr):.0%})")
    print()
    print("NON-ENGLISH FULL PASSES BY PROMPT")
    for v in variants:
        vr = [r for r in rows if r["prompt_variant"] == v and r["language"] != "en"]
        passes = sum(r["passed"] for r in vr)
        print(f"{v:<10}{passes}/{len(vr)}")


def print_failures(rows):
    print("FAILED RESPONSES")
    for r in rows:
        if r["passed"]:
            continue
        label = f"{r['id']} - {r['language']}"
        if r["error"]:
            reason = "error: " + r["error"]
        else:
            reason = "failed: " + ", ".join(c for c in CRITERIA if not r[c])
        print(f"{r['model']:<10}{r['prompt_variant']:<10}{label:<10}{reason}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    rows = load_rows(path)
    groups = group_rows(rows)
    variants = list(dict.fromkeys(r["prompt_variant"] for r in rows))

    print(f"Dataset: {path} ({len(rows)} graded responses)")
    print("A response passes only if it meets all criteria: " + ", ".join(CRITERIA))
    print()
    print_model_table(groups)
    print()
    print_prompt_rates(rows, variants)
    print()
    print_failures(rows)

if __name__ == "__main__":
    main()


