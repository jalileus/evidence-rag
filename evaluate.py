"""
Evaluation for all experiments and creation of experiments.csv.

Usage:
    python evaluate.py --gold ./data/dev_gold.jsonl --answers-dir data --traces-dir data
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

EXPERIMENTS = [
    "final",
    "no_context",
    "rag_single",
    "ablation_no_rerank",
    "ablation_top1",
]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalise_block_id(bid: str) -> str:
    return bid.replace("_", "").lower()


def get_citation_sets(gold: dict) -> list:
    if "citation_sets" in gold:
        raw = gold["citation_sets"]
        if raw and not isinstance(raw[0], list):
            return [raw]
        return raw
    if "citations" in gold:
        return [gold["citations"]]
    return []


def citation_f1(pred_citations: list, gold_sets: list) -> float:
    if not gold_sets:
        return 1.0

    pred = {normalise_block_id(b) for b in pred_citations}
    best = 0.0

    for gold in gold_sets:
        g = {normalise_block_id(b) for b in gold}

        if not g:
            best = max(best, 1.0)
            continue

        tp = len(pred & g)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(g)

        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

        best = max(best, f1)

    return best


def score_question(pred: dict, gold: dict):
    category = gold.get("category", "unknown")

    if gold.get("abstain", False):
        correct = pred.get("abstain", False)
        s = 1.0 if correct else 0.0
        return (
            s,
            category,
            {
                "type": "abstain",
                "s": s,
                "correct": correct,
            },
        )

    if pred.get("abstain", False):
        return (
            0.0,
            category,
            {
                "type": "answerable",
                "A": 0.0,
                "F": 0.0,
                "s": 0.0,
                "correct": False,
            },
        )

    gold_ids = {str(x) for x in gold.get("answer_ids", [])}
    pred_ids = {str(x) for x in pred.get("answer_ids", [])}

    A = 1.0 if pred_ids == gold_ids else 0.0
    F = citation_f1(pred.get("citations", []), get_citation_sets(gold))
    s = A * (0.8 + 0.2 * F)

    return (
        s,
        category,
        {
            "type": "answerable",
            "A": A,
            "F": round(F, 4),
            "s": round(s, 4),
            "correct": A == 1.0,
            "pred_ids": sorted(pred_ids),
            "gold_ids": sorted(gold_ids),
        },
    )


def compute_cost(trace: dict) -> float:
    return (
        trace.get("input_tokens", 0)
        + 4 * trace.get("output_tokens", 0)
        + 256 * trace.get("llm_calls", 0)
    )


def evaluate_one(name, answers_path, trace_path, golds, c_ref):
    if not answers_path.exists():
        print(f"Skipping {name}: {answers_path.name} not found.")
        return None

    if not trace_path.exists():
        print(f"Skipping {name}: {trace_path.name} not found.")
        return None

    preds = {str(r["id"]): r for r in read_jsonl(answers_path)}
    traces = {str(r["id"]): r for r in read_jsonl(trace_path)}

    missing = set(golds) - set(preds)

    if missing:
        print(
            f"Note: {len(missing)} question(s) have no prediction "
            "and will count as abstentions."
        )

    category_scores = {}
    scores_all, costs, details = [], [], []

    for qid, gold in golds.items():
        pred = preds.get(
            qid,
            {"abstain": True, "answer_ids": [], "citations": []},
        )
        trace = traces.get(
            qid,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "llm_calls": 0,
            },
        )

        s, cat, detail = score_question(pred, gold)
        detail["id"] = qid

        category_scores.setdefault(cat, []).append(s)
        scores_all.append(s)
        costs.append(compute_cost(trace))
        details.append(detail)

    cat_means = {
        cat: statistics.mean(values) for cat, values in category_scores.items()
    }

    Q = statistics.mean(cat_means.values()) if cat_means else 0.0
    mean_cost = statistics.mean(costs) if costs else 0.0

    if c_ref is not None:
        E = min(1.0, c_ref / max(mean_cost, 1))
        итог = 75 * Q + 25 * Q * E
    else:
        E, итог = None, None

    all_traces = list(traces.values())

    mean_in = (
        statistics.mean(t.get("input_tokens", 0) for t in all_traces)
        if all_traces
        else 0
    )

    mean_out = (
        statistics.mean(t.get("output_tokens", 0) for t in all_traces)
        if all_traces
        else 0
    )

    mean_calls = (
        statistics.mean(t.get("llm_calls", 0) for t in all_traces) if all_traces else 0
    )

    median_lat = (
        statistics.median(t.get("elapsed_ms", 0) for t in all_traces)
        if all_traces
        else 0
    )

    print(f"\nExperiment: {name}")
    print(f"Questions evaluated: {len(golds)}")

    print("\nScores by category:")
    for cat, mean in sorted(cat_means.items()):
        n = len(category_scores[cat])
        print(f"  {cat}: {mean:.2f} ({n} questions)")

    print(f"\nQ (mean of category means): {Q:.4f}")

    if итог is not None:
        print(
            f"Efficiency: {E:.4f} "
            f"(mean cost: {mean_cost:.0f}, reference cost: {c_ref:.0f})"
        )
        print(f"Final grade: {итог:.2f} / 100")
    else:
        print("Efficiency: not calculated (no reference cost).")

    print(f"\nAverage input tokens: {mean_in:.0f}")
    print(f"Average output tokens: {mean_out:.0f}")
    print(f"Average LLM calls: {mean_calls:.2f}")
    print(f"Average cost: {mean_cost:.0f}")
    print(f"Median latency: {median_lat:.0f} ms")

    return {
        "name": name,
        "answer_accuracy": (f"{statistics.mean(scores_all):.4f}" if scores_all else ""),
        "Q": "" if name == "no_context" else f"{Q:.4f}",
        "mean_input_tokens": f"{mean_in:.1f}",
        "mean_output_tokens": f"{mean_out:.1f}",
        "mean_llm_calls": f"{mean_calls:.2f}",
        "mean_cost": f"{mean_cost:.1f}",
        "median_latency_ms": f"{median_lat:.0f}",
        "_mean_cost_raw": mean_cost,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--answers-dir", default=".")
    ap.add_argument("--traces-dir", default=".")
    ap.add_argument("--csv", default="experiments.csv")
    a = ap.parse_args()

    golds = {str(r["id"]): r for r in read_jsonl(Path(a.gold))}

    answers_dir = Path(a.answers_dir)
    traces_dir = Path(a.traces_dir)
    csv_path = Path(a.csv)

    fieldnames = [
        "name",
        "answer_accuracy",
        "Q",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_llm_calls",
        "mean_cost",
        "median_latency_ms",
    ]

    # rag_single as the reference cost
    rag_single_trace_path = traces_dir / "dev_trace_rag_single.jsonl"
    c_ref = None

    if rag_single_trace_path.exists():
        ft = read_jsonl(rag_single_trace_path)

        if ft:
            c_ref = statistics.mean(compute_cost(t) for t in ft)
            print(f"\nReference cost (rag_single): {c_ref:.0f}")

    if c_ref is None:
        print(
            "Warning: dev_trace_rag_single.jsonl not found. "
            "Efficiency will not be calculated."
        )

    rows = []

    for name in EXPERIMENTS:
        result = evaluate_one(
            name,
            answers_dir / f"dev_answers_{name}.jsonl",
            traces_dir / f"dev_trace_{name}.jsonl",
            golds,
            c_ref,
        )

        if result:
            rows.append(result)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)

    print("\nExperiment summary:")
    for r in rows:
        q_value = r["Q"] if r["Q"] else "N/A"
        print(f"{r['name']}: " f"answer_accuracy={r['answer_accuracy']}, Q={q_value}")

    print(f"\nCSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
