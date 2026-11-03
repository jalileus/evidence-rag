"""Multiple models choices + OpenAI-compatible Chat Completions RAG predictor.

Experiments controlled via runtime.json["experiment"]["name"]:
  no_context       — no retrieval; LLM answers from memory only
  rag_single       — BM25 retrieval + single LLM call, no retry
  final            — BM25 retrieval + retry on bad citations  (default)
  ablation_no_rerank — like final but passages are taken in corpus order not ranked by BM25 score
  ablation_top1    — like final but top_k is forced to 1
"""

import argparse
import json
import os
import pickle
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MATH_SYMBOLS = "+\\-−*/=<>≤≥≠≈∂∆△∇∑∏√∞∈∉⊂⊃⊆⊇∩∪∅" "∧∨∀∃→↔ℂℝℤℕ∫∮×·⋅±"
WORD_PATTERN = re.compile(r"[^\W\_]+", re.UNICODE)
MATH_PATTERN = re.compile(
    r"[^\W\_]+|[" + re.escape(MATH_SYMBOLS) + r"]",
    re.UNICODE,
)

VALID_EXPERIMENTS = {
    "no_context",
    "rag_single",
    "final",
    "ablation_no_rerank",
    "ablation_top1",
}



def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_output_path(base_path: str, experiment_name: str) -> str:
    if not experiment_name or experiment_name in base_path:
        return base_path
    p = Path(base_path)
    return str(p.parent / f"{p.stem}_{experiment_name}{p.suffix}")



def tokenize(text, cfg):
    settings = cfg["tokenization"]
    pattern = MATH_PATTERN if settings["include_math_symbols"] else WORD_PATTERN
    tokens = pattern.findall(text)
    if settings["lowercase"]:
        tokens = [token.casefold() for token in tokens]
    min_length = settings["min_token_length"]
    if min_length > 1:
        tokens = [token for token in tokens if len(token) >= min_length]
    return tokens



def load_bm25(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for key in ("bm25", "index", "model"):
            if key in obj and hasattr(obj[key], "get_scores"):
                return obj[key]
    if hasattr(obj, "get_scores"):
        return obj
    raise TypeError(
        "index/bm25.pkl is not a BM25 object with get_scores(); "
        "adapt load_bm25() to build_index.py's format."
    )



def retrieve(docs, bm25, tokens, top_k, experiment_name):
    if experiment_name == "ablation_no_rerank":
        # Take the first top_k docs in their original corpus order
        passages = docs[:top_k]
    else:
        scores = bm25.get_scores(tokens)
        order = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )[:top_k]
        passages = [docs[i] for i in order]
    return passages



def call_llm(messages, qid, timeout, max_tokens, temperature=0.0):
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    model = os.environ.get("LLM_MODEL", "")
    key = os.environ.get("LLM_API_KEY", "")
    submission_id = os.environ.get("SUBMISSION_ID", "")
    if not base or not model:
        raise RuntimeError(
            "Set LLM_BASE_URL and LLM_MODEL to an OpenAI-compatible endpoint."
        )
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Question-ID": qid,
        "X-Submission-ID": submission_id,
    }
    if key:
        headers["Authorization"] = "Bearer " + key
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            body = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"LLM HTTP {e.code}: {e.read().decode(errors='replace')[:700]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach LLM endpoint {url}: {e}") from e
    text = body["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = "".join(x.get("text", "") for x in text if isinstance(x, dict))
    usage = body.get("usage", {})
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        raise RuntimeError(
            "API omitted usage.prompt_tokens/completion_tokens; "
            "cannot create accurate token trace."
        )
    return text, int(usage["prompt_tokens"]), int(usage["completion_tokens"])



def build_no_context_prompt(question_text, options):
    """Prompt with NO retrieved evidence"""
    system = (
        "Ты эксперт по ТФКП. Отвечай на основе своих знаний. "
        "Не добавляй Markdown. Верни только JSON."
    )
    user = f"""Реши вопрос по ТФКП.

ПРАВИЛА ОТВЕТА:
- Выбери все верные варианты из списка.
- Если знаний недостаточно для уверенного ответа — установи abstain=true, оба списка пустыми.
- Если отвечаешь (abstain=false): answer_ids НЕПУСТОЙ, citations ПУСТОЙ (нет фрагментов).
- rationale: непустое обоснование, не более 300 символов (кратко).
- Верни ТОЛЬКО JSON без Markdown:
{{"answer_ids":["A","C"],"abstain":false,"rationale":"Обоснование","citations":[]}}

ВОПРОС: {question_text}

ВАРИАНТЫ:
{options}

JSON:"""
    return system, user


def build_main_prompt(evidence, question_text, options):
    """Standard RAG prompt with retrieved passages"""
    system = (
        "Ты эксперт по ТФКП. Отвечай строго по фрагментам. "
        "Не выдумывай факты. Не добавляй Markdown. Верни только JSON."
    )
    evidence_text = "\n\n".join(
        f"[{d['block_id']}] с.{d.get('page','?')} §{d.get('section','?')}\n{d['text']}"
        for d in evidence
    )
    user = f"""Реши вопрос по ТФКП, опираясь только на фрагменты ниже.

ПРАВИЛА ОТВЕТА:
- Выбери все верные варианты из списка.
- Если фрагментов недостаточно для уверенного ответа — установи abstain=true, оба списка пустыми.
- Если в вопросе отсутствует существенное условие (например, ориентация контура, конкретное значение параметра),
  которое нужно для однозначного ответа — это признак недостаточности данных: abstain=true.
- Если отвечаешь (abstain=false): answer_ids и citations оба НЕПУСТЫЕ.
- citations: от 1 до 4 block_id только из фрагментов ниже. НЕ выдумывай block_id.
- rationale: непустое обоснование, не более 300 символов (кратко).
- Верни ТОЛЬКО JSON без Markdown, пробелов до/после:
{{"answer_ids":["A","C"],"abstain":false,"rationale":"Обоснование","citations":["b_0001","b_0002"]}}

ФРАГМЕНТЫ:
{evidence_text}

ВОПРОС: {question_text}

ВАРИАНТЫ:
{options}

JSON:"""
    return system, user


def build_retry_prompt(prev_response, evidence, question_text, options, reason):
    """Retry prompt"""
    system = (
        "Ты эксперт по ТФКП. Отвечай строго по фрагментам. "
        "Не выдумывай факты. Не добавляй Markdown. Верни только JSON."
    )
    evidence_text = "\n\n".join(
        f"[{d['block_id']}] с.{d.get('page','?')} §{d.get('section','?')}\n{d['text']}"
        for d in evidence
    )
    available_ids = ", ".join(d["block_id"] for d in evidence)
    user = f"""Реши вопрос по ТФКП, опираясь только на фрагменты ниже.

ПРАВИЛА ОТВЕТА:
- Выбери все верные варианты из списка.
- Если фрагментов недостаточно для уверенного ответа — установи abstain=true, оба списка пустыми.
- Если отвечаешь (abstain=false): answer_ids и citations оба НЕПУСТЫЕ.
- citations: от 1 до 4 block_id только из фрагментов ниже. НЕ выдумывай block_id.
- rationale: непустое обоснование, не более 300 символов (кратко).
- Верни ТОЛЬКО JSON без Markdown, пробелов до/после:
{{"answer_ids":["A","C"],"abstain":false,"rationale":"Обоснование","citations":["b_0001","b_0002"]}}

ФРАГМЕНТЫ:
{evidence_text}

ВОПРОС: {question_text}

ВАРИАНТЫ:
{options}

JSON:"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": prev_response},
        {
            "role": "user",
            "content": (
                f"ОШИБКА в предыдущем ответе: {reason}\n"
                f"Доступные block_id: {available_ids}\n"
                "Исправь ответ. Верни ТОЛЬКО корректный JSON:"
            ),
        },
    ]
    return messages


def validate(obj, valid_options, block_ids, allow_empty_citations=False):
    """
    Validate parsed JSON against schema rules
    allow_empty_citations=True is used for no_context mode
    Returns (abstain, ids, citations, rationale) or raises ValueError
    """
    abstain = obj.get("abstain", False)
    ids = obj.get("answer_ids", [])
    citations = obj.get("citations", [])
    rationale = obj.get("rationale", "")

    if not isinstance(abstain, bool):
        raise ValueError("'abstain' must be a boolean")
    if not isinstance(ids, list):
        raise ValueError("'answer_ids' must be a list")
    if not isinstance(citations, list):
        raise ValueError("'citations' must be a list")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("'rationale' must be a non-empty string")

    ids = list(dict.fromkeys(str(x) for x in ids))
    citations = list(dict.fromkeys(str(x) for x in citations))[:4]

    bad_opts = [x for x in ids if x not in valid_options]
    if bad_opts:
        raise ValueError(f"Unknown option IDs: {bad_opts}")

    if not allow_empty_citations:
        bad_cites = [x for x in citations if x not in block_ids]
        if bad_cites:
            raise ValueError(f"Citations not in retrieved blocks: {bad_cites}")

    if abstain:
        ids, citations = [], []
    else:
        if not ids:
            raise ValueError("Non-abstaining answer needs answer_ids")
        if not allow_empty_citations and not citations:
            raise ValueError("Non-abstaining answer needs citations")

    return abstain, ids, citations, rationale.strip()



def answer_question(q, docs, bm25, cfg, experiment_name, top_k, timeout, max_tokens, temperature):
    """
    Run one question through the appropriate pipeline for experiment_name.
    Returns (answer_dict, trace_dict).
    """
    qid = str(q["id"])
    start = time.perf_counter()

    question_text = q.get("question", "")
    valid_options = {str(o["id"]) for o in q.get("options", [])}
    options_text = "\n".join(f"{o['id']}: {o['text']}" for o in q.get("options", []))

    total_calls = 0
    total_in = 0
    total_out = 0
    final_answer = None
    last_error = None
    passages = []

    effective_top_k = 1 if experiment_name == "ablation_top1" else top_k

    # All experiments except no_context
    if experiment_name != "no_context":
        option_texts = " ".join(o["text"] for o in q.get("options", []))
        tokens = tokenize(question_text + " " + option_texts, cfg)
        passages = retrieve(docs, bm25, tokens, effective_top_k, experiment_name)

    block_ids = set(d["block_id"] for d in passages)

    if experiment_name == "no_context":
        system, user = build_no_context_prompt(question_text, options_text)
    else:
        system, user = build_main_prompt(passages, question_text, options_text)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        raw, in_tok, out_tok = call_llm(messages, qid, timeout, max_tokens, temperature)
        total_calls += 1
        total_in += in_tok
        total_out += out_tok

        obj = parse_response(raw)
        allow_empty = (experiment_name == "no_context")
        abstain, ids, citations, rationale = validate(
            obj, valid_options, block_ids, allow_empty_citations=allow_empty
        )
        final_answer = (abstain, ids, citations, rationale)

    except ValueError as e:
        last_error = str(e)

        # Retry logic: only for 'final' and ablation variants 
        retry_eligible = experiment_name not in ("rag_single", "no_context")
        citation_error = any(kw in last_error for kw in ("citations", "Citation"))

        if retry_eligible and citation_error:
            try:
                retry_messages = build_retry_prompt(
                    raw, passages, question_text, options_text, last_error
                )
                raw2, in_tok2, out_tok2 = call_llm(
                    retry_messages, qid, timeout, max_tokens, temperature
                )
                total_calls += 1
                total_in += in_tok2
                total_out += out_tok2

                obj2 = parse_response(raw2)
                abstain, ids, citations, rationale = validate(
                    obj2, valid_options, block_ids
                )
                final_answer = (abstain, ids, citations, rationale)
                last_error = None

            except Exception as e2:
                last_error = f"Retry also failed: {e2}"

    except Exception as e:
        last_error = str(e)

    elapsed_ms = round((time.perf_counter() - start) * 1000)

    if final_answer is not None:
        abstain, ids, citations, rationale = final_answer
        answer = {
            "id": qid,
            "answer_ids": ids,
            "abstain": abstain,
            "rationale": rationale[:1200],
            "citations": citations,
        }
    else:
        answer = {
            "id": qid,
            "answer_ids": [],
            "abstain": True,
            "rationale": ("Ошибка: " + (last_error or "неизвестно"))[:1200],
            "citations": [],
        }

    trace = {
        "id": qid,
        "llm_calls": total_calls,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "elapsed_ms": elapsed_ms,
        "retrieved_block_ids": list(block_ids),
    }

    return answer, trace


def parse_response(text):
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("Model did not return a valid JSON object: " + text[:300])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--runtime", default="runtime.json")
    ap.add_argument(
        "--experiment-name",
        default=None,
        help=(
            f"Which experiment to run. One of: {', '.join(sorted(VALID_EXPERIMENTS))}. "
            "Falls back to runtime.json[\"experiment\"][\"name\"], then 'final'."
        ),
    )
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    runtime_path = Path(a.runtime)
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.exists()
        else {}
    )

    # Experiment name
    experiment_name = (
        a.experiment_name
        or runtime.get("experiment", {}).get("name", "final")
    )
    if experiment_name not in VALID_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. "
            f"Choose one of: {', '.join(sorted(VALID_EXPERIMENTS))}"
        )

    output_path = resolve_output_path(a.output, experiment_name)
    trace_path  = resolve_output_path(a.trace,  experiment_name)

    print(f"=== Experiment: {experiment_name} ===")
    print(f"  answers → {output_path}")
    print(f"  trace   → {trace_path}")

    # Runtime params
    retrieval_cfg  = runtime.get("retrieval", {})
    generation_cfg = runtime.get("generation", {})

    top_k = a.top_k if a.top_k is not None else int(retrieval_cfg.get("top_k", 5))
    timeout = (
        a.timeout if a.timeout is not None else int(runtime.get("timeout_seconds", 180))
    )
    max_tokens  = int(generation_cfg.get("max_tokens", 512))
    temperature = float(generation_cfg.get("temperature", 0.0))

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if timeout < 1:
        raise ValueError("timeout_seconds must be >= 1")
    if not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be between 1 and 4096")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    # Load index
    docs = read_jsonl(Path(a.index) / "documents.jsonl")
    bm25 = load_bm25(Path(a.index) / "bm25.pkl")
    questions = read_jsonl(a.questions)

    answers, traces = [], []

    for n, q in enumerate(questions, 1):
        qid = str(q["id"])
        answer, trace = answer_question(
            q, docs, bm25, cfg,
            experiment_name, top_k, timeout, max_tokens, temperature,
        )
        answers.append(answer)
        traces.append(trace)

        label = answer["answer_ids"] or "Отказ"
        retry_note = f" [retry, calls={trace['llm_calls']}]" if trace["llm_calls"] > 1 else ""
        print(f"[{n}/{len(questions)}] {qid}: {label}{retry_note}")

    write_jsonl(output_path, answers)
    write_jsonl(trace_path, traces)
    print(f"Done. Wrote {output_path} and {trace_path}")


if __name__ == "__main__":
    main()