"""Recall-channel evaluation runner for the four retrieval channels.

Evaluates search_history / search_security / search_codebase (RAGRetriever)
and search_feedback (LongTermMemory) against a FIXED seed snapshot
(seed_data.json) loaded into a throwaway SQLite database, so live-library
drift can never pollute scores.

Layers (per the agreed design):
- deterministic: FTS/SQL/filter semantics. Golden + regression groups are
  GATED (failure -> exit code 1). Runs offline with a deterministic fake
  embedding so results never depend on an external model.
- semantic: embedding-quality cases. Requires --live (real embedding
  model on both write and query side). Never gated; recall@k / MRR are
  recorded for baseline comparison.
- challenge: known-limitation observations (top-k cap, over-recall on
  garbage queries). Recorded, never gated.

Usage:
    python evals/recall/run_eval.py            # offline, deterministic layer
    python evals/recall/run_eval.py --live     # + semantic layer with real embeddings
    python evals/recall/run_eval.py --verbose  # per-case detail
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from src.memory.long_term import LongTermMemory  # noqa: E402
from src.rag.retriever import RAGRetriever  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic offline embedding (bag-of-words, crc32 buckets). Mirrors the
# unit-test fake: stable across processes, good enough that FTS guarantees
# drive the deterministic-layer outcomes.
# ---------------------------------------------------------------------------


class DeterministicEmbedding:
    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 512
        for token in text.lower().split():
            vec[zlib.crc32(token.encode()) % 512] += 1.0
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    channel: str
    group: str
    layer: str
    status: str  # pass | fail | observe | skipped
    detail: str = ""
    mrr: float | None = None


@dataclass
class ChannelReport:
    gated_pass: int = 0
    gated_fail: int = 0
    semantic_recall: list[tuple[str, bool]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Seed ingestion
# ---------------------------------------------------------------------------


def load_seed(tmp_dir: Path, embed_client) -> tuple[dict, dict]:
    """Load seed rows into a fresh DB; return (handle, key->rowid map).

    The key map is namespaced by table ("past_risks:pr-031" -> rowid)
    because the four tables live in one SQLite file and their rowids
    overlap — a flat map would silently cross-map history rows to
    codebase rows sharing the same integer id.
    """
    seed = json.loads((HERE / "seed_data.json").read_text(encoding="utf-8"))
    db_path = str(tmp_dir / "eval.db")

    rag = RAGRetriever(db_path, embed_client)
    ltm = LongTermMemory(db_path)
    ltm.init_tables()

    key_to_rowid: dict[str, int] = {}

    # -- past_risks (production write path: computes embedding + FTS sync)
    for row in seed["past_risks"]:
        rag.add_history(
            row["thread_id"],
            row["file_path"],
            row["diff_summary"] or "",
            row["risk_titles"],
            row["risk_categories"],
            row["overall_score"] or 0.0,
        )
        key_to_rowid[f"past_risks:{row['key']}"] = _last_rowid(db_path, "past_risks")

    # -- security_knowledge (mirror the indexer write fields + FTS sync)
    conn = sqlite3.connect(db_path)
    try:
        for row in seed["security_knowledge"]:
            embedding = embed_client.embed(f"{row['title']} {row['content']}")
            cur = conn.execute(
                "INSERT INTO security_knowledge (title, category, rule_id,"
                " content, best_practice, embedding, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row["title"], row["category"], row["rule_id"],
                    row["content"], row["best_practice"],
                    json.dumps(embedding), "2026-01-01",
                ),
            )
            key_to_rowid[f"security_knowledge:{row['key']}"] = cur.lastrowid
            conn.execute(
                "INSERT INTO security_knowledge_fts (rowid, title, content,"
                " best_practice) VALUES (?, ?, ?, ?)",
                (cur.lastrowid, row["title"], row["content"],
                 row["best_practice"] or ""),
            )
        # -- codebase_index (post-F: no embeddings; read path is exact-path)
        for row in seed["codebase_index"]:
            cur = conn.execute(
                "INSERT INTO codebase_index (file_path, symbol_name,"
                " symbol_type, line_range, content, imports, embedding,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, NULL, '2026-01-01')",
                (
                    row["file_path"], row["symbol_name"], row["symbol_type"],
                    row["line_range"], row["content"],
                    json.dumps(row["imports"]),
                ),
            )
            key_to_rowid[f"codebase_index:{row['key']}"] = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    # -- feedback (production write path: normalizes file_pattern)
    for row in seed["feedback"]:
        ltm.add_feedback(
            row["thread_id"], row["file_pattern"], row["rule_id"],
            row["feedback_type"], row["feedback_content"],
        )
        key_to_rowid[f"feedback:{row['key']}"] = _last_rowid(db_path, "feedback")

    handle = {"rag": rag, "ltm": ltm}
    return handle, key_to_rowid


def _last_rowid(db_path: str, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT MAX(id) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Channel runners
# ---------------------------------------------------------------------------


def run_history(handle, key_to_rowid: dict, case: dict) -> CaseResult:
    inp = case["input"]
    results = handle["rag"].search_history(inp["query"], inp.get("file_pattern"))
    return _judge_id_results(handle_case=case, results=results,
                             id_key="id", table="past_risks",
                             key_to_rowid=key_to_rowid,
                             channel="history")


def run_security(handle, key_to_rowid: dict, case: dict) -> CaseResult:
    inp = case["input"]
    results = handle["rag"].search_security(inp["query"], inp.get("rule_ids"))
    rowid_to_key = _invert(key_to_rowid, "security_knowledge")

    expected_cats = set(case.get("expected_categories") or [])
    forbidden_cats = set(case.get("forbidden_categories") or [])
    got_cats = [r.get("rule_id") for r in results]
    got_keys = [rowid_to_key.get(r.get("id"), "?") for r in results]

    detail = f"returned {len(results)} rows, rule_ids={got_cats[:5]}, keys={got_keys[:5]}"

    if case.get("expected_empty"):
        return CaseResult(case["id"], "security", case["group"], case["layer"],
                          "observe" if case["group"] == "challenge" else "pass"
                          if not results else "fail", detail)
    if case.get("expected_nonempty"):
        ok = bool(results)
        return CaseResult(case["id"], "security", case["group"], case["layer"],
                          "pass" if ok else "fail", detail)
    if expected_cats:
        hit = any(c in expected_cats for c in got_cats)
        bad = any(c in forbidden_cats for c in got_cats)
        ok = hit and not bad
        mrr = None
        for rank, c in enumerate(got_cats, 1):
            if c in expected_cats:
                mrr = 1.0 / rank
                break
        if case.get("expect_first_category"):
            ok = bool(got_cats) and got_cats[0] in expected_cats
        return CaseResult(case["id"], "security", case["group"], case["layer"],
                          "pass" if ok else "fail", detail, mrr)
    return CaseResult(case["id"], "security", case["group"], case["layer"],
                      "observe", detail)


def run_codebase(handle, key_to_rowid: dict, case: dict) -> CaseResult:
    inp = case["input"]
    results = handle["rag"].search_codebase(inp["file_path"])
    rowid_to_key = _invert(key_to_rowid, "codebase_index")
    diff_file = [r for r in results if r.get("source") == "diff_file"]
    cross = [r for r in results if r.get("source") == "cross_file"]
    diff_keys = [rowid_to_key.get(r.get("id"), "?") for r in diff_file]

    expected = case.get("expected_ids") or []

    if expected == "OBSERVE":
        n = len(diff_file)
        return CaseResult(
            case["id"], "codebase", case["group"], case["layer"], "observe",
            f"diff_file rows={n} (cap observation, expected ~{case.get('observe_expected')})",
        )
    if expected == "ALL":
        want = case.get("expected_file_symbol_count")
        ok = len(diff_file) == want
        return CaseResult(
            case["id"], "codebase", case["group"], case["layer"],
            "pass" if ok else "fail",
            f"diff_file rows={len(diff_file)}, expected {want}",
        )
    if case.get("expected_cross_file_paths"):
        want = set(case["expected_cross_file_paths"])
        got = {r.get("file_path") for r in cross}
        ok = bool(got & want)
        return CaseResult(
            case["id"], "codebase", case["group"], case["layer"],
            "pass" if ok else "fail",
            f"cross_file paths={sorted(got)[:5]}",
        )
    if case.get("forbidden_other_files"):
        bad = {r.get("file_path") for r in diff_file} - {inp["file_path"]}
        return CaseResult(
            case["id"], "codebase", case["group"], case["layer"],
            "pass" if not bad else "fail",
            f"foreign files in diff_file results: {sorted(bad)[:5]}",
        )
    if case.get("expected_empty"):
        ok = not results
        return CaseResult(case["id"], "codebase", case["group"], case["layer"],
                          "pass" if ok else "fail",
                          f"returned {len(results)} rows")
    if expected:
        want = set(expected)
        got = set(diff_keys)
        ok = want <= got
        if case.get("expect_exact_order"):
            ok = ok and diff_keys == expected
        mrr = None
        for rank, k in enumerate(diff_keys, 1):
            if k in want:
                mrr = 1.0 / rank
                break
        return CaseResult(
            case["id"], "codebase", case["group"], case["layer"],
            "pass" if ok else "fail",
            f"diff_file keys={diff_keys[:10]}",
            mrr,
        )
    return CaseResult(case["id"], "codebase", case["group"], case["layer"],
                      "observe", f"diff_file rows={len(diff_file)}")


def run_feedback(handle, key_to_rowid: dict, case: dict) -> CaseResult:
    inp = case["input"]
    results = handle["ltm"].search_feedback(
        inp["file_path"], inp["symbols"], limit=10
    )
    return _judge_id_results(handle_case=case, results=results,
                             id_key="id", table="feedback",
                             key_to_rowid=key_to_rowid,
                             channel="feedback")


def _invert(key_to_rowid: dict, table: str) -> dict[int, str]:
    """Invert the namespaced key map for ONE table only."""
    prefix = f"{table}:"
    return {v: k.split(":", 1)[1] for k, v in key_to_rowid.items()
            if k.startswith(prefix)}


def _judge_id_results(handle_case: dict, results: list[dict],
                      id_key: str, table: str, key_to_rowid: dict,
                      channel: str) -> CaseResult:
    """Common pass/fail logic for id-based channels (history, feedback)."""
    case = handle_case
    rowid_to_key = _invert(key_to_rowid, table)
    got_keys = [rowid_to_key.get(r.get(id_key), "?") for r in results]
    expected = case.get("expected_ids") or []
    forbidden = set(case.get("forbidden_ids") or [])

    detail = f"returned keys={got_keys[:8]}"

    if case.get("expected_empty"):
        ok = not results
        bad = forbidden & set(got_keys) if results else set()
        ok = ok and not bad
        return CaseResult(case["id"], channel, case["group"], case["layer"],
                          "pass" if ok else "fail", detail)

    want = set(expected)
    hit = want & set(got_keys)
    bad = forbidden & set(got_keys)
    mrr = None
    for rank, k in enumerate(got_keys, 1):
        if k in want:
            mrr = 1.0 / rank
            break
    if case.get("expect_first"):
        ok = bool(got_keys) and got_keys[0] in want
    else:
        ok = bool(hit) and not bad
    return CaseResult(case["id"], channel, case["group"], case["layer"],
                      "pass" if ok else "fail", detail, mrr)


RUNNERS = {
    "history": run_history,
    "security": run_security,
    "codebase": run_codebase,
    "feedback": run_feedback,
}

CASE_FILES = {
    "history": "history.json",
    "security": "security.json",
    "codebase": "codebase.json",
    "feedback": "feedback.json",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Run semantic-layer cases with the real "
                             "embedding model (requires .env credentials).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-case detail lines.")
    args = parser.parse_args()

    embed_client = DeterministicEmbedding()
    if args.live:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
        from config.settings import settings
        from src.llm.client import EmbeddingClient

        embed_client = EmbeddingClient(
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            model=settings.embedding_model,
        )

    with tempfile.TemporaryDirectory(prefix="recall_eval_") as tmp:
        handle, key_to_rowid = load_seed(Path(tmp), embed_client)

        all_results: list[CaseResult] = []
        for channel, filename in CASE_FILES.items():
            cases = json.loads(
                (HERE / "cases" / filename).read_text(encoding="utf-8")
            )
            for case in cases:
                if case["layer"] == "semantic" and not args.live:
                    all_results.append(CaseResult(
                        case["id"], channel, case["group"], case["layer"],
                        "skipped", "semantic layer requires --live",
                    ))
                    continue
                try:
                    res = RUNNERS[channel](handle, key_to_rowid, case)
                except Exception as exc:  # noqa: BLE001
                    res = CaseResult(case["id"], channel, case["group"],
                                     case["layer"], "fail", f"EXCEPTION: {exc}")
                all_results.append(res)

    # ---- Aggregate & report ----
    reports: dict[str, ChannelReport] = {}
    for res in all_results:
        rep = reports.setdefault(res.channel, ChannelReport())
        gated = res.group in ("golden", "regression") and res.layer == "deterministic"
        if gated:
            if res.status == "pass":
                rep.gated_pass += 1
            elif res.status == "fail":
                rep.gated_fail += 1
        if res.layer == "semantic" and res.status in ("pass", "fail"):
            rep.semantic_recall.append((res.case_id, res.status == "pass"))
        if res.status == "observe":
            rep.observations.append(f"{res.case_id}: {res.detail}")

    print("=" * 72)
    print("RECALL CHANNEL EVALUATION")
    print("=" * 72)
    for channel in CASE_FILES:
        rep = reports.get(channel, ChannelReport())
        total = rep.gated_pass + rep.gated_fail
        line = (f"{channel:<12} gated: {rep.gated_pass}/{total} pass"
                if total else f"{channel:<12} gated: (no gated cases)")
        sem = ""
        if rep.semantic_recall:
            hits = sum(1 for _, ok in rep.semantic_recall if ok)
            sem = f"  semantic: {hits}/{len(rep.semantic_recall)} recall"
        print(line + sem)
        for note in rep.observations:
            print(f"             [observe] {note}")

    if args.verbose:
        print("-" * 72)
        for res in all_results:
            mrr = f" mrr={res.mrr:.2f}" if res.mrr else ""
            print(f"[{res.status.upper():<7}] {res.channel:<9} {res.case_id:<18}"
                  f"({res.group}/{res.layer}){mrr}  {res.detail[:90]}")

    failed = sum(r.gated_fail for r in reports.values())
    print("-" * 72)
    print(f"gated failures: {failed}" + ("  -> EXIT 1" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
