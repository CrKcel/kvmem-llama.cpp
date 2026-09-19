#!/usr/bin/env python3
"""Multi-fact needle cases on RTX 5050.

Default is a ~2k-token notes document. --target-tokens N pads filler so the
prompt is about N tokens (use 10000 for the long-doc / 1k-budget run).
Each case uses a short question; list-all is optional.

Methods:
  full       — KVMem identity (budget=0, no eviction). Model capability ceiling.
  recency    — equal tight budget; expect miss except the tail control fact.
  retrieval  — same budget; ranker may revive mid-document blocks.
  force      — retrieval + --kvmem-force-substr (writeback oracle).

Generation is auxiliary (small models hallucinate). Ranker/window membership
is the mechanism signal.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"

FILLER_PARA = (
    "On Tuesday the logistics desk recorded ordinary pallet movements across "
    "the north yard. Crews checked tire pressure, restacked empty crates, and "
    "signed the hourly sheet. The weather stayed overcast with light wind from "
    "the east. Radio chatter covered lunch rotations and a delayed inbound truck "
    "that later arrived without incident. Inventory counts matched the morning "
    "printout. None of these sentences contain a credential, password, badge, "
    "dock number, city name, supervisor identity, or access code."
)

INTRO = (
    "Read the following warehouse notes carefully. Several operational facts "
    "are buried in the text. After the notes, answer the question with the "
    "exact value only, no extra words.\n\n=== WAREHOUSE NOTES ===\n"
)


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    sentence: str
    question: str
    recency_control: bool = False


FACTS = (
    Fact(
        "access_code",
        "BLUEBIRD-42",
        "The secret access code is BLUEBIRD-42.",
        "What is the secret access code?",
    ),
    Fact(
        "supervisor",
        "Haruto Voss",
        "The night supervisor on duty is named Haruto Voss.",
        "What is the name of the night supervisor?",
    ),
    Fact(
        "password",
        "MAPLE-91",
        "The warehouse door password is MAPLE-91.",
        "What is the warehouse door password?",
    ),
    Fact(
        "city",
        "Kesseldorf",
        "The outbound shipment destination city is Kesseldorf.",
        "What city is the outbound shipment going to?",
    ),
    Fact(
        "dock",
        "PINE-88",
        "The assigned loading dock number is PINE-88.",
        "What is the assigned loading dock number?",
    ),
    Fact(
        "badge",
        "ORCHID-15",
        "The inspector badge serial is ORCHID-15.",
        "What is the inspector badge serial?",
    ),
    Fact(
        "locker",
        "CEDAR-33",
        "The staff locker pin is CEDAR-33.",
        "What is the staff locker pin?",
        recency_control=True,
    ),
)

LIST_QUESTION = (
    "List every secret access code, night supervisor name, warehouse door "
    "password, destination city, loading dock number, inspector badge serial, "
    "and staff locker pin mentioned in the notes."
)


def find_cli() -> Path:
    for p in (
        ROOT / "build/bin/llama-kvmem-cli",
        ROOT / "build/llama-kvmem-cli",
    ):
        if p.is_file():
            return p
    raise SystemExit("llama-kvmem-cli not found; run scripts/build-cuda.sh")


def filler(n: int, start: int) -> tuple[str, int]:
    if n <= 0:
        return "", start
    parts: list[str] = []
    i = start
    for _ in range(n):
        i += 1
        parts.append(f"Log entry {i}. {FILLER_PARA}")
    return "\n" + "\n".join(parts) + "\n", i


def n_mid_gaps() -> int:
    n_mid = sum(1 for f in FACTS if not f.recency_control)
    return max(n_mid - 1, 1)


def build_notes(gap: int, tail: int) -> str:
    body: list[str] = [INTRO]
    seq = 0
    for fact in FACTS:
        if fact.recency_control:
            chunk, seq = filler(tail, seq)
            body.append(chunk)
        elif len(body) > 1:
            chunk, seq = filler(gap, seq)
            body.append(chunk)
        body.append(fact.sentence + "\n")
    body.append("\n=== END OF NOTES ===\n")
    return "".join(body)


def make_prompt(notes: str, question: str) -> str:
    return notes + "\nQuestion: " + question + " Answer:"


def parse_trace(stderr: str) -> dict:
    info: dict = {
        "n_prompt": None,
        "force_pos": None,
        "needle_block": None,
        "selected": [],
        "stage_in_raw": [],
        "query_replay": None,
        "device_ok": "RTX 5050" in stderr,
    }
    for ln in stderr.splitlines():
        if m := re.search(r"n_prompt=(\d+)", ln):
            info["n_prompt"] = int(m.group(1))
        if m := re.search(r"force_pos=(\-?\d+)", ln):
            info["force_pos"] = int(m.group(1))
        if m := re.search(r"needle_block=(\-?\d+)", ln):
            info["needle_block"] = int(m.group(1))
        if m := re.search(r"block~=(\-?\d+)", ln):
            if info["needle_block"] is None:
                info["needle_block"] = int(m.group(1))
        if ln.startswith("KVMEM_TRACE selected") or "KVMEM_TRACE selected" in ln:
            info["selected"] = [int(x) for x in ln.split() if x.isdigit()]
        if "stage_in_raw block=" in ln:
            if m := re.search(r"block=(\d+)", ln):
                info["stage_in_raw"].append(int(m.group(1)))
        if "query_replay" in ln:
            info["query_replay"] = ln.strip()
    return info


def recency_window_hit(n_prompt: int | None, force_pos: int | None,
                       budget: int, block_tokens: int) -> bool | None:
    if n_prompt is None or force_pos is None or force_pos < 0:
        return None
    sink = block_tokens
    tail = max(budget - sink, block_tokens)
    return force_pos < sink or force_pos >= n_prompt - tail


def run_cli(cli: Path, model: Path, extra: list[str], prompt: str,
            n_predict: int) -> tuple[str, str]:
    env = gpu_env.apply_gpu(os.environ.copy(), "small")
    env.setdefault("KVMEM_TRACE", "1")
    # Hand the prompt over in a file. A single argv entry is capped by the
    # kernel at MAX_ARG_STRLEN (128 KiB on Linux), and a long-document prompt
    # blows past that well before the model's context does: a 32k-token prompt
    # is already ~167 KB, which fails with E2BIG ("Argument list too long").
    # The CLI's -f/--file reads the same text back verbatim, with no argv
    # ceiling and no shell quoting involved.
    prompt_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8", newline="\n") as fh:
            # Record the path before writing so write/flush failures also clean up.
            prompt_file = fh.name
            fh.write(prompt)
        cmd = [
            str(cli), "-m", str(model), "-n", str(n_predict), "-ngl", "99",
            "--no-prompt", "--temp", "0", *extra, "-f", prompt_file,
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                              env=env)
    finally:
        if prompt_file is not None:
            os.unlink(prompt_file)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"command failed rc={proc.returncode}")
    gpu_env.require_device(proc.stderr, "RTX 5050")
    return proc.stdout, proc.stderr


def kvmem_flags(method: str, budget: int, block_tokens: int, gen_reserve: int,
                ctx: int, batch: int, query_last: int, force_substr: str) -> list[str]:
    extra = [
        "--kvmem",
        "--kvmem-block-tokens", str(block_tokens),
        "--kvmem-budget", str(budget),
        "--kvmem-gen-reserve", str(gen_reserve),
        "-c", str(ctx),
        "-b", str(batch),
    ]
    if method in ("retrieval", "force"):
        extra += ["--kvmem-method", "retrieval", "--kvmem-query-last", str(query_last)]
        if force_substr:
            extra += ["--kvmem-force-substr", force_substr]
    elif method == "recency":
        extra += ["--kvmem-method", "recency"]
    return extra


def snippet(text: str, n: int = 160) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 3] + "..."


def hit_value(out: str, value: str) -> bool:
    return value.lower() in out.lower()


def count_prompt_tokens(cli: Path, model: Path, extra: list[str], prompt: str) -> int:
    _, err = run_cli(cli, model, extra, prompt, n_predict=1)
    tr = parse_trace(err)
    if not tr["n_prompt"]:
        raise SystemExit("failed to read n_prompt while sizing filler")
    return int(tr["n_prompt"])


def size_gap_tail(cli: Path, model: Path, target: int, budget: int,
                  block_tokens: int, gen_reserve: int, ctx: int,
                  batch: int) -> tuple[int, int, int]:
    """Pick gap/tail paragraph counts so n_prompt is about `target` tokens."""
    extra = kvmem_flags("recency", budget, block_tokens, gen_reserve, ctx, batch, 0, "")
    n0 = count_prompt_tokens(cli, model, extra, make_prompt(build_notes(0, 0), FACTS[0].question))
    n1 = count_prompt_tokens(cli, model, extra, make_prompt(build_notes(1, 0), FACTS[0].question))
    gaps = n_mid_gaps()
    # gap=1 inserts one paragraph in *each* mid-document gap.
    para = max((n1 - n0) // gaps, 1)
    # Keep the last mid-fact outside the recency window (budget minus sink).
    tail_tokens = budget + 2 * block_tokens
    tail = max(1, (tail_tokens + para - 1) // para)
    remain = max(target - n0 - tail * para, para)
    gap = max(1, remain // (gaps * para))
    est = n0 + gap * gaps * para + tail * para
    if est + gaps * para <= target + para:
        gap += 1
    return gap, tail, para


def locate_facts(cli: Path, model: Path, notes: str, budget: int,
                 block_tokens: int, gen_reserve: int, ctx: int, batch: int,
                 query_last: int, only: set[str] | None = None) -> dict[str, dict]:
    """One short retrieval+force run per fact to pin token pos / block id."""
    located: dict[str, dict] = {}
    for fact in FACTS:
        if only and fact.key not in only:
            continue
        p = make_prompt(notes, fact.question)
        extra = kvmem_flags("force", budget, block_tokens, gen_reserve, ctx, batch,
                            query_last, fact.value)
        out, err = run_cli(cli, model, extra, p, n_predict=1)
        tr = parse_trace(err)
        located[fact.key] = tr
        print(
            f"  locate {fact.key:12} n_prompt={tr['n_prompt']} "
            f"pos={tr['force_pos']} block={tr['needle_block']} "
            f"in_recency={recency_window_hit(tr['n_prompt'], tr['force_pos'], budget, block_tokens)}",
            flush=True,
        )
        del out
    return located


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, action="append", dest="models")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--block-tokens", type=int, default=32)
    ap.add_argument("--gen-reserve", type=int, default=128)
    ap.add_argument("--query-last", type=int, default=80)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gap", type=int, default=4, help="filler paragraphs between mid facts")
    ap.add_argument("--tail", type=int, default=4, help="filler paragraphs before recency-control fact")
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="if >0, scale filler so n_prompt is about this many tokens")
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--n-predict-list", type=int, default=96)
    ap.add_argument("--methods", default="full,recency,retrieval,force")
    ap.add_argument("--only", default="",
                    help="comma-separated fact keys and/or list_all")
    ap.add_argument("--skip-list", action="store_true",
                    help="do not run the long list-all question")
    ap.add_argument("--skip-locate", action="store_true")
    ap.add_argument("--locate-model", type=Path, default=None,
                    help="optional smaller GGUF for locate (same tokenizer)")
    args = ap.parse_args()

    models = args.models or [DEFAULT_MODEL]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cli = find_cli()
    if args.target_tokens > 0:
        if args.ctx < args.target_tokens + 256:
            args.ctx = args.target_tokens + 512
        measure = args.locate_model if args.locate_model and args.locate_model.is_file() else models[0]
        print(f"sizing filler to ~{args.target_tokens} tokens via {measure.name}", flush=True)
        args.gap, args.tail, para = size_gap_tail(
            cli, measure, args.target_tokens, args.budget, args.block_tokens,
            args.gen_reserve, args.ctx, args.batch,
        )
        print(f"  para~{para} tok  gap={args.gap} tail={args.tail} ctx={args.ctx}", flush=True)
    notes = build_notes(args.gap, args.tail)
    print(f"notes chars={len(notes)} gap={args.gap} tail={args.tail}", flush=True)

    rows: list[dict] = []

    for model in models:
        if not model.is_file():
            print(f"skip missing {model}")
            continue
        print(f"\n################ {model.name} ################", flush=True)
        only = {x.strip() for x in args.only.split(",") if x.strip()}
        located: dict[str, dict] = {}
        if not args.skip_locate:
            print("--- locate ---", flush=True)
            loc_model = args.locate_model if args.locate_model and args.locate_model.is_file() else model
            if loc_model != model:
                print(f"  locate model {loc_model.name}", flush=True)
            located = locate_facts(
                cli, loc_model, notes, args.budget, args.block_tokens,
                args.gen_reserve, args.ctx, args.batch, args.query_last,
                only=only or None,
            )

        cases: list[tuple[str, Fact | None, str, int]] = []
        for fact in FACTS:
            if only and fact.key not in only:
                continue
            cases.append((fact.key, fact, fact.question, args.n_predict))
        if not args.skip_list and (not only or "list_all" in only):
            cases.append(("list_all", None, LIST_QUESTION, args.n_predict_list))

        for case_key, fact, question, npred in cases:
            prompt = make_prompt(notes, question)
            for method in methods:
                force = fact.value if (method == "force" and fact is not None) else ""
                if method == "force" and fact is None:
                    continue
                budget = 0 if method == "full" else args.budget
                extra = kvmem_flags(
                    "retrieval" if method == "force" else method,
                    budget, args.block_tokens, args.gen_reserve,
                    args.ctx, args.batch, args.query_last, force,
                )
                tag = f"{model.name} {method:10} {case_key}"
                print(f"\n--- {tag} ---", flush=True)
                out, err = run_cli(cli, model, extra, prompt, npred)
                tr = parse_trace(err)
                keys = (
                    "KVMEM_TRACE retrieval",
                    "KVMEM_TRACE selected",
                    "stage_in_raw",
                    "query_replay",
                    "force_pos",
                    "n_prompt=",
                    "needle_block",
                    "Device 0:",
                )
                for ln in err.splitlines():
                    if any(k in ln for k in keys):
                        print("  ", ln, file=sys.stderr, flush=True)

                values = [f.value for f in FACTS]
                hits = [v for v in values if hit_value(out, v)]
                target_hit = hit_value(out, fact.value) if fact else None
                needle_block = tr["needle_block"]
                if needle_block is None and fact is not None and fact.key in located:
                    needle_block = located[fact.key].get("needle_block")
                selected = tr["selected"]
                ranker_hit = (
                    needle_block in selected if (needle_block is not None and selected) else None
                )
                loc = located.get(fact.key, {}) if fact else {}
                fpos = tr["force_pos"] if tr["force_pos"] is not None else loc.get("force_pos")
                npr = tr["n_prompt"] if tr["n_prompt"] is not None else loc.get("n_prompt")
                in_rec = recency_window_hit(npr, fpos, args.budget, args.block_tokens)

                row = {
                    "model": model.name,
                    "method": method,
                    "case": case_key,
                    "n_prompt": npr,
                    "pos": fpos,
                    "block": needle_block,
                    "in_recency": in_rec,
                    "ranker_hit": ranker_hit,
                    "gen_hit": target_hit,
                    "all_hits": hits,
                    "out": snippet(out),
                }
                rows.append(row)
                print(
                    f"  n_prompt={npr} pos={fpos} block={needle_block} "
                    f"in_recency={in_rec} ranker={ranker_hit} gen_hit={target_hit} "
                    f"codes_in_out={hits}",
                    flush=True,
                )
                print(f"  output: {snippet(out, 240)}", flush=True)

    print("\n========== SUMMARY ==========")
    hdr = (
        f"{'model':28} {'method':10} {'case':12} {'n':>6} {'pos':>5} {'blk':>4} "
        f"{'recW':5} {'rank':5} {'gen':5} hits"
    )
    print(hdr)
    for r in rows:
        def yn(v):
            if v is True:
                return "Y"
            if v is False:
                return "n"
            return "-"

        print(
            f"{r['model'][:28]:28} {r['method']:10} {r['case']:12} "
            f"{str(r['n_prompt'] or '-'):>6} {str(r['pos'] if r['pos'] is not None else '-'):>5} "
            f"{str(r['block'] if r['block'] is not None else '-'):>4} "
            f"{yn(r['in_recency']):5} {yn(r['ranker_hit']):5} {yn(r['gen_hit']):5} "
            f"{','.join(r['all_hits']) if r['all_hits'] else '-'}  | {r['out']}"
        )

    print(
        "\nLegend: recW=fact sits in recency window; rank=needle block in "
        "KVMEM_TRACE selected; gen=target string in generation. "
        "full = no eviction (capability ceiling). "
        "GO/NO-GO: retrieval miss is not an automatic stop."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
