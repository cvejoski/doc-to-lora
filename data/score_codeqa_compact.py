"""Score codeqa_compact responses with a base model to generate logprobs for KL-loss training.

Produces a self_gen-format dataset at:
  data/raw_datasets/self_gen/<vllm_model>/codeqa_compact/train/ds_XXXX.parquet

Each output sample has the same schema as self_generate_qa.py output:
  ctx_ids, input_ids, response_start_end, logprobs_vals, logprobs_indices

Usage:
  uv run python data/score_codeqa_compact.py --vllm_model google/gemma-2-2b-it
"""

import argparse
import gc
import os

import numpy as np
from datasets import Dataset, load_dataset

from ctx_to_lora.data.definitions import RAW_DATA_DIR, SELF_GEN_DATA_DIR
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.utils import clear_gpu

MODEL_CTX_LEN = {
    "google/gemma-2-2b-it": 8192,
    "google/gemma-2-9b-it": 8192,
    "google/gemma-2-27b-it": 8192,
}

TOP_K = 16
CHUNK_SIZE = 500


def tokenize_qa_pair(tk, prompt: str, response: str):
    """Tokenize a single prompt+response chat turn; return input_ids and (resp_start, resp_end)."""
    messages = [
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": response.strip()},
    ]
    tokens = tk.apply_chat_template(
        messages,
        tokenize=True,
        add_special_tokens=False,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    input_ids = tokens["input_ids"]
    assistant_masks = tokens["assistant_masks"]

    resp_start = None
    resp_end = None
    for i, mask in enumerate(assistant_masks):
        if mask and resp_start is None:
            resp_start = i
        if mask:
            resp_end = i + 1

    if resp_start is None:
        return None, None

    # Truncate at resp_end (matching self_gen convention; avoids trailing special tokens)
    return input_ids[:resp_end], (resp_start, resp_end)


def tokenize_contexts(tk, contexts: list[str]) -> list[list[int]]:
    """Tokenize contexts in the same format as tokenize_ctx_text (with generation prompt)."""
    return tk.apply_chat_template(
        [
            [{"role": "system", "content": ""}, {"role": "user", "content": ctx.strip()}]
            for ctx in contexts
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_attention_mask=False,
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_dict=True,
    )["input_ids"]


def score_chunk(llm, tk, ctxs, prompts_list, responses_list, chunk_idx: int, out_dir: str):
    from vllm import SamplingParams

    # Flatten QA pairs while tracking which context each belongs to
    flat_items = []  # (ctx_idx, input_ids, (resp_start, resp_end))
    n_skipped = 0

    for ctx_idx, (prompts, responses) in enumerate(zip(prompts_list, responses_list)):
        for prompt, response in zip(prompts, responses):
            input_ids, rs_re = tokenize_qa_pair(tk, prompt, response)
            if input_ids is None:
                n_skipped += 1
                continue
            flat_items.append((ctx_idx, input_ids, rs_re))

    print(f"  {len(flat_items)} QA pairs ({n_skipped} skipped) from {len(ctxs)} contexts")

    if not flat_items:
        print("  No valid QA pairs, skipping chunk.")
        return

    # Score with vLLM: pass full sequence as prompt, use prompt_logprobs
    clear_gpu()
    vllm_inputs = [{"prompt_token_ids": ids} for _, ids, _ in flat_items]

    completions = llm.generate(
        vllm_inputs,
        sampling_params=SamplingParams(
            max_tokens=1,  # generate 1 dummy token to satisfy vLLM; we only use prompt_logprobs
            prompt_logprobs=TOP_K,
            temperature=0.0,
            seed=42,
        ),
    )

    # Tokenize contexts for ctx_ids
    ctx_ids_list = tokenize_contexts(tk, ctxs)

    # Accumulate per context
    per_ctx: dict[int, dict] = {
        i: {
            "ctx_ids": ctx_ids_list[i],
            "input_ids": [],
            "response_start_end": [],
            "logprobs_vals": [],
            "logprobs_indices": [],
        }
        for i in range(len(ctxs))
    }

    n_invalid_logprobs = 0
    for (ctx_idx, input_ids, (resp_start, resp_end)), completion in zip(flat_items, completions):
        plogprobs = completion.prompt_logprobs
        n_response_tokens = resp_end - resp_start

        vals = np.empty((n_response_tokens, TOP_K), dtype=np.float16)
        indices = np.empty((n_response_tokens, TOP_K), dtype=np.int32)

        ok = True
        for ti, pos in enumerate(range(resp_start, resp_end)):
            logp_dict = plogprobs[pos] if plogprobs is not None and pos < len(plogprobs) else None
            if logp_dict is None:
                ok = False
                break
            for j, (idx, tok_info) in enumerate(logp_dict.items()):
                if j >= TOP_K:
                    break
                indices[ti, j] = idx
                vals[ti, j] = tok_info.logprob

        if not ok:
            n_invalid_logprobs += 1
            continue

        per_ctx[ctx_idx]["input_ids"].append(input_ids)
        per_ctx[ctx_idx]["response_start_end"].append((resp_start, resp_end))
        per_ctx[ctx_idx]["logprobs_vals"].append(vals)
        per_ctx[ctx_idx]["logprobs_indices"].append(indices)

    if n_invalid_logprobs:
        print(f"  {n_invalid_logprobs} QA pairs had invalid logprobs and were skipped")

    samples = [v for v in per_ctx.values() if v["input_ids"]]
    print(f"  Saving {len(samples)} contexts with valid logprobs")

    out_ds = Dataset.from_list(samples)
    fpath = os.path.join(out_dir, f"ds_{chunk_idx:04d}.parquet")
    out_ds.to_parquet(fpath)
    print(f"  Saved → {fpath}")

    del completions, out_ds, samples
    clear_gpu()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Score codeqa_compact with base model logprobs")
    parser.add_argument("--vllm_model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--debug", action="store_true", help="Process only 5 samples")
    args = parser.parse_args()

    tk = get_tokenizer(args.vllm_model, train=True)

    from vllm import LLM

    llm = LLM(
        model=args.vllm_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=MODEL_CTX_LEN.get(args.vllm_model, 8192),
        max_num_batched_tokens=16384,
        max_num_seqs=32,
    )

    ds = load_dataset(
        path="parquet",
        data_files=[f"{RAW_DATA_DIR}/codeqa_compact/train/ds.parquet"],
        split="train",
    )

    if args.debug:
        ds = ds.take(5)
        print("Debug mode: processing 5 samples only")

    print(f"Loaded {len(ds)} codeqa_compact train samples")

    out_dir = f"{SELF_GEN_DATA_DIR}/{args.vllm_model}/codeqa_compact/train"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    for chunk_idx, start in enumerate(range(0, len(ds), CHUNK_SIZE)):
        chunk = ds[start : start + CHUNK_SIZE]
        print(f"\nChunk {chunk_idx} (samples {start}–{start + len(chunk['context']) - 1})")
        score_chunk(
            llm=llm,
            tk=tk,
            ctxs=chunk["context"],
            prompts_list=chunk["prompts"],
            responses_list=chunk["responses"],
            chunk_idx=chunk_idx,
            out_dir=out_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
