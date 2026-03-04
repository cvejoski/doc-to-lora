import gc
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm

# Split layout (indices into the original train split):
#   validation : [0,     1_000)   → 1 000 samples
#   test       : [1_000, 3_000)   → 2 000 samples
#   train      : [3_000, end)     → ~73 840 samples
VAL_END = 1_000
TEST_END = 3_000

if __name__ == "__main__":
    print("Loading vm2825/CodeQA-dataset …")
    full_ds = load_dataset("vm2825/CodeQA-dataset", split="train")
    print(f"Total samples: {len(full_ds)}")

    raw_splits = {
        "validation": full_ds.select(range(0, VAL_END)),
        "test": full_ds.select(range(VAL_END, TEST_END)),
        "train": full_ds.select(range(TEST_END, len(full_ds))),
    }

    for split, ds in raw_splits.items():
        print(f"\n--- {split} ({len(ds)} samples) ---")
        ctx_qa_dict: dict[str, dict] = {}
        for sample in tqdm(ds):
            ctx = sample["input_code"]
            if ctx not in ctx_qa_dict:
                ctx_qa_dict[ctx] = {"prompts": [], "responses": []}
            ctx_qa_dict[ctx]["prompts"].append(sample["Instruction"].strip())
            ctx_qa_dict[ctx]["responses"].append(sample["output_code"].strip())

        print(f"Unique code contexts: {len(ctx_qa_dict)}")
        samples = [
            {
                "context": ctx,
                "prompts": v["prompts"],
                "responses": v["responses"],
            }
            for ctx, v in ctx_qa_dict.items()
        ]
        print(f"Example: {samples[0]}")

        out_ds = Dataset.from_list(samples)
        save_path = f"./data/raw_datasets/codeqa_compact/{split}/ds.parquet"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving → {save_path}")
        out_ds.to_parquet(save_path)
        print("=" * 80)
        del out_ds, samples, ctx_qa_dict
        gc.collect()

    print("Done.")
