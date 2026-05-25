import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1]))
import config as cfg
import json
import numpy as np
import pandas as pd

RANDOM_STATE = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def create_prompt(row):
    if pd.notna(row.get("price")):
        instruction = f"Recommend a {row['variety']} from {row['country']} that costs around ${row['price']}."
    else:
        instruction = f"Recommend a {row['variety']} from {row['country']} with a flexible budget."
    
    thought = {
        "user_analysis": {
            "grape_preference": row['variety'],
            "region_preference": row['country'],
            "budget": f"${row['price']}" if not pd.isna(row['price']) else "Unknown"
        },
        "target_wine_profile": {
            "notes": "Extracted from description matching user intent."
        },
        "candidate": f"{row.get('title', 'Unknown Wine')}",
        "selected_id": row['Semantic_ID']
    }
    
    response = f"I suggest the [{row['Semantic_ID']}]. {row['description']}"
    
    return {
        "instruction": instruction,
        "thought": json.dumps(thought),
        "response": response,
        "target_id": row['Semantic_ID']
    }

def split_by_semantic_id(
    df,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    random_state=RANDOM_STATE,
):
    rng = np.random.default_rng(random_state)
    group_sizes = df.groupby("Semantic_ID", sort=False).size().reset_index(name="n")
    shuffled = group_sizes.sample(frac=1, random_state=random_state).reset_index(drop=True)

    total_rows = len(df)
    train_target = total_rows * train_ratio
    val_target = total_rows * val_ratio

    train_ids, val_ids, test_ids = set(), set(), set()
    counts = {"train": 0, "val": 0, "test": 0}

    shuffled["_tie_break"] = rng.random(len(shuffled))
    shuffled = shuffled.sort_values(["n", "_tie_break"], ascending=[False, True])

    for row in shuffled.itertuples(index=False):
        semantic_id = row.Semantic_ID
        size = int(row.n)

        deficits = {
            "train": train_target - counts["train"],
            "val": val_target - counts["val"],
            "test": (total_rows - train_target - val_target) - counts["test"],
        }
        split = max(deficits, key=deficits.get)
        if split == "train":
            train_ids.add(semantic_id)
        elif split == "val":
            val_ids.add(semantic_id)
        else:
            test_ids.add(semantic_id)
        counts[split] += size

    train_df = df[df["Semantic_ID"].isin(train_ids)].sample(frac=1, random_state=random_state)
    val_df = df[df["Semantic_ID"].isin(val_ids)].sample(frac=1, random_state=random_state)
    test_df = df[df["Semantic_ID"].isin(test_ids)].sample(frac=1, random_state=random_state)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def main():
    cfg.ensure_dirs()
    csv_path = cfg.WINE_SEMANTIC_CSV
    train_path = cfg.TRAIN_JSONL
    val_path = cfg.VAL_JSONL
    test_path = cfg.TEST_JSONL
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run build_semantic_ids.py first.")
        return
        
    print("Loading semantic ID data...")
    df = pd.read_csv(csv_path)
    
    print("Generating prompts... (this may take a few seconds)")
    train_df, val_df, test_df = split_by_semantic_id(df)

    train_records = [create_prompt(row) for _, row in train_df.iterrows()]
    val_records = [create_prompt(row) for _, row in val_df.iterrows()]
    test_records = [create_prompt(row) for _, row in test_df.iterrows()]

    print(
        "Group split complete: "
        f"train={len(train_records):,} rows / {train_df['Semantic_ID'].nunique():,} IDs, "
        f"val={len(val_records):,} rows / {val_df['Semantic_ID'].nunique():,} IDs, "
        f"test={len(test_records):,} rows / {test_df['Semantic_ID'].nunique():,} IDs"
    )
        
    print(f"Saving {len(train_records)} records to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for record in train_records:
            f.write(json.dumps(record) + '\n')
            
    print(f"Saving {len(val_records)} records to {val_path}...")
    with open(val_path, 'w', encoding='utf-8') as f:
        for record in val_records:
            f.write(json.dumps(record) + '\n')
            
    print(f"Saving {len(test_records)} records to {test_path}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        for record in test_records:
            f.write(json.dumps(record) + '\n')
            
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
