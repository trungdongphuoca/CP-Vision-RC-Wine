import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import config as cfg
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# Import helpers from baseline_eval
from evaluation.baseline_eval import load_catalog, load_test, K_VALUES, run_gnn_retrieval

def main():
    # Load catalog and test data
    cat = load_catalog(str(cfg.WINE_CSV))
    test = load_test(str(cfg.TEST_JSONL), 1000)

    # Run GNN retrieval
    _, s = run_gnn_retrieval(cat, test, K_VALUES)

    # Update baseline_comparison.csv
    baseline_path = str(cfg.BASELINE_CSV)
    if os.path.exists(baseline_path):
        df = pd.read_csv(baseline_path, index_col=0)
        
        # Ensure all columns in s exist in the df index
        for k, v in s.items():
            df.loc["GNN-Filter", k] = v
            
        df.to_csv(baseline_path)
        print("\nUpdated baseline_comparison.csv with GNN-Filter metrics:")
        print(df.loc["GNN-Filter"])
    else:
        print(f"Error: {baseline_path} not found.")

if __name__ == "__main__":
    main()
