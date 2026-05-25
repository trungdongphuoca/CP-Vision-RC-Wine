"""check_generation_length.py — Kiểm tra độ dài generation và xem model sinh gì"""
import sys; sys.path.insert(0,'.')
import config as cfg, pandas as pd, re

df = pd.read_csv('evaluation/llm_eval_results.csv')
print(f'Total rows: {len(df)}')
print(f'ValidID rate: {df["ValidID"].mean()*100:.1f}%')
print()

# Analyze generated text lengths
gen_lens = df['generated'].str.len()
print(f'Generated text length stats:')
print(f'  Min: {gen_lens.min()}')
print(f'  Max: {gen_lens.max()}')
print(f'  Mean: {gen_lens.mean():.0f}')
print()

# Does any generated text contain </thought>?
has_end_thought = df['generated'].str.contains('</thought>', na=False)
print(f'Has </thought> tag: {has_end_thought.sum()}/{len(df)} = {has_end_thought.mean()*100:.1f}%')

# Does any generated text contain the ID pattern?
has_id_pattern = df['generated'].str.contains(r'\[[A-Z]{2,5}-[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-(?:\d{4}|NV)\]', na=False, regex=True)
print(f'Has [ID] pattern: {has_id_pattern.sum()}/{len(df)} = {has_id_pattern.mean()*100:.1f}%')

has_id_bare = df['generated'].str.contains(r'[A-Z]{2,5}-[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-(?:\d{4}|NV)', na=False, regex=True)
print(f'Has bare ID pattern: {has_id_bare.sum()}/{len(df)} = {has_id_bare.mean()*100:.1f}%')
print()

# Show some longer examples
print('=== LONGEST GENERATED TEXTS (see if ID appears) ===')
long_samples = df.nlargest(3, lambda x: x['generated'].str.len() if hasattr(x, 'str') else x)
for idx in df['generated'].str.len().nlargest(3).index:
    row = df.loc[idx]
    gen = str(row['generated'])
    print(f'Target: {row["target_id"]}')
    print(f'Length: {len(gen)}')
    print(f'Full text: {gen}')
    print()

print('=== SAMPLES WHERE ID WAS FOUND ===')
found = df[has_id_bare]
if len(found) > 0:
    for _, row in found.head(5).iterrows():
        print(f'Target: {row["target_id"]}')
        print(f'Pred:   {row["pred_id"]}')
        print(f'Gen:    {str(row["generated"])[:200]}')
        print()
else:
    print('No samples with valid ID pattern found!')
    print()
    print('=== TYPICAL GENERATION (first 5 samples in full) ===')
    for _, row in df.head(3).iterrows():
        print(f'Target: {row["target_id"]}')
        print(f'Full:   {str(row["generated"])}')
        print()
