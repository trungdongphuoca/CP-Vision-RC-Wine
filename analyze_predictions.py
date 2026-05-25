"""analyze_predictions.py"""
import sys; sys.path.insert(0,'.')
import config as cfg, json, pandas as pd

df = pd.read_csv('evaluation/llm_eval_results.csv')
print('=== LLM EVAL RESULTS STATS ===')
print(f'Total: {len(df)}')
print(f'Columns: {list(df.columns)}')
print()

# Sample predictions
print('=== SAMPLE PREDICTIONS (10 examples) ===')
for _, row in df.head(10).iterrows():
    print(f'  Target:    {row["target_id"]}')
    print(f'  Predicted: {row["pred_id"]}')
    gen = str(row.get("generated",""))[:80]
    print(f'  Generated: {gen}')
    print()

# Per-field accuracy
def split_id(x):
    parts = str(x).split('-')
    return parts if len(parts)==4 else [None,None,None,None]

pred_parts = df['pred_id'].apply(split_id)
tgt_parts  = df['target_id'].apply(split_id)

country_ok  = sum(p[0]==t[0] for p,t in zip(pred_parts, tgt_parts))
province_ok = sum(p[1]==t[1] for p,t in zip(pred_parts, tgt_parts))
variety_ok  = sum(p[2]==t[2] for p,t in zip(pred_parts, tgt_parts))
year_ok     = sum(p[3]==t[3] for p,t in zip(pred_parts, tgt_parts))
n = len(df)

print('=== PER-FIELD ACCURACY ===')
print(f'COUNTRY  correct: {country_ok}/{n} = {country_ok/n*100:.1f}%')
print(f'PROVINCE correct: {province_ok}/{n} = {province_ok/n*100:.1f}%')
print(f'VARIETY  correct: {variety_ok}/{n} = {variety_ok/n*100:.1f}%')
print(f'YEAR     correct: {year_ok}/{n} = {year_ok/n*100:.1f}%')
print(f'ALL correct (EM): {df["ExactMatch"].mean()*100:.2f}%')
if 'IntentMatch@1' in df.columns:
    print(f'Intent (C+V):     {df["IntentMatch@1"].mean()*100:.2f}%')
print()

print('=== VALID ID RATE ===')
print(f'Valid 4-part ID: {df["ValidID"].mean()*100:.1f}%')

# Year distribution - what years does model predict?
pred_years = [p[3] for p in pred_parts if p[3] is not None]
tgt_years  = [t[3] for t in tgt_parts  if t[3] is not None]
print()
print(f'=== YEAR ANALYSIS ===')
pred_year_series = pd.Series(pred_years).value_counts().head(10)
tgt_year_series  = pd.Series(tgt_years).value_counts().head(10)
print('Predicted year distribution (top 10):')
print(pred_year_series.to_string())
print('Target year distribution (top 10):')
print(tgt_year_series.to_string())
