"""diagnose_recall.py — Phân tích nguyên nhân Recall thấp"""
import sys; sys.path.insert(0, '.')
import config as cfg, json, pandas as pd, re

# Load
test = [json.loads(l) for l in open(str(cfg.TEST_JSONL))]
df = pd.read_csv(str(cfg.WINE_CSV)).dropna(subset=['country','variety','description','title'])

def extract_year(t):
    m = re.search(r'(19|20)\d{2}', str(t))
    return m.group(0) if m else 'NV'
def clean_text(t):
    return re.sub(r'[^A-Za-z0-9]','',str(t)).upper()[:4]

df['vintage'] = df['title'].apply(extract_year)
df['Semantic_ID'] = df.apply(
    lambda r: f"{clean_text(r['country'])}-{clean_text(r.get('province',''))}-{clean_text(r['variety'])}-{r['vintage']}", axis=1)

catalog_ids = set(df['Semantic_ID'].tolist())
test_targets = [q['target_id'] for q in test]

# 1. Coverage
found = sum(1 for t in test_targets if t in catalog_ids)
print(f"[1] Target IDs found in catalog: {found}/{len(test_targets)} = {found/len(test_targets)*100:.1f}%")

# 2. Uniqueness / collision
id_counts = df['Semantic_ID'].value_counts()
multi = id_counts[id_counts > 1]
print(f"\n[2] Semantic ID uniqueness:")
print(f"    Unique IDs in catalog: {len(id_counts):,}")
print(f"    IDs mapping to >1 wine: {len(multi):,} ({len(multi)/len(id_counts)*100:.1f}% of unique IDs)")
print(f"    Max duplicates for one ID: {id_counts.max()}")
print(f"    IDs with 10+ duplicates: {(id_counts >= 10).sum():,}")

# 3. Ambiguity - how many test targets have multiple wines with same ID
test_id_counts = pd.Series(test_targets).map(id_counts.to_dict()).fillna(0)
dup_targets = (test_id_counts > 1).sum()
print(f"\n[3] Test target ambiguity:")
print(f"    Test targets mapping to >1 wine: {dup_targets}/{len(test_targets)} = {dup_targets/len(test_targets)*100:.1f}%")
print(f"    Distribution (top 8):")
print(test_id_counts.value_counts().sort_index().head(8).to_string())

# 4. What is the query asking vs what's in Semantic ID?
print("\n[4] Semantic ID structure analysis (sample):")
print("    Format: COUNTRY(4)-PROVINCE(4)-VARIETY(4)-YEAR(4)")
print("    Query only mentions: country + variety + price (sometimes)")
print("    Ambiguous parts: PROVINCE, YEAR -> model must guess these!")
sample_targets = test_targets[:5]
for t in sample_targets:
    parts = t.split('-')
    print(f"    {t} -> {parts}")

# 5. Province ambiguity: how many provinces per country-variety combo?
df['country4'] = df['country'].apply(clean_text)
df['variety4'] = df['variety'].apply(clean_text)
df['province4'] = df.get('province', pd.Series(['UNKN']*len(df))).apply(clean_text)
cv_provinces = df.groupby(['country4','variety4'])['province4'].nunique()
print(f"\n[5] Province ambiguity per (Country, Variety) pair:")
print(f"    Avg provinces per CV pair: {cv_provinces.mean():.1f}")
print(f"    Max provinces per CV pair: {cv_provinces.max()}")
print(f"    CV pairs with >5 provinces: {(cv_provinces > 5).sum()}")

# 6. Year ambiguity
cv_years = df.groupby(['country4','variety4'])['vintage'].nunique()
print(f"\n[6] Year ambiguity per (Country, Variety) pair:")
print(f"    Avg years per CV pair: {cv_years.mean():.1f}")
print(f"    Max years per CV pair: {cv_years.max()}")
print(f"    CV pairs with >5 years: {(cv_years > 5).sum()}")

# 7. How much info from query can reduce search space?
# Best possible Recall@1 if we perfectly match country+variety but random on province+year
print("\n[7] Theoretical best Recall@1 (if model perfectly knows country+variety):")
total = 0
for q in test[:500]:
    tid = q['target_id']
    parts = tid.split('-')
    if len(parts) < 4: continue
    # Wines matching same country+variety
    same_cv = df[(df['country4']==parts[0]) & (df['variety4']==parts[2])]
    total += 1.0 / len(same_cv) if len(same_cv) > 0 else 0
print(f"    E[Recall@1 | correct country+variety] = {total/500:.4f} ({total/500*100:.2f}%)")
print("    (This is the CEILING given ambiguity in province+year)")

print("\nDone.")
