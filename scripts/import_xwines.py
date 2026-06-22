import os
import urllib.request
import zipfile
import ast
import pandas as pd
from pathlib import Path

def download_data():
    dest_dir = Path("data/xwines")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    urls = {
        "XWines_Test_100_wines.csv": "https://raw.githubusercontent.com/rogerioxavier/X-Wines/main/Dataset/last/XWines_Test_100_wines.csv",
        "XWines_Test_1K_ratings.csv": "https://raw.githubusercontent.com/rogerioxavier/X-Wines/main/Dataset/last/XWines_Test_1K_ratings.csv",
        "XWines_Test_100_labels.zip": "https://raw.githubusercontent.com/rogerioxavier/X-Wines/main/Dataset/last/XWines_Test_100_labels.zip"
    }
    
    # 1. Download files
    for filename, url in urls.items():
        filepath = dest_dir / filename
        if filepath.exists():
            print(f"[X-Wines] {filename} already exists. Skipping download.")
            continue
            
        print(f"[X-Wines] Downloading {filename} from {url}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[X-Wines] Successfully downloaded {filename}.")
        except Exception as e:
            print(f"[X-Wines] Failed to download {filename}: {e}")
            
    # 2. Extract labels zip file
    zip_path = dest_dir / "XWines_Test_100_labels.zip"
    extract_dir = dest_dir / "images"
    if zip_path.exists() and not extract_dir.exists():
        print(f"[X-Wines] Extracting {zip_path.name} to {extract_dir}...")
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("[X-Wines] Extraction complete.")
        except Exception as e:
            print(f"[X-Wines] Failed to extract zip: {e}")
    else:
        print("[X-Wines] Extraction skipped (zip missing or target directory already exists).")

def process_data():
    print("[X-Wines] Starting preprocessing and textualization...")
    wines_path = Path("data/xwines/XWines_Test_100_wines.csv")
    ratings_path = Path("data/xwines/XWines_Test_1K_ratings.csv")
    
    if not wines_path.exists() or not ratings_path.exists():
        print("[X-Wines] Source CSV files not found. Cannot process.")
        return
        
    wines_df = pd.read_csv(wines_path)
    ratings_df = pd.read_csv(ratings_path)
    
    # 1. Textualize wines
    processed_wines = []
    sapo_catalog_rows = []
    
    for idx, row in wines_df.iterrows():
        # Parse lists safely
        try:
            grapes_list = ast.literal_eval(row['Grapes'])
            grapes_str = ", ".join(grapes_list)
            first_grape = grapes_list[0] if grapes_list else "Unknown"
        except:
            grapes_str = str(row['Grapes'])
            first_grape = "Unknown"
            
        try:
            harmonize_list = ast.literal_eval(row['Harmonize'])
            harmonize_str = ", ".join(harmonize_list)
        except:
            harmonize_str = str(row['Harmonize'])
            
        # Build textual description (No price)
        desc = (
            f"This is a {row['Body']} bodied {row['Type']} wine produced by {row['WineryName']} "
            f"in the region of {row['RegionName']}, {row['Country']}. "
            f"It has {row['Acidity']} acidity, an ABV of {row['ABV']}%, and is elaborated as a {row['Elaborate']}. "
            f"It is made from {grapes_str} grapes. "
            f"This wine harmonizes beautifully with {harmonize_str}."
        )
        
        # doc_text is used for TF-IDF search
        doc_text = f"{row['WineName']} {grapes_str} {row['Country']} {row['RegionName']} {row['WineryName']} {desc}"
        
        wine_id = str(row['WineID'])
        
        processed_wines.append({
            "title": row['WineName'],
            "variety": first_grape,
            "country": row['Country'],
            "price": None, # Price removed
            "points": 90, # default points
            "description": desc,
            "winery": row['WineryName'],
            "province": row['RegionName'],
            "doc_text": doc_text,
            "Semantic_ID_Cluster": "", # Will be filled by data_prep.py
            "Item_Index": idx,
            "Semantic_ID": "" # Will be filled by data_prep.py
        })
        
        # sapo_catalog entry mapping
        sapo_catalog_rows.append({
            "sku": wine_id,
            "name": row['WineName'],
            "type": row['Type'],
            "brand": row['WineryName'],
            "tags": f"{grapes_str}, {harmonize_str}",
            "price": None,
            "price_compare": None,
            "description": desc,
            "full_text": desc,
            "item_idx": idx
        })
        
    # Export processed semantic catalog
    proc_wines_df = pd.DataFrame(processed_wines)
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a temporary copy (without clusters)
    proc_wines_df.to_csv(proc_dir / "wine_catalog_semantic.csv", index=False)
    
    # Overwrite the raw catalog file so build_semantic_ids.py runs on it
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_wines_df.to_csv(raw_dir / "winemag-data-130k-v2.csv", index=False)
    print(f"[X-Wines] Exported {len(proc_wines_df)} wines to data/raw/winemag-data-130k-v2.csv")
    
    # Export Sapo Catalog
    sapo_dir = Path("data/sapo")
    sapo_dir.mkdir(parents=True, exist_ok=True)
    sapo_cat_df = pd.DataFrame(sapo_catalog_rows)
    sapo_cat_df.to_csv(sapo_dir / "sapo_catalog.csv", index=False)
    print(f"[X-Wines] Exported Sapo catalog to data/sapo/sapo_catalog.csv")
    
    # 2. Export Sapo Interactions from X-Wines ratings
    # Sapo interactions: user, sku, qty
    sapo_inter = pd.DataFrame({
        "user": ratings_df['UserID'].astype(str),
        "sku": ratings_df['WineID'].astype(str),
        "qty": ratings_df['Rating']
    })
    sapo_inter.to_csv(sapo_dir / "sapo_interactions.csv", index=False)
    print(f"[X-Wines] Exported {len(sapo_inter)} interactions to data/sapo/sapo_interactions.csv")
    print("[X-Wines] Preprocessing and textualization complete.")

def main():
    download_data()
    process_data()

if __name__ == "__main__":
    main()
