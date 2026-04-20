"""
Download EGFR (CHEMBL203) IC50 bioactivity data from ChEMBL REST API.
Uses direct HTTPS calls with explicit pagination + progress output.
"""
import sys
import time
import requests
import pandas as pd

BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
PARAMS = {
    "target_chembl_id": "CHEMBL203",
    "standard_type":    "IC50",
    "limit":            1000,
}

def main():
    all_records = []
    url = BASE
    params = PARAMS
    page = 0

    while url:
        page += 1
        t0 = time.time()
        r = requests.get(url, params=params if page == 1 else None, timeout=60)
        r.raise_for_status()
        data = r.json()

        activities = data.get("activities", [])
        all_records.extend(activities)
        elapsed = time.time() - t0

        meta = data.get("page_meta", {})
        total = meta.get("total_count", "?")
        print(f"  page {page:>3}: +{len(activities):>4} records  "
              f"(running total: {len(all_records)}/{total})  "
              f"[{elapsed:.1f}s]",
              flush=True)

        next_path = meta.get("next")
        url = f"https://www.ebi.ac.uk{next_path}" if next_path else None
        params = None  # next URL already encodes them

    print(f"\nTotal records fetched: {len(all_records)}")

    # Keep only the columns we need
    keep = ["molecule_chembl_id", "canonical_smiles", "standard_type",
            "standard_value", "standard_units", "assay_type"]
    df = pd.DataFrame(all_records)
    df = df[[c for c in keep if c in df.columns]]
    print(f"Final shape: {df.shape}")

    df.to_csv("data/egfr_chembl.csv", index=False)
    print("Saved -> data/egfr_chembl.csv")


if __name__ == "__main__":
    print("Fetching EGFR (CHEMBL203) IC50 activities from ChEMBL REST API...")
    sys.stdout.flush()
    main()
