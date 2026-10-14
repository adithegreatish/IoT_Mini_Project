import json
import os
from pathlib import Path
import time
import urllib.request

def build_uvh26_index():
    index_cache = Path("data/raw/uvh26/file_index.json")
    if index_cache.exists():
        print(f"Using cached index: {index_cache}")
        with open(index_cache, "r", encoding="utf-8") as f:
            return json.load(f)
    
    os.makedirs(index_cache.parent, exist_ok=True)
    mapping = {} # filename -> relative path within repo
    
    folders = [
        ("UVH-26-Train/data/000", "UVH-26-Train"),
        ("UVH-26-Train/data/001", "UVH-26-Train"),
        ("UVH-26-Train/data/002", "UVH-26-Train"),
        ("UVH-26-Train/data/003", "UVH-26-Train"),
        ("UVH-26-Train/data/004", "UVH-26-Train"),
        ("UVH-26-Val/data/000", "UVH-26-Val"),
        ("UVH-26-Val/data/001", "UVH-26-Val"),
    ]
    
    t0 = time.time()
    for folder, split in folders:
        url = f"https://huggingface.co/api/datasets/iisc-aim/UVH-26/tree/main/{folder}?limit=1000"
        count = 0
        while url:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                for item in data:
                    fname = item["path"].split("/")[-1]
                    mapping[fname] = item["path"]
                    count += 1
                link = resp.headers.get("Link")
                url = None
                if link and "next" in link:
                    for part in link.split(","):
                        if "next" in part:
                            url = part.split(";")[0].strip("<> ")
        print(f"Indexed {folder}: {count} files")
        
    print(f"Total files indexed: {len(mapping)} in {time.time()-t0:.2f}s")
    with open(index_cache, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return mapping

if __name__ == "__main__":
    idx = build_uvh26_index()
    print("Sample lookup:")
    for test_k in ["100103.png", "356278.png", "100130.png"]:
        print(f"  {test_k} -> {idx.get(test_k)}")
