import urllib.request
import json
import re

def get_chunk_for_filename(filename, is_train=True):
    # Try all chunks with HEAD request or check cache
    chunks = ["000", "001", "002", "003", "004"] if is_train else ["000", "001"]
    split = "UVH-26-Train" if is_train else "UVH-26-Val"
    for c in chunks:
        url = f"https://huggingface.co/datasets/iisc-aim/UVH-26/resolve/main/{split}/data/{c}/{filename}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return c
        except Exception:
            continue
    return None

def test():
    fnames = ["356278.png", "105615.png", "314443.png", "682193.png", "834214.png"]
    for fn in fnames:
        c = get_chunk_for_filename(fn, is_train=True)
        print(f"{fn} -> {c}")

if __name__ == "__main__":
    test()
