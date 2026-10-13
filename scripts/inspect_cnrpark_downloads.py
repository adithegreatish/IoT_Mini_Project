import ssl
import urllib.request
import re

ctx = ssl._create_unverified_context()
req = urllib.request.Request("https://cnrpark.it/", headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
        html = r.read().decode("utf-8", errors="ignore")
        print(f"Fetched cnrpark.it homepage ({len(html)} chars)")
        links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)
        for l in links:
            if any(k in l.lower() for k in ["zip", "tar", "patch", "download", "dataset", "dataset_cnr"]):
                print("Link:", l)
except Exception as e:
    print("Error:", e)
