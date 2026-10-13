import ssl
import urllib.request
import re

ctx = ssl._create_unverified_context()

def check_url(url, name):
    print(f"=== Checking {name} ({url}) ===")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
            html = r.read().decode("utf-8", errors="ignore")
            print(f"Fetched {len(html)} chars")
            links = re.findall(r'href=["\']([^"\']+)["\']', html)
            for l in links:
                if any(k in l.lower() for k in ["pklot", "tar", "zip", "xml", "inf.ufpr", "download", "drive", "dropbox", "box"]):
                    print("  Link:", l)
    except Exception as e:
        print(f"  Error accessing {name}: {e}")

check_url("https://web.inf.ufpr.br/vri/databases/parking-lot-database/", "UFPR PKLot")
check_url("https://github.com/DSBD-Research/MetaPKLot-Dataset", "MetaPKLot GitHub")
