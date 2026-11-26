"""Verify archived content without loading or executing models."""
from pathlib import Path
import hashlib, json, sys
ROOT=Path(__file__).resolve().parent.parent

def main():
    checks=json.loads((ROOT/'FILE_MANIFEST.json').read_text())
    missing=[];bad=[]
    for path,info in checks['files'].items():
        p=ROOT/path
        if not p.is_file():missing.append(path);continue
        if hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256'] or p.stat().st_size!=info['bytes']:
            bad.append(path)
    print(json.dumps({'verified_files':len(checks['files']),'missing':missing,'mismatched':bad,'ok':not missing and not bad},indent=2))
    assert not missing and not bad
if __name__=='__main__':main()
