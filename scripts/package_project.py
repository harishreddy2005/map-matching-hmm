"""Create a cache-free offline project archive using explicit content roots."""
from pathlib import Path
import zipfile
import hashlib

ROOT=Path(__file__).resolve().parents[1]

def main():
    destination=ROOT/'submission'
    destination.mkdir(exist_ok=True)
    archive=destination/'map_matching_project.zip'
    roots=['src','scripts','tests','docs','results','data']
    files=[ROOT/name for name in ['README.md','requirements.txt','requirements-lock.txt','.gitignore','run_demo.py','team.json']]
    for name in roots:
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ['.pkl','.pyc','.zip'])
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for path in sorted(files):
            out.write(path,str(path.relative_to(ROOT)))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (destination/'SHA256.txt').write_text(digest+'  '+archive.name+'\n')
    print(archive, 'bytes:',archive.stat().st_size,'sha256:',digest)

if __name__=='__main__':main()
