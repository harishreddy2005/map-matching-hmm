"""Download the original Microsoft Research benchmark, without dependencies."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/'

def main():
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    manifest = {}
    for name in ['gps_data.txt', 'road_network.zip', 'ground_truth_route.txt']:
        target = data / name
        if not target.exists():
            print('Downloading', name, flush=True)
            subprocess.run(['curl', '-L', '--fail', BASE + name + '?download=1',
                            '-o', str(target)], check=True)
        manifest[name] = {'url': BASE + name, 'bytes': target.stat().st_size,
                          'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    with zipfile.ZipFile(data / 'road_network.zip') as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith('road_network.txt'))
        (data / 'road_network.txt').write_bytes(archive.read(member))
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Benchmark ready in', data)

if __name__ == '__main__':
    main()
