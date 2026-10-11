"""Verify archive contents and run the demo in a cache-free extracted copy."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'verification'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True)
    archive=ROOT/'submission/map_matching_project.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'Corrupt archive'
        assert not any('.venv/' in name or name.endswith('.pkl') for name in z.namelist())
        z.extractall(folder)
    steps=[['-m','unittest','discover','-s','tests','-v'],
           ['run_demo.py','--T','10','--N','7','--slice','5300','5600'],
           ['scripts/build_deliverables.py']]
    logs=[]
    for args in steps:
        p=subprocess.run([sys.executable,*args],cwd=folder,capture_output=True,text=True,encoding='utf-8')
        logs.append({'command':['python',*args],'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        print(' '.join(args), 'exit:', p.returncode, flush=True)
        if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    from pypdf import PdfReader
    from pptx import Presentation
    pages={name:len(PdfReader(folder/'docs'/name).pages) for name in ['writeup.pdf','writeup_one_page.pdf','presentation.pdf']}
    assert pages=={'writeup.pdf':2,'writeup_one_page.pdf':1,'presentation.pdf':12},pages
    assert len(Presentation(folder/'docs/presentation.pptx').slides)==12
    sys.path.insert(0,str(folder))
    from src.pipeline import prepare
    import numpy as np
    data,candidates,distances,fallback,em,discrepancy=prepare()
    assert len(data.xy)==7531 and len(data.lines)==306182
    assert np.all(np.diff(data.route_position)>=0)
    assert set(data.truth).issubset(set(data.route))
    assert all(len(c)>0 for c in candidates)
    assert all(np.isfinite(f).all() for f in em)
    assert (folder/'results/demo_map.html').stat().st_size>10000
    report={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'isolated_workdir':str(folder),
            'checks':'7 unit tests; cache-free offline demo; document generation; page and slide counts; dataset/labels/features sanity',
            'pdf_pages':pages,'slides':12,'steps':logs,
            'dependency_note':'Isolated source/data copy uses the installed project interpreter; no independent OS/machine test.',
            'route_alignment_start_end':[int(data.route_position[0]),int(data.route_position[-1])]}
    (ROOT/'docs/verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='steps'},indent=2))

if __name__=='__main__':main()
