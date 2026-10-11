"""Dense map matching without shortest-path search."""
import os
from pathlib import Path

# Keep plotting caches inside the project on restricted Windows machines.
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.mplconfig'))
