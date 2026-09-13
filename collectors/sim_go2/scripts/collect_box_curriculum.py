#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "collection_box_curriculum.yaml"
REPO_ROOT = PACKAGE_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    sys.argv = [str(SCRIPT_DIR / "collect_raw_trajectories.py"), "--config", str(DEFAULT_CONFIG), *sys.argv[1:]]
    from collectors.sim_go2.scripts.collect_raw_trajectories import main
    main()
