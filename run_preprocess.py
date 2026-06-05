#!/usr/bin/env python3
"""
Launcher for preprocessing. Run from project directory:
  python run_preprocess.py --config preprocess/config.yaml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess.run_preprocess import main
if __name__ == "__main__":
    main()
