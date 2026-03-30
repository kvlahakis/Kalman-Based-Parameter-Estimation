"""
Root-relative path constants for the EnKF_PPE repository.

Import from anywhere in the repo — ROOT is always the directory
containing this file, regardless of current working directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR        = ROOT / "Data"
GLV_DATA_DIR    = DATA_DIR / "gLV" / "data"
LORENTZ63_DIR   = DATA_DIR / "Lorentz63"
EXPERIMENTS_DIR = ROOT / "experiments"
L63_EXP_DIR     = EXPERIMENTS_DIR / "l63"
GLV_EXP_DIR     = EXPERIMENTS_DIR / "glv"
