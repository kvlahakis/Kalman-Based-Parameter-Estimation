from pathlib import Path

ROOT = Path(__file__).parent

DATA_DIR = ROOT / "Data"
LORENTZ63_DIR = DATA_DIR / "Lorentz63"

MODELS_DIR = ROOT / "Models"
ENKF_DIR = MODELS_DIR / "ENKF"
NEURAL_ODE_DIR = MODELS_DIR / "NeuralODE"
TRANSPORT_DIR = MODELS_DIR / "Transport"

EXPERIMENTS_DIR = ROOT / "Experiments"
VIS_DIR = ROOT / "Vis"
