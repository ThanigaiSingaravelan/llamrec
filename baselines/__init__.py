"""LLAMAREC baselines module"""

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NCF Architecture Configuration
NCF_CONFIG = {
    "embedding_dim": 32,
    "hidden_layers": [64, 16],
    "activation": "ReLU",
    "learning_rate": 0.001,
    "epochs": 5
}
