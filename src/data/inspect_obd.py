from pathlib import Path

import numpy as np
from obp.dataset import OpenBanditDataset

# Load data directly
raw_dir = Path("data/raw/obd")
dataset = OpenBanditDataset(behavior_policy="random", campaign="all", data_path=raw_dir)
bandit_feedback = dataset.obtain_batch_bandit_feedback()

print(f"Positions: {np.unique(bandit_feedback['position'])}")
print(f"Pscores: Min={np.min(bandit_feedback['pscore'])}, Max={np.max(bandit_feedback['pscore'])}")
print(f"Actions: Min={np.min(bandit_feedback['action'])}, Max={np.max(bandit_feedback['action'])}")
print(f"Rewards: {np.unique(bandit_feedback['reward'])}")
