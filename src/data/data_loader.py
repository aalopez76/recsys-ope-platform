"""
src.data.data_loader
====================
Unified `DataLoader` for the recsys-ope-platform.

Supports three backends, all returning a common ``BanditDataset`` namedtuple:

+-----------+--------------------------------------------------+
| Backend   | Description                                      |
+===========+==================================================+
| synthetic | Reads existing ``data/sample/*.npz`` files.       |
|           | Default — no downloads required.                  |
+-----------+--------------------------------------------------+
| obd       | Wraps built artifacts from ``build_obd_datasets`` |
|           | stored in ``data/bandit_feedback/``.              |
+-----------+--------------------------------------------------+
| movielens | Downloads ML-1M via RecBole and converts          |
|           | rating interactions to bandit feedback format.   |
+-----------+--------------------------------------------------+

Usage
-----
```powershell
# Print shape summary (synthetic)
python -m src.data.data_loader --dataset synthetic

# Load OBD artifacts
python -m src.data.data_loader --dataset obd --out data/loaded/

# Load MoviesLens-1M (downloads on first run)
python -m src.data.data_loader --dataset movielens --out data/loaded/
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------


class BanditDataset(NamedTuple):
    """Minimal bandit feedback container shared across all backends.

    Attributes
    ----------
    context:
        User/round feature matrix, shape ``(n_rounds, context_dim)``.
    action:
        Observed action indices, shape ``(n_rounds,)``.
    reward:
        Binary reward signal, shape ``(n_rounds,)``.
    pscore:
        Logging propensity scores, shape ``(n_rounds,)``.
    action_context:
        Item/action feature matrix, shape ``(n_actions, action_dim)``.
    n_actions:
        Number of discrete actions.
    n_rounds:
        Total number of logged rounds.
    train_idx:
        Indices for the training split.
    val_idx:
        Indices for the validation split.
    test_idx:
        Indices for the test split.
    source:
        String identifier of the backend used.
    """

    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    action_context: np.ndarray
    n_actions: int
    n_rounds: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    source: str


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _split_indices(
    n: int,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproducible train/val/test split of ``range(n)``."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def _load_synthetic(
    bandit_path: str,
    splits_path: str,
    seed: int,
) -> BanditDataset:
    """Load from existing ``data/sample/*.npz`` files."""
    logger.info("Backend=synthetic | bandit=%s", bandit_path)
    d = np.load(bandit_path, allow_pickle=True)
    s = np.load(splits_path, allow_pickle=True)
    n_rounds = int(d["context"].shape[0])
    n_actions = int(d["action_context"].shape[0])
    train_idx = s["train_idx"].astype(int)
    val_idx = s["val_idx"].astype(int)
    test_idx = s["test_idx"].astype(int)
    ds = BanditDataset(
        context=d["context"].astype(np.float32),
        action=d["action"].astype(int),
        reward=d["reward"].astype(np.float32),
        pscore=d["pscore"].astype(np.float32),
        action_context=d["action_context"].astype(np.float32),
        n_actions=n_actions,
        n_rounds=n_rounds,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        source="synthetic",
    )
    logger.info(
        "  n_rounds=%d  n_actions=%d  context_dim=%d",
        n_rounds,
        n_actions,
        ds.context.shape[1],
    )
    return ds


def _load_obd(
    artifact_dir: str,
    split_ratios: tuple[float, float, float],
    seed: int,
) -> BanditDataset:
    """Load from OBD pipeline artifacts (``data/bandit_feedback/``)."""
    p = Path(artifact_dir)
    bf_path = p / "bandit_feedback.npz"
    splits_path = p / "split_manifest.json"

    if not bf_path.exists():
        raise FileNotFoundError(
            f"OBD artifact not found: {bf_path}\n" "Run `make build_data` first to generate OBD artifacts."
        )

    logger.info("Backend=obd | path=%s", artifact_dir)
    d = np.load(str(bf_path), allow_pickle=True)
    n_rounds = int(d["context"].shape[0])
    n_actions = int(d["action_context"].shape[0])

    # Try pre-computed splits; fall back to random split
    if splits_path.exists():
        import json

        with open(splits_path, encoding="utf-8") as f:
            sm = json.load(f)
        train_idx = np.array(sm["train_idx"], dtype=int)
        val_idx = np.array(sm["val_idx"], dtype=int)
        test_idx = np.array(sm["test_idx"], dtype=int)
        logger.info("  Loaded pre-computed splits from split_manifest.json")
    else:
        train_idx, val_idx, test_idx = _split_indices(n_rounds, split_ratios, seed)
        logger.info("  Generated random splits (no split_manifest.json found)")

    ds = BanditDataset(
        context=d["context"].astype(np.float32),
        action=d["action"].astype(int),
        reward=d["reward"].astype(np.float32),
        pscore=d["pscore"].astype(np.float32),
        action_context=d["action_context"].astype(np.float32),
        n_actions=n_actions,
        n_rounds=n_rounds,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        source="obd",
    )
    logger.info(
        "  n_rounds=%d  n_actions=%d  context_dim=%d",
        n_rounds,
        n_actions,
        ds.context.shape[1],
    )
    return ds


def _load_movielens(
    recbole_config: str,
    rating_threshold: float,
    split_ratios: tuple[float, float, float],
    seed: int,
) -> BanditDataset:
    """Load MovieLens-1M via RecBole and convert to bandit format.

    Ratings >= ``rating_threshold`` → reward = 1 (positive interaction).

    Parameters
    ----------
    recbole_config:
        Path to the RecBole YAML config (must specify ``dataset: ml-1m``).
    rating_threshold:
        Minimum rating to treat as a positive reward.
    split_ratios:
        (train, val, test) proportions.
    seed:
        Random seed for reproducible splits.
    """
    try:
        from recbole.config import Config  # type: ignore[import]
        from recbole.data import create_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "RecBole is required for the movielens backend.\n" "Install with: pip install recbole"
        ) from exc

    logger.info("Backend=movielens | threshold=%.1f | config=%s", rating_threshold, recbole_config)
    cfg = Config(model="BPR", dataset="ml-1m", config_file_list=[recbole_config])
    dataset = create_dataset(cfg)

    inter = dataset.inter_feat
    item_ids = inter[dataset.iid_field].numpy().astype(int)
    ratings = inter["rating"].numpy().astype(np.float32)
    user_ids = inter[dataset.uid_field].numpy().astype(int)

    n_actions = int(item_ids.max()) + 1
    n_rounds = len(item_ids)
    context_dim = 8  # simple one-hot-like user embedding placeholder

    # Build context: normalised user_id as embedding (placeholder)
    context = np.zeros((n_rounds, context_dim), dtype=np.float32)
    for i, u in enumerate(user_ids):
        context[i, u % context_dim] = 1.0

    # Action context: identity-ish item embeddings (placeholder)
    action_context = np.eye(n_actions, min(n_actions, 16), dtype=np.float32)

    reward = (ratings >= rating_threshold).astype(np.float32)
    pscore = np.full(n_rounds, 1.0 / n_actions, dtype=np.float32)
    action = item_ids.astype(int)

    train_idx, val_idx, test_idx = _split_indices(n_rounds, split_ratios, seed)
    ds = BanditDataset(
        context=context,
        action=action,
        reward=reward,
        pscore=pscore,
        action_context=action_context,
        n_actions=n_actions,
        n_rounds=n_rounds,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        source="movielens",
    )
    logger.info(
        "  n_rounds=%d  n_actions=%d  context_dim=%d  positive_rate=%.4f",
        n_rounds,
        n_actions,
        context_dim,
        float(reward.mean()),
    )
    return ds


# ---------------------------------------------------------------------------
# Public DataLoader class
# ---------------------------------------------------------------------------


class DataLoader:
    """Unified dataset loader for the recsys-ope-platform.

    Parameters
    ----------
    dataset:
        Backend selector: ``"synthetic"``, ``"obd"``, or ``"movielens"``.
    seed:
        Random seed for reproducible splits.
    split_ratios:
        Tuple of ``(train, val, test)`` proportions; must sum to 1.
    bandit_path:
        Path to ``bandit_feedback_sample.npz`` (synthetic backend only).
    splits_path:
        Path to ``split_manifest_sample.npz`` (synthetic backend only).
    obd_artifact_dir:
        Directory containing OBD pipeline artifacts (obd backend only).
    recbole_config:
        Path to RecBole YAML config (movielens backend only).
    rating_threshold:
        Minimum rating for positive reward (movielens backend only).
    """

    def __init__(
        self,
        dataset: str = "synthetic",
        seed: int = 42,
        split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
        bandit_path: str = "data/sample/bandit_feedback_sample.npz",
        splits_path: str = "data/sample/split_manifest_sample.npz",
        obd_artifact_dir: str = "data/bandit_feedback",
        recbole_config: str = "configs/recbole.yaml",
        rating_threshold: float = 4.0,
    ) -> None:
        self.dataset = dataset
        self.seed = seed
        self.split_ratios = split_ratios
        self.bandit_path = bandit_path
        self.splits_path = splits_path
        self.obd_artifact_dir = obd_artifact_dir
        self.recbole_config = recbole_config
        self.rating_threshold = rating_threshold

    def load(self) -> BanditDataset:
        """Load and return the dataset as a ``BanditDataset``."""
        if self.dataset == "synthetic":
            return _load_synthetic(self.bandit_path, self.splits_path, self.seed)
        if self.dataset == "obd":
            return _load_obd(self.obd_artifact_dir, self.split_ratios, self.seed)
        if self.dataset == "movielens":
            return _load_movielens(self.recbole_config, self.rating_threshold, self.split_ratios, self.seed)
        raise ValueError(f"Unknown dataset: {self.dataset!r}. " "Choose from: synthetic | obd | movielens")

    def to_tf_dataset(
        self,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> Any:  # returns tf.data.Dataset when TensorFlow is available
        """Wrap the loaded dataset in a ``tf.data.Dataset`` for large-scale training.

        Parameters
        ----------
        split:
            ``"train"``, ``"val"``, or ``"test"``.
        batch_size:
            Number of samples per batch.
        shuffle:
            Whether to shuffle the data each epoch.

        Returns
        -------
        tf.data.Dataset
            Yields ``(context, action, reward, pscore)`` tuples.
        """
        try:
            import tensorflow as tf  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("TensorFlow is required for to_tf_dataset().") from exc

        ds = self.load()
        idx_map = {"train": ds.train_idx, "val": ds.val_idx, "test": ds.test_idx}
        if split not in idx_map:
            raise ValueError(f"split must be one of {list(idx_map)}")
        idx = idx_map[split]
        tfd = tf.data.Dataset.from_tensor_slices(
            {
                "context": ds.context[idx],
                "action": ds.action[idx],
                "reward": ds.reward[idx],
                "pscore": ds.pscore[idx],
            }
        )
        if shuffle:
            tfd = tfd.shuffle(len(idx), seed=self.seed)
        return tfd.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DataLoader CLI — load and inspect a dataset")
    p.add_argument(
        "--dataset",
        choices=["synthetic", "obd", "movielens"],
        default="synthetic",
    )
    p.add_argument("--bandit", default="data/sample/bandit_feedback_sample.npz")
    p.add_argument("--splits", default="data/sample/split_manifest_sample.npz")
    p.add_argument("--obd-dir", default="data/bandit_feedback")
    p.add_argument("--recbole-config", default="configs/recbole.yaml")
    p.add_argument("--rating-threshold", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Optional: save dataset as .npz to this path")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()
    loader = DataLoader(
        dataset=args.dataset,
        seed=args.seed,
        bandit_path=args.bandit,
        splits_path=args.splits,
        obd_artifact_dir=args.obd_dir,
        recbole_config=args.recbole_config,
        rating_threshold=args.rating_threshold,
    )
    ds = loader.load()
    print(f"\nBanditDataset [{ds.source}]")
    print(f"  n_rounds       : {ds.n_rounds}")
    print(f"  n_actions      : {ds.n_actions}")
    print(f"  context.shape  : {ds.context.shape}")
    print(f"  action.shape   : {ds.action.shape}")
    print(f"  reward mean    : {ds.reward.mean():.4f}")
    print(f"  train / val / test : {len(ds.train_idx)} / {len(ds.val_idx)} / {len(ds.test_idx)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(out_path),
            context=ds.context,
            action=ds.action,
            reward=ds.reward,
            pscore=ds.pscore,
            action_context=ds.action_context,
            train_idx=ds.train_idx,
            val_idx=ds.val_idx,
            test_idx=ds.test_idx,
        )
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
