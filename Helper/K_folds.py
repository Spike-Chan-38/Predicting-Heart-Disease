from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def build_folds(
    data: pd.DataFrame,
    target_col: str,
    k: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    random_state: int = 42,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Split a dataset into k folds for training/validation.

    Returns a list of dicts: {"fold": int, "train": DataFrame, "val": DataFrame}
    """
    if target_col not in data.columns:
        raise ValueError(f"target_col '{target_col}' not found in data columns")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    if stratified:
        splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        split_iter = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        split_iter = splitter.split(X)

    folds: List[Dict[str, pd.DataFrame]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)
        folds.append({"fold": fold_idx, "train": train_df, "val": val_df})

    return folds


def kth_fold(
    data: pd.DataFrame,
    target_col: str,
    k: int = 5,
    fold_index: int = 1,
    stratified: bool = True,
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience helper to get train/val for a specific fold (1-based index).
    """
    if fold_index < 1 or fold_index > k:
        raise ValueError(f"fold_index must be between 1 and {k}")

    folds = build_folds(
        data=data,
        target_col=target_col,
        k=k,
        stratified=stratified,
        shuffle=shuffle,
        random_state=random_state,
    )

    fold = folds[fold_index - 1]
    return fold["train"], fold["val"]
