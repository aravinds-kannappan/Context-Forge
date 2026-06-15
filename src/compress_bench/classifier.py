"""Token droppability classifier.

Trains a small, fast model that predicts whether a token can be dropped from a
prompt without losing task-critical information. This is a miniature of the
core idea behind learned prompt compression: instead of hand-tuned heuristics,
learn token importance from data.

Features combine:
- character n-grams of the token string (TF-IDF), and
- positional / lexical signals (relative position, length, alpha/digit/case).

Labels are weakly supervised: tokens overlapping target-critical words are
"keep" (0), everything else is "droppable" (1).
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import load_task_examples, read_manifest
from .quality import normalize_words
from .tokenizers import split_token_strings

NUMERIC_COLS = ["position", "length", "is_alpha", "is_digit", "has_upper", "is_punct"]


@dataclass
class ClassifierReport:
    run_id: str
    created_at: str
    model_id: str
    n_tokens: int
    n_train: int
    n_test: int
    droppable_rate: float
    accuracy: float
    f1: float
    roc_auc: float
    precision: float
    recall: float
    top_droppable_tokens: List[str]
    top_keep_tokens: List[str]
    model_path: str


def _clean_token(token: str) -> str:
    """Normalize a token string across BPE / SentencePiece / tiktoken backends."""
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", " ").strip().strip("#").lower()


def _build_frame(examples, model_id: str) -> pd.DataFrame:
    rows = []
    for example in examples:
        tokenized = split_token_strings(example.prompt, model_id)
        target_words = set(normalize_words(example.target))
        prompt_len = max(1, len(tokenized.tokens))
        for i, token in enumerate(tokenized.tokens):
            clean = _clean_token(token)
            is_targetish = clean in target_words if clean else False
            rows.append(
                {
                    "token_text": clean or token,
                    "position": i / prompt_len,
                    "length": len(clean),
                    "is_alpha": float(clean.isalpha()),
                    "is_digit": float(clean.isdigit()),
                    "has_upper": float(any(ch.isupper() for ch in token)),
                    "is_punct": float(bool(clean) and not clean.isalnum() and not clean.isspace()),
                    "droppable": 0 if is_targetish else 1,
                }
            )
    return pd.DataFrame(rows)


def train_droppable_classifier(
    manifest_path: str,
    tasks: Iterable[str],
    model_id: str,
    limit_per_task: int,
    out_dir: str,
    run_id: str = None,
) -> ClassifierReport:
    manifest = read_manifest(manifest_path)
    examples = load_task_examples(manifest, tasks, limit_per_task)
    frame = _build_frame(examples, model_id)

    if frame.empty or frame["droppable"].nunique() < 2:
        raise RuntimeError("Classifier labels collapsed to one class; increase limit_per_task.")

    X = frame.drop(columns=["droppable"])
    y = frame["droppable"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    features = ColumnTransformer(
        [
            (
                "char_ngrams",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=4000),
                "token_text",
            ),
            ("numeric", "passthrough", NUMERIC_COLS),
        ]
    )
    model = Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7),
            ),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )

    top_drop, top_keep = _informative_tokens(X_test, prob)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "droppable_classifier.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    report = ClassifierReport(
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_id=model_id,
        n_tokens=len(frame),
        n_train=len(X_train),
        n_test=len(X_test),
        droppable_rate=float(y.mean()),
        accuracy=float(accuracy_score(y_test, pred)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, prob)),
        precision=float(precision),
        recall=float(recall),
        top_droppable_tokens=top_drop,
        top_keep_tokens=top_keep,
        model_path=str(model_path),
    )
    (out / "classifier_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    return report


def _informative_tokens(X_test: pd.DataFrame, prob: np.ndarray, k: int = 12):
    """Return the most confidently droppable / keep *content* tokens for display.

    Restricted to alphabetic tokens of length >= 3 seen often enough to be
    stable, so the showcase reflects learned signal rather than rare fragments.
    """
    df = X_test.copy()
    df["prob_drop"] = prob
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "has", "have", "had", "not", "but", "which", "their", "its", "will", "would",
        "into", "such", "they", "than", "then", "also", "any", "all", "may", "our",
    }
    df = df[df["token_text"].str.fullmatch(r"[a-z]{3,}")]
    df = df[~df["token_text"].isin(stop)]
    counts = df["token_text"].value_counts()
    common = counts[counts >= 3].index
    df = df[df["token_text"].isin(common)]
    agg = df.groupby("token_text")["prob_drop"].mean().sort_values()
    keep = agg.head(k).index.tolist()
    drop = agg.tail(k).index.tolist()[::-1]
    return drop, keep
