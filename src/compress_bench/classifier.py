from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import load_task_examples, read_manifest
from .quality import normalize_words
from .tokenizers import split_token_strings


@dataclass
class ClassifierReport:
    run_id: str
    created_at: str
    n_tokens: int
    n_train: int
    n_test: int
    accuracy: float
    f1: float
    roc_auc: float
    precision: float
    recall: float
    model_path: str


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
    rows = []
    for example in examples:
        tokenized = split_token_strings(example.prompt, model_id)
        target_words = set(normalize_words(example.target))
        prompt_len = max(1, len(tokenized.tokens))
        for i, token in enumerate(tokenized.tokens):
            clean = token.replace("Ġ", "").replace("▁", "").strip("#").lower()
            is_targetish = clean in target_words if clean else False
            label = 0 if is_targetish else 1
            rows.append(
                {
                    "token": token,
                    "position": i / prompt_len,
                    "length": len(clean),
                    "is_alpha": float(clean.isalpha()),
                    "is_digit": float(clean.isdigit()),
                    "has_upper": float(any(ch.isupper() for ch in token)),
                    "task_hash": float(abs(hash(example.task)) % 997) / 997,
                    "droppable": label,
                }
            )

    frame = pd.DataFrame(rows)
    if frame["droppable"].nunique() < 2:
        raise RuntimeError("Classifier labels collapsed to one class; increase limit_per_task.")

    X = frame.drop(columns=["droppable"])
    y = frame["droppable"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    numeric_cols = ["position", "length", "is_alpha", "is_digit", "has_upper", "task_hash"]
    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    (
                                        "select",
                                        FunctionTransformer(lambda df: df[numeric_cols], validate=False),
                                    ),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    n_jobs=1,
                    random_state=7,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "droppable_classifier.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    report = ClassifierReport(
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_tokens=len(frame),
        n_train=len(X_train),
        n_test=len(X_test),
        accuracy=float(accuracy_score(y_test, pred)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, prob)),
        precision=float(precision),
        recall=float(recall),
        model_path=str(model_path),
    )
    (out / "classifier_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    return report
