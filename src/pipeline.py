"""Sections 8-9 — Pipeline scikit-learn complète."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import heure_cyclique

FEATURE_COLS = ["temoin", "shape", "country", "heure", "duree"]


def build_pipeline() -> Pipeline:
    """Construit la chaîne complète ColumnTransformer + LogisticRegression."""
    return Pipeline(
        [
            (
                "preparation",
                ColumnTransformer(
                    [
                        (
                            "texte",
                            TfidfVectorizer(min_df=5, ngram_range=(1, 2)),
                            "temoin",
                        ),
                        (
                            "categories",
                            OneHotEncoder(
                                min_frequency=50,
                                handle_unknown="infrequent_if_exist",
                            ),
                            ["shape", "country"],
                        ),
                        (
                            "heure",
                            FunctionTransformer(heure_cyclique, validate=False),
                            "heure",
                        ),
                        (
                            "duree",
                            Pipeline(
                                [
                                    (
                                        "trous",
                                        SimpleImputer(
                                            strategy="median", add_indicator=True
                                        ),
                                    ),
                                    ("echelle", StandardScaler()),
                                ]
                            ),
                            ["duree"],
                        ),
                    ]
                ),
            ),
            (
                "modele",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def get_X(df):
    """Retourne le DataFrame de features pour le pipeline."""
    return df[FEATURE_COLS]
