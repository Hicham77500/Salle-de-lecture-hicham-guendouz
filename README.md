# Salle-de-lecture-hicham-guendouz

**Bureau d'Analyse Terrestre — Détection de canulars Klaxo-3**

Projet de Machine Learning Avancé implémentant les 12 sections du *Manuel du Bureau* : détection de canulars (`hoax`) dans les relevés de transmissions UFO (dataset NUFORC).

## Contexte

Les relevés Klaxo-3 proviennent du [National UFO Reporting Center (NUFORC)](https://nuforc.org). La cible `canular` est fabriquée par **weak supervision** (présence du mot `hoax` dans le commentaire). Toutes les métriques mesurent l'accord avec cette règle, pas la vérité terrain.

> **Note** : le dataset public scrubbed (~80 332 lignes) diffère légèrement du fichier cours (~88 875 lignes avec 196 lignes malformées). La méthodologie reste identique ; seuls les ordres de grandeur changent.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Lancement

```bash
# Pipeline complet (entraînement + figures + modèle)
python scripts/run_pipeline.py

# Notebook interactif
jupyter notebook notebooks/analyse_canulars.ipynb
```

## Résultats principaux

| Métrique | Modèle | Baseline (DummyClassifier) |
|----------|--------|---------------------------|
| Accuracy | 0.970 | 0.993 |
| Precision | 0.037 | 0.000 |
| Recall | 0.143 | 0.000 |
| PR-AUC | 0.040 | — |

Le baseline bat le modèle en accuracy (99,3 %) en prédisant toujours « non-canular » — illustration de l'**accuracy paradox** sur classe déséquilibrée (0,82 % de canulars).

**Seuil optimal** (grille Conseil : 30 cr./canular raté, 2 cr./fausse alerte) : **0,85** → facture 3 872 cr. vs 4 400 au seuil 0,50.

## Tableau anti-fuite

| Colonne | Disponible au moment de la prédiction ? | Dans le modèle ? |
|---------|----------------------------------------|-----------------|
| `temoin` (commentaire sans notes Bureau) | Oui — témoignage terrain | Oui |
| `comments` brut (avec notes `((...))`) | Non — note postérieure | Non |
| `shape`, `country` | Oui | Oui |
| `heure`, `duree` | Oui | Oui |
| `canular` (cible) | Non | Non (cible) |

## Structure du projet

```
├── data/releves_klaxo3.csv      # Données (non versionnées, ~13 Mo)
├── notebooks/analyse_canulars.ipynb
├── src/                         # Modules Python réutilisables
│   ├── load_data.py             # Section 1 — CSV robuste
│   ├── clean.py                 # Sections 2-3, 5
│   ├── features.py              # Section 10
│   ├── split.py                 # Section 7
│   ├── pipeline.py              # Sections 8-9
│   ├── evaluate.py              # Sections 4-6, 11
│   └── explain.py               # Section 12
├── scripts/
│   ├── download_data.py
│   └── run_pipeline.py
├── models/                      # pipeline_canular.joblib
└── reports/figures/             # Graphiques générés
```

## Auteur

Hicham Guendouz — Projet Salle de lecture, Machine Learning Avancé.

## Licence

Données NUFORC : usage éducatif. Code : MIT.
