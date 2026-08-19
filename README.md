# Salle-de-lecture-hicham-guendouz

**Bureau d'Analyse Terrestre — Klaxo-3**

Projet en deux parties :
- **Partie 1** (sklearn) : détection de canulars — *Manuel du Bureau*
- **Partie 2** (PyTorch) : classification de formes + attention + transfer learning — *La salle de lecture*

## Statut du projet

| Partie | Statut | Documentation |
|--------|--------|---------------|
| Partie 1 — Canulars | **~90 % complet** | [Cahier des charges](docs/CAHIER_DES_CHARGES.md) |
| Partie 2 — Formes / Attention | **Préparé, à implémenter** | [Roadmap](docs/PARTIE2_ROADMAP.md) |

**Rapport Conseil** : [`RAPPORT.md`](RAPPORT.md) — une section par phase (template prêt).

## Installation

### Partie 1 (sklearn)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_pipeline.py
```

### Partie 2 (PyTorch — en plus)

```bash
pip install -r requirements-part2.txt
python -m src.shape.eda          # Phase 0 — chiffres dossier 4 juillet
```

> **Colab recommandé** pour les phases GPU (3, 5, 14). Pas de GPU local requis pour phases 10-13 (attention manuelle sur petits textes).

## Données

- Fichier : `data/releves_klaxo3.csv` (non versionné, ~13 Mo)
- Source : NUFORC public ([planetsig/ufo-reports](https://github.com/planetsig/ufo-reports))
- **Note** : dataset scrubbed ~80 332 lignes ; fichier cours ~88 875 lignes avec 196 lignes malformées

```bash
python scripts/download_data.py
```

11 colonnes (sans en-tête dans la source) :

`datetime`, `city`, `state`, `country`, `shape`, `duration_seconds`, `duration_hours_min`, `comments`, `date_posted`, `latitude`, `longitude`

Les `comments` sont tronqués à ~135 caractères (médiane ~13 mots).

## Partie 1 — Résultats canulars

| Métrique | Modèle | Baseline |
|----------|--------|----------|
| Accuracy | 0,970 | 0,993 |
| Precision | 0,037 | 0,000 |
| Recall | 0,143 | 0,000 |
| PR-AUC | 0,040 | — |

Seuil optimal (30 cr./canular raté, 2 cr./fausse alerte) : **0,85**

Détail : `reports/results.json`, figures dans `reports/figures/`.

### Checklist partie 1

| Section | Statut |
|---------|--------|
| 1. CSV robuste | Partiel (0 lignes écartées sur dataset public) |
| 2. Coercion types | Fait |
| 3. Weak supervision | Fait |
| 4. Precision/recall | Fait |
| 5. Anti-fuite | Fait |
| 6. Baseline | Fait |
| 7. Split groupe + temporel | Fait |
| 8. Missing + indicator | Fait |
| 9. Pipeline sklearn | Fait |
| 10. Feature engineering | Fait |
| 11. Seuil + calibration + CV | Fait |
| 12. Audit + ablation | Partiel |

## Partie 2 — Prochaines étapes

1. **Phase 0** : `python -m src.shape.eda` → remplir `RAPPORT.md`
2. **Phase 1** : rédiger critique comptage vs texte
3. **Phase 2-3** : PyTorch shape classifier vs TF-IDF linéaire
4. **Phase 10-13** : attention codée à la main (`src/attention/`)
5. **Phase 14-17** : HuggingFace + LoRA + déploiement

Voir [docs/PARTIE2_ROADMAP.md](docs/PARTIE2_ROADMAP.md) pour le plan complet.

## Structure

```
├── RAPPORT.md                   # Rapport Conseil (phases 0-17)
├── docs/
│   ├── CAHIER_DES_CHARGES.md    # Audit conformité P1 + P2
│   └── PARTIE2_ROADMAP.md       # Plan implémentation partie 2
├── src/
│   ├── load_data.py … explain.py   # Partie 1 (sklearn)
│   ├── shape/                      # Partie 2 acte 2
│   ├── attention/                  # Partie 2 acte 3
│   └── transfer/                   # Partie 2 acte 4
├── scripts/
│   ├── download_data.py
│   └── run_pipeline.py             # Pipeline partie 1
├── notebooks/
│   └── analyse_canulars.ipynb
├── requirements.txt                # Partie 1
└── requirements-part2.txt          # PyTorch + transformers
```

## Tableau anti-fuite (partie 1)

| Colonne | Au moment prédiction | Modèle |
|---------|---------------------|--------|
| `temoin` | Oui | Oui |
| `comments` brut | Non | Non |
| `shape`, `country`, `heure`, `duree` | Oui | Oui |
| `canular` | Non (cible) | Non |

## Auteur

Hicham Guendouz — [GitHub](https://github.com/Hicham77500/Salle-de-lecture-hicham-guendouz)

## Licence

Données NUFORC : usage éducatif. Code : MIT.
