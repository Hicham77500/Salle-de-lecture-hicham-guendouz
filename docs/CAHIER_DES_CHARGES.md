# Cahier des charges — Audit de conformité

Document de référence croisant l'implémentation actuelle avec :
- **Partie 1** : *Le Manuel du Bureau* (`partie2-1_le_manuel_du_bureau.pdf`) — 12 sections ML classique
- **Partie 2** : *La salle de lecture* (`partie2-2_attention_transformers_et_reutilisation_de_modeles.pdf`) — PyTorch, attention, transfer learning

---

## Partie 1 — Détection de canulars (sklearn)

| # | Section | Statut | Détail |
|---|---------|--------|--------|
| 1 | Chargement CSV hostile | **PARTIEL** | `src/load_data.py` : lecture `csv.reader`, comptage `gardees`/`ecartees`. Dataset public scrubbed → 0 ligne écartée (fichier cours : 196). |
| 2 | Coercion de types | **FAIT** | `errors="coerce"` sur lat/lon/duration ; datetime format explicite `%m/%d/%Y %H:%M`. |
| 3 | Cible fabriquée (weak supervision) | **FAIT** | `canular` via `contains("hoax")` ; limites documentées dans README. |
| 4 | Précision et rappel | **FAIT** | Matrice confusion, PR-AUC, `classification_report` dans `evaluate.py`. |
| 5 | Fuite de données | **FAIT** | Retrait notes `((...))` → colonne `temoin` ; tableau anti-fuite dans README. |
| 6 | Modèle bête + déséquilibre | **FAIT** | `DummyClassifier` + `class_weight="balanced"`. |
| 7 | Découpe honnête | **FAIT** | `temporal_split` + `group_split` ; comparaison des deux dans `run_pipeline.py`. |
| 8 | Valeurs manquantes (MNAR) | **FAIT** | `SimpleImputer(add_indicator=True)` ; analyse MNAR dans `missing_signal_report`. |
| 9 | Pipeline sklearn | **FAIT** | `ColumnTransformer` + `Pipeline` complet, pas de fuite de preprocessing. |
| 10 | Feature engineering | **FAIT** | HTML unescape, heure cyclique (sin/cos), `OneHotEncoder(min_frequency=50)`. |
| 11 | Seuil + calibration | **FAIT** | Grille coûts Conseil, `CalibratedClassifierCV`, Brier ; intervalle PR-AUC via CV stratifiée. |
| 12 | Explication + audit | **PARTIEL** | Ablation par colonne + audit par pays ; `permutation_importance` disponible mais non exécutée par défaut (coût TF-IDF). |

### Écarts connus partie 1

| Écart | Impact | Action |
|-------|--------|--------|
| Dataset 80 332 vs 88 875 lignes | Chiffres différents du manuel | Obtenir `releves_klaxo3.csv` cours ou accepter NUFORC scrubbed |
| 0 lignes CSV malformées | Section 1 non démontrée sur vrais cas | Code prêt ; note dans README |
| Pas de `RAPPORT.md` partie 1 | Livrable Conseil absent pour P1 | Partie 2 exige `RAPPORT.md` — voir template racine |

### Verdict partie 1

**~90 % complet** — pipeline reproductible, méthodologie conforme. Manques : dataset cours exact, rapport écrit partie 1, permutation importance en prod.

---

## Partie 2 — Classification de formes + Attention + Transfer (PyTorch)

> **Changement de tâche** : la partie 2 prédit `shape` à partir de `comments`, pas `canular`.

### Contraintes générales (PDF)

| Exigence | Statut |
|----------|--------|
| Framework PyTorch imposé | **À FAIRE** |
| `RAPPORT.md` — une section par phase | **TEMPLATE CRÉÉ** |
| Repo git, commits atomiques, pas de `git add .` | **EN COURS** (repo existant) |
| Données hors repo, téléchargement auto | **FAIT** (`scripts/download_data.py`) |
| Colab autorisé (pas de GPU local) | **À PLANIFIER** |
| 88 875 relevés, 11 colonnes sans en-tête | **PARTIEL** (80 332 lignes actuelles) |

### Acte 1 — L'héritage (Phases 0-1)

| Phase | Objectif | Statut |
|-------|----------|--------|
| 0 | Reproduire chiffres dossier 4 juillet (8894 jours, 9,2/j, 51 le 4 juil., etc.) | **À FAIRE** |
| 1 | Page RAPPORT sans code : critique comptage vs lecture témoignages | **À FAIRE** |

### Acte 2 — Détecteur de formes (Phases 2-9)

| Phase | Objectif | Statut |
|-------|----------|--------|
| Phase 2 overfit 8 | **FAIT** | 2 itérations, 8/8 corrects |
| 3 | PyTorch vs linéaire (TF-IDF) ; courbes train/val | **À FAIRE** |
| 4 | Carnet de pannes (3 pannes volontaires) | **À FAIRE** |
| 5 | Optimisation budget calcul (chronomètre) | **À FAIRE** |
| 6 | Champ de vision : récepteur + preuve dépendance tous mots | **À FAIRE** |
| 7 | Batch size = 4 ; corriger dépendance inter-batch | **À FAIRE** |
| 8 | Masquer vocabulaire des formes (anti-copie mot) | **À FAIRE** |
| 9 | Explicabilité mot-à-mot sur 3 relevés test | **À FAIRE** |

### Acte 3 — Attention (Phases 10-13)

| Phase | Objectif | Statut |
|-------|----------|--------|
| 10 | Attention single-head codée à la main (pas de lib prête) | **À FAIRE** |
| 11 | Encodage positionnel ; prouver sensibilité à l'ordre | **À FAIRE** |
| 12 | Complexité O(n²) : benchmark 32→512 jetons | **À FAIRE** |
| 13 | Multi-head (2 têtes) + mesure de désaccord | **À FAIRE** |

### Acte 4 — Transfer learning (Phases 14-17)

| Phase | Objectif | Statut |
|-------|----------|--------|
| 14 | Modèle pré-entraîné : frozen / fine-tune / LoRA | **À FAIRE** |
| 15 | Q&R en langage naturel avec citations de relevés | **À FAIRE** |
| 16 | Compression + déploiement vaisseau (quantization, ONNX) | **À FAIRE** |
| 17 | Génération faux témoignages (sans modifier poids) | **À FAIRE** |

### Verdict partie 2

**0 % implémenté** — structure préparée (`src/shape/`, `src/attention/`, `src/transfer/`). Voir [PARTIE2_ROADMAP.md](PARTIE2_ROADMAP.md).
