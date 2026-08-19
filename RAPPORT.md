# RAPPORT — Bureau d'Analyse Terrestre

> Document lu par le Conseil. Une section par phase. Chiffres mesurés, décisions justifiées, échecs inclus.

**Auteur** : Hicham Guendouz  
**Projet** : Salle-de-lecture-hicham-guendouz  
**Données** : `releves_klaxo3.csv` (téléchargé via `python scripts/download_data.py`)

---

## Partie 1 — Manuel du Bureau (sklearn, canulars)

> Pipeline implémenté dans `scripts/run_pipeline.py`. Résultats dans `reports/results.json`.

### Résumé partie 1

| Métrique | Valeur |
|----------|--------|
| Relevés | 80 332 |
| Canulars | 660 (0,82 %) |
| Precision / Recall / PR-AUC | 0,037 / 0,143 / 0,040 |
| Seuil optimal (coût Conseil) | 0,85 |

*Détail complet : voir README.md et `docs/CAHIER_DES_CHARGES.md`.*

---

## Acte 1 — L'héritage

### Phase 0 — Refaire les calculs du disparu

**Date utilisée** : `datetime` (date d'observation)

**Justification du choix** : le dossier recommande de poser la flotte un 4 juillet parce que *« la population regarde le ciel ce jour-là »*. C'est la date de l'**observation** qui compte, pas celle de la **publication** (`date_posted`), qui reflète quand le relevé a été déposé au service de transmission — souvent des semaines plus tard, avec des pics artificiels (ex. 1 510 relevés publiés le 12/12/2009). Avec `datetime`, les pourcentages hebdo/mensuels et la moyenne de ~50 relevés par 4 juillet recollent au dossier ; avec `date_posted`, ils s'écartent nettement (moy. 253/j, samedi 9,6 %).

> **Note dataset** : fichier public NUFORC scrubbed (80 332 lignes) vs fichier cours (88 875). Les ordres de grandeur du dossier sont reproduits ; les écarts restants viennent du sous-ensemble de données et des ~694 dates d'observation non parseables.

| Indicateur | Dossier | Mesure | Question répondue |
|------------|---------|--------|-------------------|
| Jours couverts | 8 894 | **7 533** | Durée totale de la transmission (1990–2014) |
| Moyenne / jour | 9,2 | **9,8** | Rythme habituel d'observations |
| Relevés 4 juillet (moy./an) | 51 | **50,2** | Pic du jour de fête nationale (1 154 au total sur 23 ans) |
| Part samedi | 17,7 % | **17,8 %** | Biais hebdomadaire |
| Part lundi | 12,6 % | **12,5 %** | |
| Part juillet | 11,3 % | **11,5 %** | Saisonnalité estivale |
| Part février | 6,2 % | **6,1 %** | Creux hivernal |
| Max en 1 jour | ~78 | **201** (2010-07-04) | Record absolu sur une journée calendaire |
| Rang 4 juillet | ~15 | **33** | Position de la moyenne annuelle (50,2) parmi toutes les journées |

**Figures** :
- [x] Courbe volume annuel → `reports/figures/volume_annuel.png`
- [x] Top 10 journées les plus chargées (voir ci-dessous)

**Top 10 journées** :

| Rang | Date | Relevés |
|------|------|---------|
| 1 | 2010-07-04 | 201 |
| 2 | 2012-07-04 | 182 |
| 3 | 1999-11-16 | 180 |
| 4 | 2013-07-04 | 175 |
| 5 | 2011-07-04 | 146 |
| 6 | 2009-09-19 | 126 |
| 7 | 2014-01-01 | 93 |
| 8 | 2013-12-31 | 89 |
| 9 | 2004-10-31 | 85 |
| 10 | 2009-07-04 | 84 |

**Conclusion phase 0** : les chiffres du disparu sont **globalement confirmés** sur `datetime` (moyenne/jour, 4 juillet, répartitions). Le volume croît jusqu'en 2012–2013. Le 4 juillet domine le top 10 (5 dates sur 10). La recommandation « poser la flotte un 4 juillet » repose sur un comptage juste — mais ne dit rien sur *ce que* les témoins observent (→ phase 1).

```bash
python -m src.shape.eda
```

---

### Phase 1 — Le chiffre était vrai, la flotte est perdue

*Page sans code — 3 parties :*

#### 1. Ce que le chiffre du 4 juillet disait réellement

*À rédiger*

#### 2. Trois relevés recopiés tels quels

| # | datetime | comments (extrait) |
|---|----------|-------------------|
| 1 | | |
| 2 | | |
| 3 | | |

#### 3. Commande au Conseil

**Entrée** :  
**Sortie** :  
**Formulation tâche** : *À partir du texte d'un témoin (`comments`), prédire la forme observée (`shape`).*

---

## Acte 2 — Le détecteur de formes

### Phase 2 — Test d'acceptation (8 relevés)

| Métrique | Valeur |
|----------|--------|
| Itérations | |
| Loss finale | |
| 8/8 corrects | ☐ |

**Ce que ce test prouve** :  
**Ce qu'il ne prouve pas** :

---

### Phase 3 — Battre le service statistique

**Décisions jeu de données** :
- Trous (2 922) :
- Fourre-tout (`unknown`, `other`) :
- Doublons (`round`/`circle`, etc.) :

| Modèle | Accuracy | Macro-F1 | Classes |
|--------|----------|----------|---------|
| Classe majoritaire | | | |
| Linéaire (TF-IDF) | | | |
| PyTorch (nôtre) | | | |

---

### Phase 4 — Carnet de pannes

| # | Panne | Geste | Signature courbe |
|---|-------|-------|------------------|
| 1 | Overfitting | | |
| 2 | LR trop haut | | |
| 3 | Loss figée | | |

---

### Phase 5 — Budget de calcul

| Version | Temps | Score | Réglage touché | Gain |
|---------|-------|-------|----------------|------|
| Baseline | | | — | — |
| Optimisé | | | | |

**Facteur d'accélération** :

---

### Phase 6 — Champ de vision

| Couche | Étendue ajoutée | Cumul |
|--------|-----------------|-------|
| | | |

**Longueur max (jetons)** :  
**Longueur médiane** :  
**Total ≥ max** : ☐

---

### Phase 7 — Quatre relevés à la fois

**Dépendance inter-batch identifiée** :  
**Score batch=4 avant/après correction** :

---

### Phase 8 — Masque vocabulaire formes

**Mots interdits (count)** :  
**Relevés avec mot interdit restant** : *doit être 0*

| Métrique | Avant masque | Après masque |
|----------|--------------|--------------|
| Accuracy | | |
| Macro-F1 | | |
| Weighted-F1 | | |

---

### Phase 9 — Trois explications

#### Relevé 1 — Succès

*Témoignage + attribution mots*

#### Relevé 2 — Échec

#### Relevé 3 — Hésitation

---

## Acte 3 — Attention

### Phase 10 — Single-head manuel

*Matrice attention + case pronom → antécédent*

### Phase 11 — Encodage positionnel

| Mesure | Phrase correcte | Phrase permutée |
|--------|-----------------|-----------------|
| Écart sorties (avant pos.) | | |
| Écart sorties (après pos.) | | |

### Phase 12 — Facture computationnelle

| Jetons | Temps (s) | Cases matrice |
|--------|-----------|---------------|
| 32 | | |
| 64 | | |
| 128 | | |
| 256 | | |
| 512 | | |

**Facteur doublement longueur** :  
**Longueur inutilisable** :

### Phase 13 — Multi-head

**Désaccord têtes 1 vs 2** :  
**Contrôle (têtes identiques)** :

---

## Acte 4 — Emprunter un cerveau terrien

### Phase 14 — Modèle emprunté

| Régime | Score | Params modifiés | Temps/epoch | Mémoire | Poids disque |
|--------|-------|-----------------|-------------|---------|--------------|
| Référence (phase 8) | | — | | | |
| Frozen | | | | | |
| Fine-tune | | | | | |
| LoRA | | | | | |

**Recommandation Bureau** :

### Phase 15 — Questions au Conseil

**Budget contexte (tokens)** :  
**Questions figées** : *(liste)*  
**Proportion réponses correctement sourcées** :

### Phase 16 — Faire entrer dans le vaisseau

**Marge score acceptée (annoncée avant)** :  
| Mesure | Avant | Après |
|--------|-------|-------|
| Poids disque | | |
| Latence/réponse | | |
| Débit (rep/s) | | |
| Score | | |

### Phase 17 — Faux témoignage

**Grille réglages** :  
**Résultat tri aveugle** :  
**Aucun poids modifié** : ☐ vérifié par code
