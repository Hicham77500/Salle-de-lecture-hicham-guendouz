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

**Date utilisée** : `[datetime | date_posted]` — *à compléter*

**Justification du choix** : *à compléter*

| Indicateur | Dossier | Mesure | Question répondue |
|------------|---------|--------|-------------------|
| Jours couverts | 8 894 | | Durée totale de la transmission |
| Moyenne / jour | 9,2 | | Rythme habituel d'observations |
| Relevés 4 juillet | 51 | | Pic du jour de fête nationale |
| Part samedi | 17,7 % | | Biais hebdomadaire |
| Part lundi | 12,6 % | | |
| Part juillet | 11,3 % | | Saisonnalité |
| Part février | 6,2 % | | |
| Max en 1 jour | ? | | Record absolu |
| Rang 4 juillet | ? | | Position du pic recommandé |

**Figures** :
- [ ] Courbe volume annuel
- [ ] Top 10 journées les plus chargées

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
