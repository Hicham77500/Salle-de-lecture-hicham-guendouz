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

*Page sans code — 3 parties.*

#### 1. Ce que le chiffre du 4 juillet disait réellement

La phase 0 a confirmé que l'analyste disparu **savait compter**. Sur `datetime`, le 4 juillet produit en moyenne **50,2 relevés par an** (1 154 sur 23 ans), soit plus de cinq fois la moyenne quotidienne (~9,8/j). Cinq des dix journées les plus chargées du fichier tombent un 4 juillet. Juillet concentre **11,5 %** des observations ; le samedi en porte **17,8 %**. Le volume annuel monte jusqu'en 2012–2013. **Aucun de ces chiffres n'est faux.**

Mais un comptage ne répond qu'à une question : *combien de lignes arrivent ce jour-là ?* Il ne dit pas *pourquoi* elles arrivent, ni *ce qu'elles décrivent*. Le dossier du disparu en a tiré une seule lecture — « la population regarde le ciel, elle est habituée à y voir des choses » — et en a conclu qu'une flotte pourrait passer inaperçue. **Cette conclusion n'est pas la seule compatible avec les chiffres.**

**Explication A — le ciel est réellement plus observé le 4 juillet.** Feux d'artifice, terrasses, promenades nocturnes : plus de témoins, plus de signalements. C'est l'hypothèse du dossier.

**Explication B — le pic est aussi un pic de bruit.** Sur nos données, `light` (285 relevés), `fireball` (193) et `circle` (169) dominent le 4 juillet. Beaucoup de témoignages mentionnent des feux d'artifice, des lanternes, des fusées. Un compteur ne distingue pas une observation structurée d'un reflet pyrotechnique mal classé.

**Explication C — d'autres jours produisent des pics comparables pour d'autres raisons.** Halloween (31 octobre), le Nouvel An (1ᵉʳ janvier), certaines nuits d'été : le top 10 inclut 2004-10-31 (85 relevés) et 2014-01-01 (93). Le 4 juillet n'est pas unique : c'est un **jour où l'humain regarde le ciel ET où il fabrique déjà des lumières dans le ciel**. Poser une flotte ce jour-là maximiserait le volume de signalements — pas la discrétion.

**Explication D — compter des dates ne dit rien sur la forme observée.** Deux relevés le même soir peuvent décrire un « cigare », une « sphère » ou une simple « lumière ». Le comptage agrège des phénomènes hétérogènes sous une même étiquette calendaire. C'est précisément ce que le Conseil reproche au Bureau : **208 témoins, 208 textes, zéro lecture.**

En résumé : le chiffre du 4 juillet mesure la **densité d'observation humaine**, pas la **nature des objets signalés**. La flotte a été perdue non parce que les chiffres étaient faux, mais parce qu'on a confondu *beaucoup de relevés* avec *beaucoup de relevés exploitables pour se cacher*.

#### 2. Trois relevés recopiés tels quels

| # | datetime | shape (colonne) | comments (extrait tel que transmis) |
|---|----------|-----------------|-------------------------------------|
| 1 | 10/10/1949 20:30 | cylinder | « This event took place in early fall around 1949-50. It occurred after a Boy Scout meeting in the Baptist Church. The Baptist Church sit » |
| 2 | 7/4/1971 22:00 | cigar | « cigar shaped object appeared low over lake in Great Bend after July 4th fireworks. » |
| 3 | 10/31/1967 20:00 | light | « I remember this event very distinctly. However I am unsure of the date. It's approximate. However I am very sure it was during a Hal » |

**Ce qu'un comptage ne voit pas :**

- **Relevé 1** : la forme annotée est `cylinder`, mais le mot n'apparaît pas dans le texte tronqué. Un compteur du 4 juillet ne verrait qu'une ligne de plus un soir d'octobre ; seule la lecture révèle qu'il faut *inférer* la forme à partir d'un contexte incomplet.
- **Relevé 2** : le témoin lie explicitement l'observation aux **feux d'artifice du 4 juillet**. Le pic calendaire et le contenu se confondent : sans lire, on classerait ce soir comme « favorable à l'infiltration » alors que le témoignage décrit un bruit de fond pyrotechnique.
- **Relevé 3** : le témoin **doute de sa propre date** (« unsure of the date ») mais insiste sur le contexte Halloween. Un agrégat par `datetime` range ce relevé au 31 octobre ; la lecture montre une incertitude que la statistique traite comme une certitude.

#### 3. Commande au Conseil

| | |
|---|---|
| **Entrée** | Le texte du témoin (`comments`), tel que transmis — télégramme tronqué (~135 caractères, médiane ~13 mots), entités HTML non décodées, parfois sans le mot de la forme |
| **Sortie** | La forme observée (`shape`) : une parmi ~29 catégories (light, triangle, fireball, unknown, other, etc.) |
| **Formulation tâche** | *À partir du texte écrit par un témoin, retrouver la forme qu'il a observée.* |

**Question que le comptage ne tranchera jamais :** « Ce témoignage décrit-il une forme reconnaissable, un bruit de fête, ou une description trop pauvre pour trancher ? » — Seule une machine (ou un analyste) qui **lit** le `comments` peut répondre. Le comptage du 4 juillet disait *combien* ; la suite du programme demande *quoi*.

**Phrase pour le Conseil :** *Le texte du témoin entre ; la forme observée sort.*

---

## Acte 2 — Le détecteur de formes

### Phase 2 — Test d'acceptation (8 relevés)

**Architecture** : `Embedding(118, 64) → mean pooling → Linear(8)` — PyTorch, Adam lr=0,05.

**8 relevés sélectionnés** (formes distinctes) :

| # | datetime | Forme vraie | Forme prédite | Extrait comments |
|---|----------|-------------|---------------|------------------|
| 1 | 6/14/2005 18:00 | triangle | triangle | « at around 6 pm i was looking in the sky and i saw a triangle shaped thing… » |
| 2 | 6/14/2009 02:00 | circle | circle | « …there was a purple circle shaped o… » |
| 3 | 11/2/1985 01:00 | cigar | cigar | « As traveling North on Route 22, NY… object with colored light » |
| 4 | 11/21/2011 13:00 | fireball | fireball | « A fireball with a tail that seemed to me maybe a comet?… » |
| 5 | 6/14/2004 20:00 | disk | disk | « The time was between 8:05 pm to 8:38 pm… » |
| 6 | 1/12/2002 21:30 | rectangle | rectangle | « …observed a rectangular object in the north east skies… » |
| 7 | 1/12/2003 07:02 | oval | oval | « …a bright oval body… » |
| 8 | 6/12/2012 23:00 | chevron | chevron | « Saw craft coming from NYC… large dome light… » |

| Métrique | Valeur |
|----------|--------|
| Itérations | **2** |
| Loss finale | **1,515** |
| 8/8 corrects | **☑** |

**Changements si échec** : aucun — convergence du premier coup (8 classes, 8 exemples, textes longs et distincts).

**Figure** : `reports/figures/phase2_overfit_loss.png`

**Ce que ce test prouve** : la boucle d'apprentissage PyTorch fonctionne de bout en bout (tokenisation → embedding → pooling → loss → backprop). Le montage peut mémoriser des exemples.

**Ce qu'il ne prouve pas** : aucune généralisation. Aucune performance sur la transmission entière. Un modèle qui échoue ici ne mérite pas le budget de calcul de la phase 3.

```bash
python -m src.shape.overfit_test
```

---

### Phase 3 — Battre le service statistique

**Décisions jeu de données** :

| Décision | Règle appliquée | Effet |
|----------|-----------------|-------|
| Trous (1 932 sans forme) | **Exclure** | Pas de classe `unknown` artificielle — on ne prédit que sur des relevés étiquetés |
| Fourre-tout (`unknown`, `other`) | **Fusionner** → `other_merged` | 11 232 relevés regroupés (14,3 % du jeu) |
| Doublons sémantiques | `round`→`circle`, `changed`→`changing` | Réduit les classes redondantes |
| Formes rares (< 50 ex.) | **Fusionner** → `rare` | 6 formes (delta, crescent, pyramid, flare, hexagon, dome) |
| Commentaires vides/courts | Exclure si < 5 caractères | 35 relevés retirés |

**Split** : temporel 75/25 sur `datetime` (observation) — cohérent avec partie 1 ; le modèle ne lit pas l'avenir.

| Statistique | Valeur |
|-------------|--------|
| Relevés gardés | **78 365** |
| Classes retenues | **21** |
| Train / Val | 58 773 / 19 592 |

**Du texte brut au premier nombre (PyTorch)** :
1. `comments` → `html.unescape`, filtrage longueur
2. Tokenisation regex + bigrammes (`min_freq=2`, vocab train only)
3. Indices entiers → `Embedding(256)` → mean pooling → `MLP(256)` → logits

| Modèle | Accuracy | Macro-F1 | Weighted-F1 |
|--------|----------|----------|-------------|
| Classe majoritaire (`light`) | 0,214 | 0,017 | 0,076 |
| Linéaire (TF-IDF + SGD) | **0,428** | 0,333 | **0,435** |
| PyTorch (nôtre) | 0,398 | **0,340** | 0,399 |

**Figure** : `reports/figures/phase3_curves.png` — perte train/val pour les deux modèles (40 époques, même découpe).

**Conclusion** : PyTorch **bat le linéaire en macro-F1** (0,340 vs 0,333), métrique adaptée aux 21 classes déséquilibrées. L'accuracy reste légèrement inférieure (0,398 vs 0,428) — le TF-IDF copie efficacement les mots de forme présents dans le texte (cf. phase 8). Signes d'overfitting : écart train/val visible sur les courbes PyTorch.

**Ce que prouve / ne prouve pas** : le montage PyTorch apprend et généralise partiellement ; il reste à masquer le vocabulaire des formes (phase 8) pour une évaluation honnête.

```bash
python -m src.shape.train
```

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
