# Plan de migration — Attributs transverses & ontologie « agent » (post-v9)

> **Statut** : brouillon de conception, à reprendre plus tard.
> **Contexte** : rédigé pendant le run v9 (`47v7q79s`, taxo 34 fine / 8 coarse + 5 attributs).
> **But** : tracer les décisions et le plan pour les prochains attributs (`collectivity`,
> `incorporation`, `sector`) et l'éventuelle fusion coarse PER/ORG en « agent » (v10).

---

## 1. État actuel (v9 — livré)

**Taxonomie** : 34 fine / 8 coarse (`PER LOC ORG TIME VALUE OBJECT EVENT CONCEPT` + NONE).

**5 attributs transverses** (dérivés *gratuitement* du fine label via `derive_attributes()`), tous
supervisés sur les spans NER positifs, `NONE` ailleurs :

| Attribut | Valeurs | Dérivation | Statut run v9 (ep2 val f1) |
|----------|---------|-----------|----------------------------|
| animacy | inanimate/animate | label lookup | 0.958 |
| living | non_living/living | label lookup | 0.950 |
| abstract | concrete/abstract | label lookup | 0.908 |
| dynamicity | stative/dynamic (EVENT only) | label lookup | 0.890 |
| work | non_work/work | label lookup | 0.892 |

**Principe clé** : un attribut « bon » est *transverse* (traverse plusieurs coarse) + *dérivable
gratuitement* (lookup sur le fine label, zéro ré-annotation) + *binaire/simple* (facile pour DeBERTa).

---

## 2. Attributs candidats étudiés (non câblés)

### 2.1 `named` (proper vs common) — ❌ écarté
- Factoriserait `event_named`↔`event_nominal`, `inst_name`↔`inst_role`, etc.
- **Verdict** : cue déjà trivial (majuscule) → le fine head l'apprend déjà seul, gain marginal.
- **Risque** : combiné à `sector`, ferait dégénérer ORG à **un seul fine** (voir §4).
- **Décision** : ne PAS l'ajouter.

### 2.2 `collectivity` (individu vs collectif) — ✅ recommandé, gratuit
- **Transverse** PER∪ORG. **Dérivable du label** :
  - individu : `person_name`, `person_role`
  - collectif : `group_role`, `inst_role`, `inst_name`, `org_name`
  - ambigu/à part : `norp` → à trancher (probablement `NONE` ou `collective`)
- Rend explicite l'adjacence `group_role ↔ inst_role/org_name`.
- **Coût** : nul (label lookup). **Risque** : faible. **→ à câbler en priorité.**

### 2.3 `incorporation` (informel vs entité formelle) — ⚠️ annoté, pilote
- Sépare `group_role` (informel) de `inst_*`/`org_name` (formel/personne morale).
- **Pas dérivable du label** ni proprement de la surface (≈50% détectable : sigles + mots
  institutionnels ; `inst_role` sous-détecté car « gouvernement »/« état » ≠ sigle).
- **Coût** : bootstrap lexique (~50%) + passe LLM sur le reste. Mesurer précision avant câblage.

### 2.4 `sector` (state / private / civil) — ⚠️⚠️ annoté, le plus cher
- 3 valeurs. Transverse PER (person_role, group_role) ∪ ORG.
- **Info NEUVE côté PER** : aujourd'hui `person_role` confond ministre (state) et PDG (private) ;
  `group_role` confond policiers (state) et manifestants (civil).
- **Pas gratuit** : lexique couvre <20% sur `org_name` (trop hétérogène).
- ⚠️ Renommer (pas « public/private » : collision avec « société cotée »). Retenir
  `sector ∈ {state, private, civil}`.
- **Coût** : lexique + LLM obligatoire. `org_name` = maillon à risque.

---

## 3. Données empiriques (train v8.24b, 27 837 phrases)

Script : `/tmp/check_collectivity.py` (à re-générer si besoin).

| label | count | %formel* | secteur détecté (lexique) |
|-------|------:|--------:|---------------------------|
| person_name | 7 113 | 1% | ?=99% |
| person_role | 7 016 | 3% | state=13% ?=83% |
| norp | 5 611 | 1% | ?=99% |
| group_role | 8 420 | 8% | civil=9% state=4% priv=4% ?=82% |
| inst_role | 1 601 | 3%† | state=37% ?=62% |
| inst_name | 3 943 | 51% | state=22% ?=72% |
| org_name | 6 606 | 30% | civil=8% priv=6% state=3% ?=82% |

*sigle majuscule ou mot institutionnel. †sous-détecté (« gouvernement »/« état » formels sans sigle).

**Lectures clés** :
1. Collectivité **non lisible en surface** (« le gouvernement » = singulier, collectif de sens) →
   dériver du **label**, pas du pluriel.
2. Contraste `incorporation` réel : group_role 8% formel vs inst_name 51% — mais ~50% surface only.
3. `sector` : 60–82% inconnu au lexique ; `org_name` franchement hétérogène → annotation requise.

---

## 4. Hypothèse v10 — ontologie « agent » (fusion PER/ORG)

**Idée** : une organisation *est* un collectif d'humains → `org`/`inst_role` voisins de `group_role`.
La frontière PER/ORG (métonymie : « la France a signé », « Paris annonce ») est en partie artificielle
et coûteuse en erreurs.

**Cible possible** : coarse unifié « AGENT/HUMAIN », variété portée par attributs
`collectivity × incorporation × sector (× named)`. Le continuum
`group_role → inst_role → org_name` devient un glissement d'attributs.

**Bénéfices** :
- Supprime l'erreur PER↔ORG (devient un attribut, moins pénalisant).
- Colle au besoin SVO/semantic_role (AGENT = tout agent capable d'agir).

**Risques** :
- Refonte lourde (coarse, `COARSE_TO_FINE`, toutes les dérivations SVO/role keyées sur ORG/PER).
- `incorporation`/`sector` non gratuits + flous.
- Consommateurs aval attendant ORG≠PER (liaison KB).
- Piège « coarse à 1 fine » : si on factorise `sector` **et** `named`, ORG s'effondre sur un seul
  fine (`org`) → le fine head ORG devient un no-op. **À éviter** sauf à assumer un modèle
  coarse+attributs pur pour cette branche.

---

## 5. Plan de migration phasé

### Phase A — `collectivity` (gratuit) — *prochaine étape*
- [ ] Ajouter `COLLECTIVITY_LABELS = ["individual", "collective"]` + sentinel NONE dans `labels.py`.
- [ ] Étendre `derive_attributes()` : individual={person_name,person_role} ; collective={group_role,
      inst_role,inst_name,org_name} ; norp → décision (NONE conseillé au départ).
- [ ] Ajouter sous-tête dans `heads/attributes.py` (6ᵉ attribut) + collate + logging W&B/console.
- [ ] Mettre à jour assertions `setup_runpod.sh` si un compteur d'attributs est vérifié.
- [ ] `test_local_launch.py` (2 ep CPU) → run v9.x.
- **Gate A** : après entraînement, matrice de confusion `collectivity + fine → coarse ORG/PER`.
  Si coarse ORG/PER quasi prédictible → feu vert conceptuel pour §Phase C.

### Phase B — `incorporation` + `sector` (annotés) — *pilote séparé*
- [ ] Construire gazetteer STATE/PRIVATE/CIVIL + marqueurs formels (lexique versionné dans le repo).
- [ ] Bootstrap : annoter par lexique, **mesurer précision** sur échantillon (≥200 spans, viser >0.85).
- [ ] Passe LLM (Claude Haiku batch) sur la longue traîne ambiguë, surtout `org_name`.
- [ ] Décision GO/NO-GO par attribut selon précision (org_name = critère bloquant pour `sector`).

### Phase C — fusion coarse « agent » (v10) — *conditionnée à Gate A + B*
- [ ] Si les attributs portent >90% de l'info ORG/PER : fusionner en coarse `AGENT`,
      retirer ORG/PER séparés, réécrire `COARSE_TO_FINE` + dérivations SVO/role.
- [ ] Sinon : garder ORG/PER coarse, conserver les attributs comme enrichissement.
- [ ] Refaire un run de référence dédié (from-scratch, breaking).

---

## 6. Décisions figées

- ✅ `named` : abandonné (gain marginal + risque dégénérescence ORG).
- ✅ `collectivity` : à ajouter, dérivé du label (gratuit).
- ⏸️ `incorporation`, `sector` : pilote annoté avant tout câblage ; `sector` renommé (pas public/private).
- ⏸️ Fusion coarse PER/ORG : **empirique**, décidée sur données après Phase A (pas d'a priori).
- ⚠️ Ne jamais empiler `sector`+`named` sur ORG (→ coarse à 1 fine).

---

## 7. Fichiers concernés (rappel)

- `training/multi-head/labels.py` — taxo, attributs, `derive_attributes()`.
- `training/multi-head/heads/attributes.py` — tête agrégée multi-attributs.
- `training/multi-head/build_multitask_dataset.py` — émission des attributs (`to_v9_fine` + dérivation).
- `training/multi-head/multitask_dataset.py` — collate des attributs.
- `training/multi-head/multi_task_model.py`, `train_multi_task.py`, `run_training.py` — forward/loss/ramp.
- `monitor_run.py` — bloc `🧬 Attributs v9` (déjà à jour pour les 5 actuels).
- `setup_runpod.sh` — assertions labels.

