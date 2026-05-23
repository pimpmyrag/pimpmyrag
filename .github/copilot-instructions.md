# Conventions de Développement — pimpmyrag

## Règles ABSOLUES pour GitHub Copilot

### Shell & Terminal (zsh)

**❌ JAMAIS utiliser :**
- `python -c "..."` avec du code multi-ligne entre guillemets → échoue avec zsh
- Heredoc `<< 'EOF'` ou `<< EOF` → échoue systématiquement avec zsh
- Commandes inline complexes avec échappement de quotes

**✅ TOUJOURS faire :**
1. **Créer un fichier Python temporaire** dans `/tmp/script_name.py`
2. **L'exécuter** avec `python3 /tmp/script_name.py`
3. **Activer le venv** avant : `source venv/bin/activate`

**Exemple correct :**
```bash
# 1. Créer le fichier
create_file("/tmp/check_wandb.py", "import wandb\n...")

# 2. Exécuter
run_in_terminal("cd .../multi-head && source venv/bin/activate && python3 /tmp/check_wandb.py")
```

### Python

**✅ Toujours :**
- Activer le venv : `source venv/bin/activate` (path complet si besoin)
- Passer par des fichiers temporaires `/tmp/*.py` au lieu de `-c`
- Utiliser `python3` explicitement (pas `python`)

**Paths venv :**
- Training : `/Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head/venv`

### Commits Git

**✅ Format des messages :**
```
<type>: <description courte>

<détails multi-lignes si besoin>
- changement 1
- changement 2

<métriques/résultats si applicable>
```

**Types :** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`

### W&B / Monitoring de runs

**✅ TOUJOURS utiliser `monitor_run.py` (script permanent, ne PAS créer de nouveaux scripts) :**
```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag
python3 monitor_run.py                    # dernier run actif — toutes métriques
python3 monitor_run.py --compare 6        # comparaison des 6 derniers runs (boundary/epoch)
python3 monitor_run.py --run <id_ou_nom>  # run spécifique
python3 monitor_run.py --watch 120        # refresh auto toutes les 120s
python3 monitor_run.py --epochs 10        # afficher seulement les 10 dernières epochs
```

**Affiche :** NER core (boundary+Δ+trend), TIME labels, INST labels, SVO, Loss, état SVO trigger.

**❌ NE PAS créer de nouveaux `/tmp/check_wandb_*.py`** pour des analyses standard de métriques — utiliser `monitor_run.py` ou l'enrichir si une nouvelle métrique est manquante.

**Pour des analyses ponctuelles spécifiques** (distribution dataset, offsets, etc.) → `/tmp/check_*.py` reste OK.

**Project W&B :** `pimpmyrag-pimpmyrag/pimpmyrag-ner`
**Métriques clés :** `val/boundary_f1` (cible >0.92), `val/fine_f1` (cible >0.84)
**SVO trigger :** `bnd > 0.77 AND coarse > 0.87` (variables `SVO_TRIGGER_BND` / `SVO_TRIGGER_COARSE`)

**Si `r.history()` retourne vide :** NE PAS filtrer avec `keys=` — utiliser `r.history(samples=500, pandas=True)` sans filtre puis filtrer les colonnes en Python.

### Données & DVC

**✅ Push dataset :**
```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag
dvc add training/multi-head/data/train_v8.X.jsonl \
        training/multi-head/data/val_v8.X.jsonl \
        training/multi-head/data/test_v8.X.jsonl
source training/multi-head/.secrets.env
dvc push training/multi-head/data/train_v8.X.jsonl.dvc \
        training/multi-head/data/val_v8.X.jsonl.dvc \
        training/multi-head/data/test_v8.X.jsonl.dvc
git add training/multi-head/data/*.dvc
git commit -m "feat: dataset v8.X"
git push origin main
```

### RunPod / Lancement Training

**✅ Script stable (ne PAS créer de launch_vX.Y.py) :**
```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag
python3 launch_training.py                          # main + GOLD_VERSION par défaut
python3 launch_training.py --gold-version v8.7      # surcharge version dataset
python3 launch_training.py --sha abc1234            # SHA git spécifique
python3 launch_training.py --gpu "RTX 4090"         # GPU préféré
python3 launch_training.py --dry-run                # vérifier sans lancer
```

**Pour changer de version dataset :**
- Modifier **`DEFAULT_GOLD_VERSION`** dans `launch_training.py` (1 seule ligne)
- Modifier **`GOLD_VERSION`** dans `training/multi-head/setup_runpod.sh` (1 seule ligne)
- Modifier **`GOLD_VERSION`** dans `training/multi-head/run_adaptive_training.sh` (1 seule ligne)

---

## Architecture du projet

```
pimpmyrag/
├── launch_training.py              ← SCRIPT DE LANCEMENT STABLE (référence)
├── monitor_run.py                  ← SCRIPT DE MONITORING W&B (référence)
├── training/multi-head/
│   ├── setup_runpod.sh             ← Setup pod : deps + DVC pull + labels check
│   ├── run_adaptive_training.sh    ← Training adaptatif (hard negatives, ramp SVO)
│   ├── build_multitask_dataset.py  ← Construit le dataset multitask (char → tokens)
│   ├── train_multitask.py          ← Script d'entraînement principal
│   ├── labels.py                   ← Labels NER/SVO/morpho (NUM_FINE=38, etc.)
│   ├── .secrets.env                ← Credentials (NE PAS committer !)
│   ├── venv/                       ← Python venv local
│   └── data/                       ← Datasets (gérés par DVC, non committés)
│       ├── train_v8.18.jsonl       ← Dataset courant ✅ (v8.18 = actif)
│       ├── val_v8.18.jsonl
│       ├── test_v8.18.jsonl
│       └── contrastive_v1_fixed.jsonl  ← ⚠️ EN ATTENTE REVIEW (ne pas intégrer)
└── .github/
    └── copilot-instructions.md     ← CE FICHIER
```

## Dataset

**Format JSONL :** chaque ligne = une phrase annotée
```json
{
  "id": "hint_gpe__france__1234",
  "text": "La France a signé l'accord.",
  "spans": [
    {"start": 3, "end": 9, "label": "hint_gpe", "text": "France",
     "svo_role": "SUBJECT", "gov_verb_start": 10}
  ]
}
```

**Versions et état des offsets :**
| Version | Offset errors | Statut |
|---------|--------------|--------|
| v8.0/v8.1 | 443 | propre |
| v8.2–v8.5 | 7 931–9 324 | corrompu (régression boundary !) |
| v8.6 | 64 | ✅ propre (64 = apostrophes, offsets OK) |
| v8.7–v8.17 | — | améliorations successives (SVO, morpho, hard negatives) |
| **v8.18** | **~0** | ✅ **VERSION COURANTE** (training actif) |

**Version courante : `v8.18`**
- 31 328 phrases train
- Extended spans Claude Haiku (`msgbatch_01XKMyCmzpRto1fSSmrg7BeG`)
- Offsets corrigés, labels SVO + morpho complets

**Prochaine version prévue : `v8.19`** (après review contrastive_v1_fixed.jsonl)

**Stockage :** Cloudflare R2 via DVC
- Remote DVC : `r2remote` (configuré dans `.dvc/config`)
- DVC credentials dans `.secrets.env` : `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DVC_R2_ENDPOINT`

## Secrets (`.secrets.env`)

Fichier : `training/multi-head/.secrets.env`  
**NE PAS committer** — ajouté au `.gitignore`

```env
ANTHROPIC_API_KEY=sk-ant-...    # Claude Batch API
WANDB_API_KEY=...               # Weights & Biases
RUNPOD_API_KEY=...              # RunPod
AWS_ACCESS_KEY_ID=...           # Cloudflare R2 (DVC)
AWS_SECRET_ACCESS_KEY=...       # idem
DVC_R2_ENDPOINT=https://...     # idem
```

**Charger dans un script Python :**
```python
from dotenv import load_dotenv
load_dotenv("/Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head/.secrets.env")
```

## Lancer un Training

**Pipeline complet :**
1. `launch_training.py` → crée pod RunPod
2. Pod clone le repo GitHub → checkout le ref
3. `setup_runpod.sh` → install deps + DVC pull dataset + vérif labels
4. `run_adaptive_training.sh` → build multitask dataset + training adaptatif
5. Upload checkpoints vers R2 + log W&B artifact

**Paramètres clés dans `run_adaptive_training.sh` :**
- `GOLD_VERSION` : version dataset (hérité de setup_runpod.sh ou env)
- `NER_WARMUP_EPOCHS` : 0 = meilleur boundary (0.926 en v8.1), 6 = valeur précédente
- `MORPHO_DELAY` : 8 epochs après fin warmup NER
- `ROLE_DELAY` : 12 epochs après fin warmup NER
- `SVO_RAMP_EPOCHS` : 20 epochs pour atteindre 100% SVO

**W&B project :** `pimpmyrag-pimpmyrag/pimpmyrag-ner`  
**Métriques à surveiller :** `val/boundary_f1` (cible > 0.92), `val/fine_f1` (cible > 0.84)

## Scripts d'annotation Claude Batch

**Review Stanza spans :**
```bash
cd training/multi-head && source venv/bin/activate && source .secrets.env
python3 scripts/review_stanza_spans_haiku_batch.py \
  --input data/train_v8.X_rescored.jsonl \
  --output data/train_v8.Y.jsonl \
  --api-key $ANTHROPIC_API_KEY \
  --batch-size 5 \
  --requests-file data/_review_stanza_train_requests.jsonl \
  --model claude-haiku-4-5 \
  --poll-interval 30
# Reprendre un batch existant :
  --batch-id msgbatch_...
```

**À faire pour chaque version (train + val + test) :**
- `_review_stanza_train_requests.jsonl`
- `_review_stanza_val_requests.jsonl`
- `_review_stanza_test_requests.jsonl`

## Fix des offsets après annotation

Après annotation Claude, toujours appliquer le fix d'offsets (script stable) :
```bash
# Adapte INPUT_PATH / OUTPUT_PATH dans le script
python3 /tmp/fix_all_errors_v86.py
```
Le script `/tmp/fix_all_errors_v86.py` est la référence pour corriger :
1. Décalage ±2 chars
2. Offset complètement déplacé (text.find nearest)
3. Apostrophes U+2019 vs U+0027 (gardés tels quels, offsets OK)

---

## ⚠️ TRAVAIL EN COURS — NE PAS TOUCHER (maj 23 mai 2026)

### Training de nuit — v8.18 (en cours / prévu)
- **Dataset actif :** `v8.18` (train=31 328 phrases)
- **Config :** `DEFAULT_GOLD_VERSION = "v8.18"` dans `launch_training.py`
- **Scripts sh :** `GOLD_VERSION=v8.18` dans `setup_runpod.sh` et `run_adaptive_training.sh`
- ⛔ **NE PAS changer GOLD_VERSION** ni re-builder le dataset avant la fin du run

### Dataset contrastif — EN ATTENTE DE REVIEW HUMAINE
Objectif : enrichir le dataset avec des paires contrastives pour améliorer la discrimination fine (hint_state vs hint_notion, hint_field vs hint_notion, etc.)

**Fichiers produits (dans `training/multi-head/data/`) :**
| Fichier | Lignes | État |
|---------|--------|------|
| `contrastive_v1_raw.jsonl` | 594 | Génération brute Claude Haiku |
| `contrastive_v1_svo.jsonl` | 594 | + annotations SVO (Stanza) |
| `_review_contrastive_v1_requests.jsonl` | 119 req | Batch Claude envoyé |
| `contrastive_v1.jsonl` | 594 | Sortie batch (batch `msgbatch_01SGKuWZmd7xUkATvH1YN9Pp` ✅ ended) |
| **`contrastive_v1_fixed.jsonl`** | **594** | ✅ **Post-fix appliqué — EN ATTENTE REVIEW** |

**Fixes appliqués sur `contrastive_v1_fixed.jsonl` (`/tmp/fix_contrastive_v1.py`) :**
- **P1** : 262 verbes mal labelisés NER → forcé `verb_trigger` ✅
- **P2** : 218 `verb_trigger` sans `mood` → ajouté `indicative`/`infinitive` ✅
- **P6** : morpho `gender`/`number` inférée via Stanza (127/249 gender, 213/250 number) ✅

**Résidus connus (acceptables) :**
- 122 spans avec `gender=null` (termes abstraits sans accord)
- 37 spans avec `number=null`
- 67 entités avec `svo_role=None` (hint_gpe, hint_time_date contextuels)

**⛔ NE PAS intégrer `contrastive_v1_fixed.jsonl` dans le dataset** avant :
1. Review humaine des exemples (vérifier que les labels contrastifs sont justes)
2. Vérification offsets (script `/tmp/fix_all_errors_v86.py`)
3. Fusion avec train_v8.18 → nouveau `train_v8.19.jsonl`
4. DVC add/push + commit

**Script d'inspection :** `/tmp/inspect_fixed.py` et `/tmp/deep_inspect_contrastive.py`

---

## Résumé : Checklist avant chaque action

- [ ] Python multi-ligne ? → Fichier temporaire `/tmp/*.py`
- [ ] Besoin venv ? → `source venv/bin/activate`
- [ ] zsh friendly ? → Pas de heredoc, pas de `-c` complexe
- [ ] Commit dataset ? → `dvc add` → `dvc push` → `git add *.dvc` → `git commit`
- [ ] Monitorer / analyser W&B ? → `python3 monitor_run.py` (PAS de nouveau script `/tmp/check_wandb_*.py`)
- [ ] Lancer training ? → `python3 launch_training.py` (PAS de nouveau launch_vX.Y.py)
- [ ] Changer version dataset ? → 1 ligne dans `launch_training.py` + `GOLD_VERSION` dans les 2 scripts sh
