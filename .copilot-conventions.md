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

### W&B / Analyse de runs

**✅ Scripts d'analyse :**
- Créer dans `/tmp/check_*.py`
- Utiliser les credentials hardcodés (déjà dans les scripts)
- Grouper par epoch avec `history.groupby('epoch')`
- Toujours vérifier que les colonnes existent avant d'y accéder

### Données & DVC

**✅ Push dataset :**
```bash
dvc add data/train_v8.X.jsonl data/val_v8.X.jsonl data/test_v8.X.jsonl
dvc push data/*.dvc
git add data/*.dvc
git commit -m "feat: dataset v8.X"
```

### RunPod

**✅ Pattern de lancement :**
- Script Python qui crée le pod (pas bash)
- Pattern : `git clone main → setup_runpod.sh → run_adaptive_training.sh`
- Toujours kill les pods existants avant de créer un nouveau

---

## Résumé : Checklist avant chaque action

- [ ] Python multi-ligne ? → Fichier temporaire `/tmp/*.py`
- [ ] Besoin venv ? → `source venv/bin/activate`
- [ ] zsh friendly ? → Pas de heredoc, pas de `-c` complexe
- [ ] Commit dataset ? → `dvc add` puis `dvc push` puis `git add *.dvc`
- [ ] Analyser W&B ? → Script temporaire avec API key hardcodée

