# v8.21 verbfam — corrections multi-têtes validées

Date : 2026-06-03

## Constat

Le modèle DeBERTa apprend bien les têtes supplémentaires dès lors que le signal arrive correctement aux heads verbales et SVO. Le run W&B récent `lq2bhpko` a crashé après l’epoch 6, mais les métriques disponibles montrent une progression nette et exploitable.

## Scores observés

### W&B — `v8.21_verbfam-all-lambdas-1e5-deberta-bs128-RTX_A6000-0602-0127`

| Epoch | boundary F1 | coarse F1 | fine F1 | concrete F1 | abstract F1 | SVO boundary | voice | certainty | gender | number | verb_ptr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.865 | 0.881 | 0.705 | 0.804 | 0.671 | 0.719 | 0.892 | 0.824 | 0.931 | 0.952 | 0.779 |
| 3 | 0.870 | 0.898 | 0.757 | 0.842 | 0.737 | 0.871 | 0.904 | 0.866 | 0.945 | 0.967 | 0.822 |
| 4 | 0.871 | 0.903 | 0.782 | 0.856 | 0.766 | 0.908 | 0.913 | 0.881 | 0.951 | 0.972 | 0.853 |
| 5 | 0.875 | 0.905 | 0.792 | 0.860 | 0.782 | 0.928 | 0.927 | 0.892 | 0.954 | 0.975 | 0.870 |
| 6 | 0.876 | 0.908 | 0.796 | 0.867 | 0.784 | 0.938 | 0.932 | 0.900 | 0.956 | 0.975 | 0.887 |

État SVO trigger à l’epoch 6 : actif (`boundary=0.876`, `coarse=0.908`, `svo_bnd=0.938`).

### Log local epoch 5 — `wandb/run-20260603_043437-9q2zcel3`

Extraits utiles du rapport test :

- fine weighted F1 : `0.805`
- role 12 labels weighted F1 : `0.899`
- verb_family weighted F1 : `0.774`
- morpho : `Gender F1=0.8465`, `Number F1=0.9518`, `Person F1=0.9260`
- SVO boundary verb_trigger : F1 `0.672` sur la classe `verb_trigger`, weighted F1 `0.982`

Les classes verbales ne sont plus en collapse mono-classe : `Communication`, `State_Change`, `Temporal`, `Causality`, `Possession`, `Social`, etc. ont toutes un signal mesurable.

## Corrections apportées

### 1. Parsing robuste des labels verbaux

`build_multitask_dataset.py` normalise désormais les labels verbaux : casse, tirets, espaces, préfixes `verb_*`, et formats legacy du type `Family_Fine` ou `Verb_Family_Fine`.

Objectif : éviter un training silencieux avec des labels `verb_family=NONE` alors que les annotations existent.

### 2. Sélection automatique des datasets `*_verbfam`

`run_training.py` privilégie les fichiers `train|val|test_<gold_version>_verbfam.jsonl` quand ils existent, sauf si `GOLD_VERSION` finit déjà par `_verbfam`.

Objectif : éviter de lancer un run verbfam sur les JSONL non annotés pour ces têtes.

### 3. Têtes verbfam réellement entraînables

`multi_task_model.py` ajoute un MLP dédié aux têtes verbales et l’alimente depuis `span_h` sans `detach()`.

Avant : les têtes verbales recevaient un signal trop isolé/faible ; l’encodeur ne s’adaptait pas correctement aux classes sémantiques verbales.

Après : le gradient verbfam remonte via le MLP dédié et permet à DeBERTa d’apprendre les propriétés sémantiques des verbes.

### 4. Pondération des losses complète

`loss_weighting.py` inclut explicitement :

- `role_coarse`, `role_oblique`, `role`
- `verb_family`, `verb_family_fine`
- `verb_polarity`, `verb_aspect`, `verb_source`

Objectif : que les stratégies de pondération voient bien toutes les têtes entraînées.

### 5. Class weights corrigés

`train_multi_task.py` :

- `role_coarse` repasse en inverse-frequency modéré (`power<=0.5`) pour éviter que la classe majoritaire domine le gradient ;
- `verb_family` utilise un plafond plus permissif (`max_weight=5.0`) pour aider les classes rares sans écraser les classes fréquentes ;
- les counts/weights verbaux sont imprimés au démarrage pour auditer immédiatement la supervision.

### 6. Configs locales de validation

Les configs locales v8.21 documentent les essais CPU/full/ablation et les lambdas utiles pour valider rapidement que le signal verbfam monte avant de payer un run GPU.

## Conclusion

Les résultats à 5–6 epochs invalident l’hypothèse “DeBERTa ne peut pas apprendre ces têtes”. Le problème venait surtout du routage du signal, de la compatibilité des labels, du choix de dataset source et de la pondération des gradients.
