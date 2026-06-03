# PimpMyRAG taxonomy

> Generated from `training/multi-head/labels.py` on 2026-06-03.
> Do not edit label lists manually here; run `python3 training/multi-head/export_taxonomy.py` after changing `labels.py`.

## Source of truth

- Python taxonomy: `training/multi-head/labels.py`
- Machine-readable export: `docs/taxonomy.json`
- JSON Schema: `docs/taxonomy.schema.json`

## Summary

| Family | Active labels | Sentinel / note |
|---|---:|---|
| NER coarse | 10 | `NONE` is an active model label at id `9` |
| NER fine | 38 | sentinel `FINE_NONE_ID=38` outside active range |
| Syntax spans | 3 | sentinel `3` |
| Role 12 labels | 12 | `NONE` is label id `6`; primary role head |
| Role coarse | 5 | auxiliary/cascade head; sentinel `5` |
| Role oblique | 10 | auxiliary/cascade head; sentinel `10` |
| Verb family | 12 | sentinel `12` |
| Verb family fine | 38 | sentinel `38` |
| Verb polarity/aspect/source | 3 / 2 / 3 | verb-trigger only |
| Voice/certainty | 2 / 3 | verb-trigger only |
| Gender/number/person | 2 / 2 / 3 | supervised where annotated |

## NER

### Coarse labels

`0:PER`, `1:LOC`, `2:ORG`, `3:TIME`, `4:EVENT`, `5:OBJECT`, `6:VALUE`, `7:WORK`, `8:ABSTRACT`, `9:NONE`

### Fine labels

`0:hint_person_name`, `1:hint_person_role`, `2:hint_norp`, `3:hint_group_role`, `4:hint_org_name`, `5:hint_inst_name`, `6:hint_gpe`, `7:hint_fac_name`, `8:hint_loc_generic`, `9:hint_weapon`, `10:hint_vehicle`, `11:hint_substance`, `12:hint_food`, `13:hint_infra`, `14:hint_tool`, `15:hint_object_generic`, `16:hint_object_name`, `17:hint_event_nominal`, `18:hint_event_named`, `19:hint_time_date`, `20:hint_time_clock`, `21:hint_time_duration`, `22:hint_measure`, `23:hint_percentage`, `24:hint_count`, `25:hint_money`, `26:hint_rate`, `27:hint_work_of_art`, `28:hint_law`, `29:hint_document`, `30:hint_disease`, `31:hint_language`, `32:hint_inst_role`, `33:hint_doctrine`, `34:hint_state`, `35:hint_notion`, `36:hint_work_generic`, `37:hint_field`

### Coarse → fine

- `PER` → `hint_person_name`, `hint_person_role`, `hint_norp`, `hint_group_role`
- `LOC` → `hint_gpe`, `hint_fac_name`, `hint_loc_generic`, `hint_infra`
- `ORG` → `hint_org_name`, `hint_inst_name`, `hint_inst_role`
- `TIME` → `hint_time_date`, `hint_time_clock`, `hint_time_duration`
- `EVENT` → `hint_event_nominal`, `hint_event_named`
- `OBJECT` → `hint_weapon`, `hint_vehicle`, `hint_substance`, `hint_food`, `hint_tool`, `hint_object_generic`, `hint_object_name`
- `VALUE` → `hint_measure`, `hint_percentage`, `hint_count`, `hint_money`, `hint_rate`
- `WORK` → `hint_work_of_art`, `hint_law`, `hint_document`, `hint_work_generic`
- `ABSTRACT` → `hint_disease`, `hint_language`, `hint_doctrine`, `hint_state`, `hint_notion`, `hint_field`

## Syntax / SVO

### Syntax span labels

`0:verb_trigger`, `1:pron_subj`, `2:pron_obj`

### Role head — 12 labels, primary

`0:SUBJECT`, `1:OBJECT`, `2:OBLIQUE`, `3:OBLIQUE_AGENT`, `4:OBLIQUE_CAUSE`, `5:APPOS`, `6:NONE`, `7:OBLIQUE_ADVERSARY`, `8:OBLIQUE_BENEFICIARY`, `9:OBLIQUE_COMITATIVE`, `10:OBLIQUE_DOMAIN`, `11:OBLIQUE_SOURCE`

The 12-label role head is currently the primary SVO role classifier. It already encodes `SUBJECT`, `OBJECT`, `APPOS`, generic `OBLIQUE`, and the fine `OBLIQUE_*` roles.

### Auxiliary role coarse

`0:SUBJ`, `1:OBJ`, `2:OBLIQ`, `3:APPOS`, `4:OTHER`

`role_coarse` is kept for diagnostics/cascade experiments. Some training configs set its lambda to `0.0` when the 12-label `role` head is sufficient.

### Auxiliary oblique fine

`0:OBLIQUE`, `1:OBLIQUE_AGENT`, `2:OBLIQUE_CAUSE`, `3:OBLIQUE_ADVERSARY`, `4:OBLIQUE_BENEFICIARY`, `5:OBLIQUE_COMITATIVE`, `6:OBLIQUE_DOMAIN`, `7:OBLIQUE_SOURCE`, `8:OBLIQUE_TIME`, `9:OBLIQUE_LOC`

`role_oblique` is an auxiliary/cascade head conditioned on oblique spans. Some training configs set its lambda to `0.0` to avoid redundant loss budget.

## Verb taxonomy

### Verb family

`0:Causality`, `1:Cognition`, `2:Communication`, `3:Conflict`, `4:Movement`, `5:OTHER`, `6:Perception`, `7:Possession`, `8:Relation`, `9:Social`, `10:State_Change`, `11:Temporal`

### Verb family fine

`0:Achat`, `1:Annonce`, `2:Appartenance`, `3:Cognitive`, `4:Combat`, `5:Concerne`, `6:Contenu`, `7:Creation`, `8:Croyance`, `9:Debut`, `10:Decision`, `11:Demande`, `12:Deplacement`, `13:Destruction`, `14:Don`, `15:Duree`, `16:Ecrit`, `17:Election`, `18:Fin`, `19:Intention`, `20:Jugement`, `21:Legislation`, `22:Lien`, `23:Negatif`, `24:Negociation`, `25:Nomination`, `26:OTHER`, `27:Opposition`, `28:Permission`, `29:Positif`, `30:Reponse`, `31:Savoir`, `32:Sensorielle`, `33:Transformation`, `34:Transport`, `35:Vente`, `36:Visuelle`, `37:Voyage`

### Verb family → fine

- `Causality` → `Negatif`, `Positif`, `Transformation`, `Destruction`, `Creation`
- `Cognition` → `Cognitive`, `Croyance`, `Decision`, `Intention`, `Savoir`, `Jugement`
- `Communication` → `Annonce`, `Demande`, `Ecrit`, `Reponse`, `Contenu`, `Negociation`
- `Conflict` → `Combat`, `Opposition`, `Negatif`
- `Movement` → `Deplacement`, `Transport`, `Voyage`
- `OTHER` → `OTHER`, `Concerne`, `Lien`
- `Perception` → `Sensorielle`, `Visuelle`, `Cognitive`
- `Possession` → `Achat`, `Appartenance`, `Don`, `Vente`
- `Relation` → `Appartenance`, `Lien`, `Concerne`
- `Social` → `Election`, `Legislation`, `Nomination`, `Negociation`, `Permission`
- `State_Change` → `Debut`, `Fin`, `Transformation`, `Nomination`
- `Temporal` → `Debut`, `Duree`, `Fin`

### Verb polarity / aspect / source

- Polarity: `0:NEGATIVE`, `1:NEUTRAL`, `2:POSITIVE`
- Aspect: `0:DURATIF`, `1:PONCTUEL`
- Source: `0:DIRECT`, `1:HYPOTHETICAL`, `2:REPORTED`

## Morphology / modality

- Voice: `0:active`, `1:passive`
- Certainty: `0:certain`, `1:modal`, `2:denied`
- Gender: `0:M`, `1:F`
- Number: `0:SG`, `1:PL`
- Person: `0:1`, `1:2`, `2:3`

## Maintenance checklist

When changing taxonomy:

1. Update `training/multi-head/labels.py` only.
2. Run:

```zsh
cd pimpmyrag
source training/multi-head/venv/bin/activate
python3 training/multi-head/export_taxonomy.py
```

3. Commit `labels.py`, `docs/taxonomy.json`, `docs/taxonomy.schema.json`, and `docs/TAXONOMY.md` together.
