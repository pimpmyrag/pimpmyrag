#!/usr/bin/env python3
"""
Boost des classes faibles : génère des phrases pour remonter toutes les classes < 250.
Cible : hint_language, hint_disease, hint_law, hint_concept, hint_work_of_art,
        hint_rate, hint_tool, hint_food, hint_object_name
"""
from __future__ import annotations
import json, argparse, random
from typing import List, Dict, Tuple

SENTENCES: List[Dict] = []

def _add(text: str, annotations: List[Tuple[str, str]]):
    spans = []
    used = set()
    for label, surface in annotations:
        s = 0
        while True:
            idx = text.find(surface, s)
            assert idx != -1, f"'{surface}' not found in: {text}"
            if idx not in used:
                used.add(idx)
                break
            s = idx + 1
        spans.append({"label": label, "start": idx, "end": idx + len(surface), "text": surface})
    SENTENCES.append({"text": text, "spans": spans})

# ═══════════════════════════════════════════════════════════════
#  hint_language  (~130 phrases en plus)
# ═══════════════════════════════════════════════════════════════

_add("Le picard est un dialecte d'oïl encore parlé dans le nord de la France.", [
    ("hint_language", "picard"), ("hint_gpe", "France")])
_add("Le créole réunionnais est issu du français et de langues malgaches.", [
    ("hint_language", "créole réunionnais"), ("hint_language", "français")])
_add("Le danois est la langue officielle du Danemark et du Groenland.", [
    ("hint_language", "danois"), ("hint_gpe", "Danemark"), ("hint_gpe", "Groenland")])
_add("Le suédois est compris par les Norvégiens et les Danois.", [
    ("hint_language", "suédois"), ("hint_norp", "Norvégiens"), ("hint_norp", "Danois")])
_add("L'islandais a très peu évolué depuis le vieux norrois médiéval.", [
    ("hint_language", "islandais"), ("hint_language", "vieux norrois")])
_add("Le tok pisin est une langue créole officielle de la Papouasie-Nouvelle-Guinée.", [
    ("hint_language", "tok pisin"), ("hint_gpe", "Papouasie-Nouvelle-Guinée")])
_add("Le lingala est une langue véhiculaire du bassin du Congo.", [
    ("hint_language", "lingala"), ("hint_gpe", "Congo")])
_add("L'ukrainien est la langue officielle de l'Ukraine depuis 1991.", [
    ("hint_language", "ukrainien"), ("hint_gpe", "Ukraine"), ("hint_time_date", "1991")])
_add("Le biélorusse est une langue slave orientale menacée de disparition.", [
    ("hint_language", "biélorusse")])
_add("Le bulgare est la plus ancienne langue slave attestée par écrit.", [
    ("hint_language", "bulgare")])
_add("Le macédonien est la langue officielle de la Macédoine du Nord.", [
    ("hint_language", "macédonien"), ("hint_gpe", "Macédoine du Nord")])
_add("Le polonais est la deuxième langue slave la plus parlée après le russe.", [
    ("hint_language", "polonais"), ("hint_language", "russe")])
_add("Le letton est une langue balte proche du lituanien.", [
    ("hint_language", "letton"), ("hint_language", "lituanien")])
_add("L'estonien est plus proche du finnois que des langues baltes.", [
    ("hint_language", "estonien"), ("hint_language", "finnois")])
_add("Le hongrois est une langue finno-ougrienne sans parenté avec ses voisines.", [
    ("hint_language", "hongrois")])
_add("L'albanais est une branche unique de la famille indo-européenne.", [
    ("hint_language", "albanais")])
_add("Le népalais est la langue officielle du Népal.", [
    ("hint_language", "népalais"), ("hint_gpe", "Népal")])
_add("Le cingalais est la langue majoritaire du Sri Lanka.", [
    ("hint_language", "cingalais"), ("hint_gpe", "Sri Lanka")])
_add("Le pendjabi est la langue la plus parlée au Pakistan.", [
    ("hint_language", "pendjabi"), ("hint_gpe", "Pakistan")])
_add("Le télougou est une langue dravidienne parlée dans le sud-est de l'Inde.", [
    ("hint_language", "télougou"), ("hint_gpe", "Inde")])
_add("Le marathi est la langue officielle de l'État du Maharashtra en Inde.", [
    ("hint_language", "marathi"), ("hint_gpe", "Maharashtra"), ("hint_gpe", "Inde")])
_add("Le kannada est une langue dravidienne écrite avec son propre alphabet.", [
    ("hint_language", "kannada")])
_add("Le malayalam est la langue officielle du Kerala.", [
    ("hint_language", "malayalam"), ("hint_gpe", "Kerala")])
_add("Le ouïghour est une langue turcique parlée dans le Xinjiang en Chine.", [
    ("hint_language", "ouïghour"), ("hint_gpe", "Xinjiang"), ("hint_gpe", "Chine")])
_add("Le kazakh est la langue officielle du Kazakhstan.", [
    ("hint_language", "kazakh"), ("hint_gpe", "Kazakhstan")])
_add("L'ouzbek est la langue officielle de l'Ouzbékistan.", [
    ("hint_language", "ouzbek"), ("hint_gpe", "Ouzbékistan")])
_add("Le turkmène est une langue turcique parlée au Turkménistan.", [
    ("hint_language", "turkmène"), ("hint_gpe", "Turkménistan")])
_add("Le kirghiz est la langue nationale du Kirghizistan.", [
    ("hint_language", "kirghiz"), ("hint_gpe", "Kirghizistan")])
_add("Le tadjik est une variante du persan écrite en alphabet cyrillique.", [
    ("hint_language", "tadjik"), ("hint_language", "persan")])
_add("Le lao est une langue tonale proche du thaï parlée au Laos.", [
    ("hint_language", "lao"), ("hint_language", "thaï"), ("hint_gpe", "Laos")])
_add("Le khmer est la seule langue austroasiatique majeure non tonale.", [
    ("hint_language", "khmer")])
_add("Le malais est la base de l'indonésien et du filipino.", [
    ("hint_language", "malais"), ("hint_language", "indonésien"), ("hint_language", "filipino")])
_add("Le javanais est la langue régionale la plus parlée d'Indonésie.", [
    ("hint_language", "javanais"), ("hint_gpe", "Indonésie")])
_add("Le maori est la langue autochtone de la Nouvelle-Zélande.", [
    ("hint_language", "maori"), ("hint_gpe", "Nouvelle-Zélande")])
_add("Le hawaïen est une langue polynésienne en cours de revitalisation.", [
    ("hint_language", "hawaïen")])
_add("Le samoan est la langue officielle des Samoa.", [
    ("hint_language", "samoan"), ("hint_gpe", "Samoa")])
_add("Le tongan est la langue officielle du royaume des Tonga.", [
    ("hint_language", "tongan"), ("hint_gpe", "Tonga")])
_add("Le frison est la langue germanique la plus proche de l'anglais.", [
    ("hint_language", "frison"), ("hint_language", "anglais")])
_add("Le scots est une langue germanique parlée en Écosse aux côtés de l'anglais.", [
    ("hint_language", "scots"), ("hint_gpe", "Écosse"), ("hint_language", "anglais")])
_add("Le catalan est aussi parlé dans les îles Baléares et à Valence.", [
    ("hint_language", "catalan"), ("hint_gpe", "Baléares"), ("hint_gpe", "Valence")])
_add("Le galicien est une langue romane proche du portugais parlée en Espagne.", [
    ("hint_language", "galicien"), ("hint_language", "portugais"), ("hint_gpe", "Espagne")])
_add("L'asturien est une langue romane en voie de normalisation en Espagne.", [
    ("hint_language", "asturien"), ("hint_gpe", "Espagne")])
_add("Le sarde est considéré comme la langue romane la plus conservatrice.", [
    ("hint_language", "sarde")])
_add("Le frioulan est une langue romane parlée dans le nord-est de l'Italie.", [
    ("hint_language", "frioulan"), ("hint_gpe", "Italie")])
_add("Le romanche est la quatrième langue nationale de la Suisse.", [
    ("hint_language", "romanche"), ("hint_gpe", "Suisse")])
_add("Le sorabe est une langue slave minoritaire parlée en Allemagne.", [
    ("hint_language", "sorabe"), ("hint_gpe", "Allemagne")])
_add("Le ladino est la langue judéo-espagnole des Séfarades.", [
    ("hint_language", "ladino"), ("hint_norp", "Séfarades")])
_add("Le féroïen est la langue officielle des îles Féroé.", [
    ("hint_language", "féroïen"), ("hint_gpe", "îles Féroé")])
_add("Le yoruba est une langue tonale parlée par 45 millions de personnes au Nigeria.", [
    ("hint_language", "yoruba"), ("hint_quantity", "45 millions"), ("hint_gpe", "Nigeria")])
_add("L'igbo est une langue majeure du sud-est du Nigeria.", [
    ("hint_language", "igbo"), ("hint_gpe", "Nigeria")])
_add("Le bambara est la langue véhiculaire la plus répandue au Mali.", [
    ("hint_language", "bambara"), ("hint_gpe", "Mali")])
_add("Le peul est une langue transfrontalière parlée de la Mauritanie au Cameroun.", [
    ("hint_language", "peul"), ("hint_gpe", "Mauritanie"), ("hint_gpe", "Cameroun")])
_add("Le mooré est la langue la plus parlée au Burkina Faso.", [
    ("hint_language", "mooré"), ("hint_gpe", "Burkina Faso")])
_add("Le kinyarwanda est la langue nationale du Rwanda.", [
    ("hint_language", "kinyarwanda"), ("hint_gpe", "Rwanda")])
_add("Le kirundi est la langue nationale du Burundi, proche du kinyarwanda.", [
    ("hint_language", "kirundi"), ("hint_gpe", "Burundi"), ("hint_language", "kinyarwanda")])
_add("Le chichewa est la langue nationale du Malawi.", [
    ("hint_language", "chichewa"), ("hint_gpe", "Malawi")])
_add("Le sesotho est l'une des onze langues officielles de l'Afrique du Sud.", [
    ("hint_language", "sesotho"), ("hint_gpe", "Afrique du Sud")])
_add("Le tswana est une langue bantoue parlée au Botswana et en Afrique du Sud.", [
    ("hint_language", "tswana"), ("hint_gpe", "Botswana"), ("hint_gpe", "Afrique du Sud")])
_add("Le xhosa se distingue par ses consonnes à clics.", [("hint_language", "xhosa")])
_add("Le ndébélé est l'une des langues bantoues du Zimbabwe.", [
    ("hint_language", "ndébélé"), ("hint_gpe", "Zimbabwe")])
_add("Le malgache est une langue austronésienne parlée à Madagascar.", [
    ("hint_language", "malgache"), ("hint_gpe", "Madagascar")])
_add("Le sango est la langue nationale de la République centrafricaine.", [
    ("hint_language", "sango"), ("hint_gpe", "République centrafricaine")])
_add("Le kituba est un créole bantou servant de lingua franca au Congo.", [
    ("hint_language", "kituba"), ("hint_gpe", "Congo")])
_add("Le tshiluba est l'une des quatre langues nationales de la RDC.", [
    ("hint_language", "tshiluba")])
_add("Le guarani est parlé par plus de 90 % de la population paraguayenne.", [
    ("hint_language", "guarani"), ("hint_percentage", "90 %"), ("hint_norp", "paraguayenne")])
_add("Le quichua équatorien est une variante du quechua andin.", [
    ("hint_language", "quichua équatorien"), ("hint_language", "quechua")])
_add("Le maya yucatèque est encore parlé par 800 000 personnes au Mexique.", [
    ("hint_language", "maya yucatèque"), ("hint_quantity", "800 000"), ("hint_gpe", "Mexique")])
_add("Le navajo est la langue amérindienne la plus parlée aux États-Unis.", [
    ("hint_language", "navajo"), ("hint_gpe", "États-Unis")])
_add("Le cherokee est la seule langue amérindienne possédant un syllabaire propre.", [
    ("hint_language", "cherokee")])
_add("Le inuktitut est la langue des Inuits du nord du Canada.", [
    ("hint_language", "inuktitut"), ("hint_norp", "Inuits"), ("hint_gpe", "Canada")])
_add("Le cri est une langue algonquienne parlée dans les provinces canadiennes.", [
    ("hint_language", "cri"), ("hint_norp", "canadiennes")])
_add("Le tahitien est la langue autochtone de la Polynésie française.", [
    ("hint_language", "tahitien"), ("hint_gpe", "Polynésie française")])
_add("Le drehu est l'une des langues kanak de la Nouvelle-Calédonie.", [
    ("hint_language", "drehu"), ("hint_gpe", "Nouvelle-Calédonie")])
_add("Le alsacien est un dialecte alémanique encore vivant en Alsace.", [
    ("hint_language", "alsacien"), ("hint_gpe", "Alsace")])
_add("Le gascon est une variété d'occitan parlée dans le sud-ouest de la France.", [
    ("hint_language", "gascon"), ("hint_language", "occitan"), ("hint_gpe", "France")])
_add("Le francoprovençal est parlé en Savoie, en Suisse romande et au Val d'Aoste.", [
    ("hint_language", "francoprovençal"), ("hint_gpe", "Savoie")])
_add("Le dialecte alsacien est inscrit au patrimoine culturel immatériel de la France.", [
    ("hint_language", "dialecte alsacien"), ("hint_gpe", "France")])
_add("Le sicilien est parfois considéré comme une langue à part entière distincte de l'italien.", [
    ("hint_language", "sicilien"), ("hint_language", "italien")])
_add("Le napolitain est parlé par environ 5 millions de personnes en Italie du Sud.", [
    ("hint_language", "napolitain"), ("hint_quantity", "5 millions"), ("hint_gpe", "Italie du Sud")])
_add("Le vénitien est une langue romane vivace en Vénétie.", [
    ("hint_language", "vénitien"), ("hint_gpe", "Vénétie")])
_add("Le plattdüütsch est un ensemble de dialectes bas-allemands du nord de l'Allemagne.", [
    ("hint_language", "plattdüütsch"), ("hint_gpe", "Allemagne")])
_add("Le bavarois est parlé en Bavière et en Autriche.", [
    ("hint_language", "bavarois"), ("hint_gpe", "Bavière"), ("hint_gpe", "Autriche")])

# ═══════════════════════════════════════════════════════════════
#  hint_disease  (~100 phrases en plus)
# ═══════════════════════════════════════════════════════════════

_add("La mononucléose est une infection virale fréquente chez les jeunes adultes.", [
    ("hint_disease", "mononucléose")])
_add("Le zona ophtalmique peut entraîner des lésions graves de l'œil.", [
    ("hint_disease", "zona ophtalmique")])
_add("La typhoïde reste un problème de santé publique dans les pays en développement.", [
    ("hint_disease", "typhoïde")])
_add("La scarlatine est causée par un streptocoque du groupe A.", [
    ("hint_disease", "scarlatine")])
_add("L'eczéma touche environ 20 % des enfants dans les pays industrialisés.", [
    ("hint_disease", "eczéma"), ("hint_percentage", "20 %")])
_add("La maladie de Bechterew provoque une raideur progressive de la colonne vertébrale.", [
    ("hint_disease", "maladie de Bechterew")])
_add("Le syndrome de Sjögren est une maladie auto-immune attaquant les glandes exocrines.", [
    ("hint_disease", "syndrome de Sjögren")])
_add("La maladie de Fabry est une maladie lysosomale liée au chromosome X.", [
    ("hint_disease", "maladie de Fabry")])
_add("Le syndrome de Tourette se manifeste par des tics moteurs et vocaux involontaires.", [
    ("hint_disease", "syndrome de Tourette")])
_add("La maladie de Basedow est la cause la plus fréquente d'hyperthyroïdie.", [
    ("hint_disease", "maladie de Basedow")])
_add("Le syndrome de Rett touche presque exclusivement les filles.", [
    ("hint_disease", "syndrome de Rett")])
_add("La maladie de Hashimoto provoque une hypothyroïdie chronique.", [
    ("hint_disease", "maladie de Hashimoto")])
_add("Le rachitisme est dû à une carence en vitamine D chez l'enfant.", [
    ("hint_disease", "rachitisme")])
_add("Le scorbut était la maladie redoutée des marins au long cours.", [
    ("hint_disease", "scorbut")])
_add("La toxoplasmose est dangereuse pour les femmes enceintes non immunisées.", [
    ("hint_disease", "toxoplasmose")])
_add("La listériose est une infection alimentaire potentiellement mortelle.", [
    ("hint_disease", "listériose")])
_add("La salmonellose provoque des gastro-entérites parfois sévères.", [
    ("hint_disease", "salmonellose")])
_add("L'amylose est une maladie rare causée par le dépôt de protéines anormales.", [
    ("hint_disease", "amylose")])
_add("Le syndrome de Klinefelter est une anomalie chromosomique touchant les hommes.", [
    ("hint_disease", "syndrome de Klinefelter")])
_add("La maladie de Paget affecte le renouvellement osseux chez les personnes âgées.", [
    ("hint_disease", "maladie de Paget")])
_add("Le syndrome de l'X fragile est la première cause héréditaire de déficience intellectuelle.", [
    ("hint_disease", "syndrome de l'X fragile")])
_add("La cataracte est la première cause de cécité réversible dans le monde.", [
    ("hint_disease", "cataracte")])
_add("L'acné est une affection cutanée qui touche la majorité des adolescents.", [
    ("hint_disease", "acné")])
_add("La goutte est causée par une accumulation d'acide urique dans les articulations.", [
    ("hint_disease", "goutte")])
_add("L'arthrose est la maladie articulaire la plus fréquente.", [
    ("hint_disease", "arthrose")])
_add("La polyarthrite rhumatoïde est une maladie auto-immune invalidante.", [
    ("hint_disease", "polyarthrite rhumatoïde")])
_add("Le mélanome est le cancer de la peau le plus agressif.", [
    ("hint_disease", "mélanome")])
_add("La leucémie est un cancer du sang qui touche aussi les enfants.", [
    ("hint_disease", "leucémie")])
_add("Le lymphome de Hodgkin est un cancer du système lymphatique souvent curable.", [
    ("hint_disease", "lymphome de Hodgkin")])
_add("La cirrhose du foie est souvent liée à l'alcoolisme chronique.", [
    ("hint_disease", "cirrhose du foie")])
_add("L'insuffisance rénale chronique nécessite parfois le recours à la dialyse.", [
    ("hint_disease", "insuffisance rénale chronique")])
_add("L'épilepsie se manifeste par des crises convulsives récurrentes.", [
    ("hint_disease", "épilepsie")])
_add("La migraine touche environ 15 % de la population mondiale.", [
    ("hint_disease", "migraine"), ("hint_percentage", "15 %")])
_add("Le trouble bipolaire alterne des phases maniaques et dépressives.", [
    ("hint_disease", "trouble bipolaire")])
_add("La dépression est la principale cause d'incapacité dans le monde selon l'OMS.", [
    ("hint_disease", "dépression"), ("hint_org_name", "OMS")])
_add("Le trouble obsessionnel-compulsif perturbe le quotidien par des pensées intrusives.", [
    ("hint_disease", "trouble obsessionnel-compulsif")])
_add("Le syndrome de stress post-traumatique peut survenir après un événement violent.", [
    ("hint_disease", "syndrome de stress post-traumatique")])
_add("La dyslexie est un trouble spécifique de l'apprentissage de la lecture.", [
    ("hint_disease", "dyslexie")])
_add("La maladie de Creutzfeldt-Jakob est une maladie à prions toujours mortelle.", [
    ("hint_disease", "maladie de Creutzfeldt-Jakob")])
_add("La rage est transmise par la morsure d'un animal infecté.", [
    ("hint_disease", "rage")])
_add("Le trachome est la première cause infectieuse de cécité évitable.", [
    ("hint_disease", "trachome")])
_add("L'onchocercose, ou cécité des rivières, est transmise par des mouches noires.", [
    ("hint_disease", "onchocercose")])
_add("La filariose lymphatique provoque un gonflement spectaculaire des membres.", [
    ("hint_disease", "filariose lymphatique")])
_add("Le choléra se propage par l'eau contaminée et provoque des diarrhées sévères.", [
    ("hint_disease", "choléra")])
_add("La brucellose est une zoonose transmise par le lait non pasteurisé.", [
    ("hint_disease", "brucellose")])
_add("La leptospirose est transmise par l'urine de rongeurs infectés.", [
    ("hint_disease", "leptospirose")])
_add("L'histoplasmose est une mycose pulmonaire causée par un champignon du sol.", [
    ("hint_disease", "histoplasmose")])
_add("Le cancer colorectal est le troisième cancer le plus fréquent au monde.", [
    ("hint_disease", "cancer colorectal")])
_add("Le cancer du sein touche une femme sur huit au cours de sa vie.", [
    ("hint_disease", "cancer du sein")])
_add("Le cancer de la prostate est le cancer le plus fréquent chez l'homme.", [
    ("hint_disease", "cancer de la prostate")])
_add("La grippe porcine H1N1 a provoqué une pandémie en 2009.", [
    ("hint_disease", "grippe porcine H1N1"), ("hint_event_nominal", "pandémie"), ("hint_time_date", "2009")])
_add("Le virus Nipah est un pathogène à potentiel pandémique identifié en Malaisie.", [
    ("hint_disease", "virus Nipah"), ("hint_gpe", "Malaisie")])

# ═══════════════════════════════════════════════════════════════
#  hint_law  (~100 phrases en plus)
# ═══════════════════════════════════════════════════════════════

_add("La loi Malraux protège les secteurs sauvegardés des centres-villes historiques.", [
    ("hint_law", "loi Malraux")])
_add("Le traité de Francfort de 1871 imposa l'annexion de l'Alsace-Moselle par l'Allemagne.", [
    ("hint_law", "traité de Francfort"), ("hint_time_date", "1871"), ("hint_gpe", "Alsace-Moselle"), ("hint_gpe", "Allemagne")])
_add("La loi Pleven de 1972 est la première loi antiraciste française.", [
    ("hint_law", "loi Pleven"), ("hint_time_date", "1972"), ("hint_norp", "française")])
_add("Le traité de Verdun de 843 partagea l'Empire carolingien en trois royaumes.", [
    ("hint_law", "traité de Verdun"), ("hint_time_date", "843")])
_add("Le Code Napoléon a influencé les systèmes juridiques de dizaines de pays.", [
    ("hint_law", "Code Napoléon")])
_add("La Convention de Rotterdam régule le commerce des produits chimiques dangereux.", [
    ("hint_law", "Convention de Rotterdam")])
_add("Le traité de Presbourg de 1805 sanctionna la victoire d'Austerlitz.", [
    ("hint_law", "traité de Presbourg"), ("hint_time_date", "1805")])
_add("La loi Montagne de 1985 encadre l'aménagement et la protection des zones de montagne.", [
    ("hint_law", "loi Montagne"), ("hint_time_date", "1985")])
_add("L'accord AUKUS de 2021 a provoqué une crise diplomatique avec la France.", [
    ("hint_law", "accord AUKUS"), ("hint_time_date", "2021"), ("hint_gpe", "France")])
_add("Le traité de Münster de 1648 est l'un des traités de Westphalie.", [
    ("hint_law", "traité de Münster"), ("hint_time_date", "1648")])
_add("La loi de modernisation de la santé de 2016 a instauré le tiers payant généralisé.", [
    ("hint_law", "loi de modernisation de la santé"), ("hint_time_date", "2016")])
_add("La Bulle d'or de 1356 fixa les règles d'élection de l'empereur du Saint-Empire.", [
    ("hint_law", "Bulle d'or"), ("hint_time_date", "1356")])
_add("Le traité de Campo-Formio de 1797 sanctionna les victoires de Bonaparte en Italie.", [
    ("hint_law", "traité de Campo-Formio"), ("hint_time_date", "1797"), ("hint_person_name", "Bonaparte"), ("hint_gpe", "Italie")])
_add("La loi de transition énergétique de 2015 fixe l'objectif de 40 % de renouvelables.", [
    ("hint_law", "loi de transition énergétique"), ("hint_time_date", "2015"), ("hint_percentage", "40 %")])
_add("Le traité de Paris de 1763 mit fin à la guerre de Sept Ans.", [
    ("hint_law", "traité de Paris"), ("hint_time_date", "1763")])
_add("La Convention de Stockholm interdit les polluants organiques persistants.", [
    ("hint_law", "Convention de Stockholm")])
_add("Le traité de Bruxelles de 1948 est l'ancêtre de l'OTAN.", [
    ("hint_law", "traité de Bruxelles"), ("hint_time_date", "1948"), ("hint_org_name", "OTAN")])
_add("La loi de programmation de la recherche de 2020 a augmenté les budgets scientifiques.", [
    ("hint_law", "loi de programmation de la recherche"), ("hint_time_date", "2020")])
_add("Le traité de Nerchinsk de 1689 fixa la frontière entre la Russie et la Chine.", [
    ("hint_law", "traité de Nerchinsk"), ("hint_time_date", "1689"), ("hint_gpe", "Russie"), ("hint_gpe", "Chine")])
_add("La loi Egalim de 2018 encadre les relations entre agriculteurs et grande distribution.", [
    ("hint_law", "loi Egalim"), ("hint_time_date", "2018"), ("hint_group_role", "agriculteurs")])
_add("Le traité de Guadalupe Hidalgo de 1848 céda la Californie aux États-Unis.", [
    ("hint_law", "traité de Guadalupe Hidalgo"), ("hint_time_date", "1848"), ("hint_gpe", "Californie"), ("hint_gpe", "États-Unis")])
_add("Le traité antarctique de 1959 réserve le continent à la recherche scientifique.", [
    ("hint_law", "traité antarctique"), ("hint_time_date", "1959")])
_add("La Convention de Ramsar protège les zones humides d'importance internationale.", [
    ("hint_law", "Convention de Ramsar")])
_add("Le traité de Maastricht impose une limite de déficit public à 3 % du PIB.", [
    ("hint_law", "traité de Maastricht"), ("hint_percentage", "3 %")])
_add("La loi de séparation de 1905 a mis fin au régime concordataire sauf en Alsace-Moselle.", [
    ("hint_law", "loi de séparation"), ("hint_time_date", "1905"), ("hint_gpe", "Alsace-Moselle")])
_add("La Convention de Montréal régule les substances appauvrissant la couche d'ozone.", [
    ("hint_law", "Convention de Montréal")])
_add("Le traité de Bucarest de 1913 mit fin à la deuxième guerre balkanique.", [
    ("hint_law", "traité de Bucarest"), ("hint_time_date", "1913")])
_add("La directive MiFID 2 encadre les marchés financiers en Europe.", [
    ("hint_law", "directive MiFID 2"), ("hint_gpe", "Europe")])
_add("Le décret de Berlin de 1806 instaura le blocus continental contre le Royaume-Uni.", [
    ("hint_law", "décret de Berlin"), ("hint_time_date", "1806"), ("hint_gpe", "Royaume-Uni")])
_add("Le traité de Kanagawa de 1854 ouvrit le Japon au commerce occidental.", [
    ("hint_law", "traité de Kanagawa"), ("hint_time_date", "1854"), ("hint_gpe", "Japon")])
_add("La Convention CITES protège les espèces menacées du commerce international.", [
    ("hint_law", "Convention CITES")])
_add("Le traité START de 1991 réduisit les arsenaux nucléaires américain et soviétique.", [
    ("hint_law", "traité START"), ("hint_time_date", "1991"), ("hint_norp", "américain"), ("hint_norp", "soviétique")])

# ═══════════════════════════════════════════════════════════════
#  hint_concept  (~70 phrases en plus)
# ═══════════════════════════════════════════════════════════════

_add("L'absolutisme concentrait tous les pouvoirs entre les mains du monarque.", [
    ("hint_concept", "absolutisme")])
_add("Le despotisme éclairé combinait pouvoir absolu et réformes inspirées des Lumières.", [
    ("hint_concept", "despotisme éclairé")])
_add("Le bonapartisme se réclame de l'héritage politique de Napoléon.", [
    ("hint_concept", "bonapartisme"), ("hint_person_name", "Napoléon")])
_add("Le gaullisme a profondément marqué la politique française de la Ve République.", [
    ("hint_concept", "gaullisme"), ("hint_norp", "française")])
_add("Le poujadisme incarnait la révolte des petits commerçants dans les années 1950.", [
    ("hint_concept", "poujadisme"), ("hint_time_date", "années 1950")])
_add("Le thatchérisme a libéralisé l'économie britannique dans les années 1980.", [
    ("hint_concept", "thatchérisme"), ("hint_norp", "britannique"), ("hint_time_date", "années 1980")])
_add("Le reaganisme a réduit le rôle de l'État fédéral dans l'économie américaine.", [
    ("hint_concept", "reaganisme"), ("hint_norp", "américaine")])
_add("Le péronisme reste une force politique majeure en Argentine.", [
    ("hint_concept", "péronisme"), ("hint_gpe", "Argentine")])
_add("Le maccarthysme désigne la chasse aux communistes aux États-Unis dans les années 1950.", [
    ("hint_concept", "maccarthysme"), ("hint_norp", "communistes"), ("hint_gpe", "États-Unis"), ("hint_time_date", "années 1950")])
_add("Le totalitarisme se caractérise par un contrôle absolu de l'État sur la société.", [
    ("hint_concept", "totalitarisme")])
_add("L'autoritarisme restreint les libertés politiques sans contrôler toute la vie sociale.", [
    ("hint_concept", "autoritarisme")])
_add("Le centralisme démocratique est le principe d'organisation des partis léninistes.", [
    ("hint_concept", "centralisme démocratique"), ("hint_norp", "léninistes")])
_add("Le trotskisme prône la révolution permanente et internationale.", [
    ("hint_concept", "trotskisme")])
_add("Le stalinisme a imposé la planification économique et la terreur politique en URSS.", [
    ("hint_concept", "stalinisme")])
_add("Le titoisme a permis à la Yougoslavie de s'affranchir de la tutelle soviétique.", [
    ("hint_concept", "titoisme"), ("hint_gpe", "Yougoslavie"), ("hint_norp", "soviétique")])
_add("Le négationnisme nie ou minimise les crimes contre l'humanité reconnus par l'histoire.", [
    ("hint_concept", "négationnisme")])
_add("Le révisionnisme cherche à remettre en question les interprétations historiques établies.", [
    ("hint_concept", "révisionnisme")])
_add("La géopolitique étudie les rapports de force entre les territoires et les puissances.", [
    ("hint_concept", "géopolitique")])
_add("Le bipartisme organise la vie politique autour de deux grands partis dominants.", [
    ("hint_concept", "bipartisme")])
_add("Le présidentialisme concentre le pouvoir exécutif entre les mains du chef de l'État.", [
    ("hint_concept", "présidentialisme")])
_add("Le parlementarisme donne la prééminence au pouvoir législatif sur l'exécutif.", [
    ("hint_concept", "parlementarisme")])
_add("La subsidiarité attribue les décisions au niveau le plus proche des citoyens.", [
    ("hint_concept", "subsidiarité")])
_add("L'abstentionnisme est un phénomène croissant dans les démocraties occidentales.", [
    ("hint_concept", "abstentionnisme")])
_add("Le corporatisme organise la société en corps professionnels.", [
    ("hint_concept", "corporatisme")])
_add("Le clientélisme échange des faveurs contre un soutien politique.", [
    ("hint_concept", "clientélisme")])
_add("Le néocolonialisme désigne la persistance de rapports de domination après la décolonisation.", [
    ("hint_concept", "néocolonialisme")])
_add("Le tiers-mondisme défend les intérêts des pays du Sud face aux puissances occidentales.", [
    ("hint_concept", "tiers-mondisme")])
_add("Le non-alignement refusait de choisir entre le bloc américain et le bloc soviétique.", [
    ("hint_concept", "non-alignement"), ("hint_norp", "américain"), ("hint_norp", "soviétique")])
_add("Le multiculturalisme valorise la coexistence de différentes cultures au sein d'une société.", [
    ("hint_concept", "multiculturalisme")])
_add("L'assimilationnisme vise l'intégration complète des minorités dans la culture dominante.", [
    ("hint_concept", "assimilationnisme")])

# ═══════════════════════════════════════════════════════════════
#  hint_rate  (~80 phrases)
# ═══════════════════════════════════════════════════════════════

_add("Le taux d'intérêt directeur de la BCE a été relevé à 4,5 % en septembre.", [
    ("hint_rate", "4,5 %"), ("hint_org_name", "BCE")])
_add("L'inflation a atteint un rythme annuel de 6,1 % dans la zone euro.", [
    ("hint_rate", "6,1 %")])
_add("Le PIB a reculé de 0,2 % au troisième trimestre.", [("hint_rate", "0,2 %")])
_add("Le taux de chômage s'établit à 7,3 % de la population active en France.", [
    ("hint_rate", "7,3 %"), ("hint_gpe", "France")])
_add("La croissance économique devrait atteindre 1,4 % cette année selon le FMI.", [
    ("hint_rate", "1,4 %"), ("hint_org_name", "FMI")])
_add("Le rendement du livret A a été porté à 3 % au 1er février.", [
    ("hint_rate", "3 %")])
_add("Le taux de participation aux élections a chuté à 42 % au second tour.", [
    ("hint_rate", "42 %")])
_add("Le taux de natalité est tombé à 1,68 enfant par femme en Europe.", [
    ("hint_rate", "1,68"), ("hint_gpe", "Europe")])
_add("Le taux de mortalité infantile est de 3,2 pour mille dans les pays développés.", [
    ("hint_rate", "3,2 pour mille")])
_add("Le dollar a perdu 1,5 % face à l'euro cette semaine.", [
    ("hint_rate", "1,5 %")])
_add("Le taux d'alphabétisation atteint 99,7 % au Japon.", [
    ("hint_rate", "99,7 %"), ("hint_gpe", "Japon")])
_add("La dette publique française représente 112 % du PIB en 2024.", [
    ("hint_rate", "112 %"), ("hint_norp", "française"), ("hint_time_date", "2024")])
_add("Le chômage des jeunes dépasse 25 % en Espagne.", [
    ("hint_rate", "25 %"), ("hint_gpe", "Espagne")])
_add("Le taux de pauvreté a augmenté de 0,8 point pour atteindre 14,5 %.", [
    ("hint_rate", "0,8 point"), ("hint_rate", "14,5 %")])
_add("Le taux de réussite au baccalauréat a atteint 91,3 % cette année.", [
    ("hint_rate", "91,3 %")])
_add("Le taux d'occupation des hôtels a bondi de 12 points par rapport à l'an dernier.", [
    ("hint_rate", "12 points")])
_add("Le PIB chinois a progressé de 5,2 % en glissement annuel.", [
    ("hint_rate", "5,2 %"), ("hint_norp", "chinois")])
_add("Le prix du pétrole a grimpé de 8 % en une semaine.", [("hint_rate", "8 %")])
_add("Le taux de remplissage des TGV dépasse 80 % le vendredi soir.", [("hint_rate", "80 %")])
_add("Le salaire minimum a été revalorisé de 2,2 % au 1er janvier.", [("hint_rate", "2,2 %")])
_add("Le taux de positivité des tests Covid est descendu sous 5 %.", [
    ("hint_rate", "5 %"), ("hint_disease", "Covid")])
_add("Le rendement obligataire français a atteint 3,15 % sur dix ans.", [
    ("hint_rate", "3,15 %"), ("hint_norp", "français")])
_add("Le taux d'emploi des 25-54 ans est de 82 % en France.", [
    ("hint_rate", "82 %"), ("hint_gpe", "France")])
_add("L'indice de confiance des ménages a progressé de 3 points en mars.", [
    ("hint_rate", "3 points")])
_add("Le taux de conversion des visiteurs en clients atteint 4,7 % sur le site.", [
    ("hint_rate", "4,7 %")])

# ═══════════════════════════════════════════════════════════════
#  hint_tool  (~70 phrases)
# ═══════════════════════════════════════════════════════════════

_add("Le stéthoscope a été inventé par Laennec en 1816.", [
    ("hint_tool", "stéthoscope"), ("hint_person_name", "Laennec"), ("hint_time_date", "1816")])
_add("Le microscope électronique permet d'observer des structures à l'échelle nanométrique.", [
    ("hint_tool", "microscope électronique")])
_add("Le radar est utilisé pour détecter les avions et les navires à distance.", [
    ("hint_tool", "radar")])
_add("Le sonar permet de cartographier les fonds marins par ondes acoustiques.", [
    ("hint_tool", "sonar")])
_add("Le lidar mesure les distances par impulsions laser.", [("hint_tool", "lidar")])
_add("Le scanner IRM permet d'obtenir des images détaillées des organes internes.", [
    ("hint_tool", "scanner IRM")])
_add("Le spectromètre de masse identifie les molécules par leur rapport masse/charge.", [
    ("hint_tool", "spectromètre de masse")])
_add("Le sismographe enregistre les vibrations du sol lors des tremblements de terre.", [
    ("hint_tool", "sismographe")])
_add("Le thermomètre a été perfectionné par Fahrenheit au XVIIIe siècle.", [
    ("hint_tool", "thermomètre"), ("hint_person_name", "Fahrenheit"), ("hint_time_date", "XVIIIe siècle")])
_add("Le baromètre mesure la pression atmosphérique pour prévoir la météo.", [
    ("hint_tool", "baromètre")])
_add("Le télescope de Hubble a révolutionné l'astronomie depuis son lancement en 1990.", [
    ("hint_tool", "télescope de Hubble"), ("hint_time_date", "1990")])
_add("Le GPS permet de se géolocaliser n'importe où sur Terre avec une précision métrique.", [
    ("hint_tool", "GPS")])
_add("Le drone est utilisé pour la surveillance, la cartographie et la livraison.", [
    ("hint_tool", "drone")])
_add("Le défibrillateur peut sauver une vie en cas d'arrêt cardiaque.", [
    ("hint_tool", "défibrillateur")])
_add("L'oscilloscope permet de visualiser les signaux électriques en temps réel.", [
    ("hint_tool", "oscilloscope")])
_add("Le chronomètre a été essentiel pour la navigation maritime au XVIIIe siècle.", [
    ("hint_tool", "chronomètre"), ("hint_time_date", "XVIIIe siècle")])
_add("L'accélérateur de particules du CERN est le plus grand instrument scientifique au monde.", [
    ("hint_tool", "accélérateur de particules"), ("hint_org_name", "CERN")])
_add("Le voltmètre mesure la différence de potentiel entre deux points d'un circuit.", [
    ("hint_tool", "voltmètre")])
_add("L'anémomètre mesure la vitesse du vent sur les stations météorologiques.", [
    ("hint_tool", "anémomètre")])
_add("L'échographe permet de visualiser le fœtus pendant la grossesse.", [
    ("hint_tool", "échographe")])
_add("Le compteur Geiger détecte et mesure les rayonnements ionisants.", [
    ("hint_tool", "compteur Geiger")])
_add("Le multimètre est l'outil de base de tout électricien.", [
    ("hint_tool", "multimètre"), ("hint_person_role", "électricien")])
_add("L'imprimante 3D fabrique des objets couche par couche à partir d'un modèle numérique.", [
    ("hint_tool", "imprimante 3D")])
_add("Le pied à coulisse mesure les dimensions extérieures et intérieures avec précision.", [
    ("hint_tool", "pied à coulisse")])
_add("Le niveau à bulle vérifie l'horizontalité d'une surface.", [
    ("hint_tool", "niveau à bulle")])

# ═══════════════════════════════════════════════════════════════
#  hint_food  (~30 phrases)
# ═══════════════════════════════════════════════════════════════

_add("Le foie gras est un mets traditionnel du sud-ouest de la France.", [
    ("hint_food", "foie gras"), ("hint_gpe", "France")])
_add("Le quinoa est cultivé principalement en Bolivie et au Pérou.", [
    ("hint_food", "quinoa"), ("hint_gpe", "Bolivie"), ("hint_gpe", "Pérou")])
_add("Le tofu est une source de protéines végétales consommée depuis des millénaires en Asie.", [
    ("hint_food", "tofu"), ("hint_gpe", "Asie")])
_add("Le manioc est l'aliment de base de millions de personnes en Afrique.", [
    ("hint_food", "manioc"), ("hint_gpe", "Afrique")])
_add("Le couscous a été inscrit au patrimoine immatériel de l'UNESCO en 2020.", [
    ("hint_food", "couscous"), ("hint_org_name", "UNESCO"), ("hint_time_date", "2020")])
_add("La baguette de pain a été inscrite au patrimoine culturel immatériel de la France.", [
    ("hint_food", "baguette de pain"), ("hint_gpe", "France")])
_add("Le kimchi est un plat fermenté emblématique de la cuisine coréenne.", [
    ("hint_food", "kimchi"), ("hint_norp", "coréenne")])
_add("Le tempeh est un aliment fermenté à base de soja originaire d'Indonésie.", [
    ("hint_food", "tempeh"), ("hint_food", "soja"), ("hint_gpe", "Indonésie")])
_add("Le safran est l'épice la plus chère au monde, cultivée notamment en Iran.", [
    ("hint_food", "safran"), ("hint_gpe", "Iran")])
_add("Le ginseng est utilisé en médecine traditionnelle chinoise depuis des siècles.", [
    ("hint_food", "ginseng"), ("hint_norp", "chinoise")])
_add("Le piment est originaire d'Amérique et a été introduit en Asie par les Portugais.", [
    ("hint_food", "piment"), ("hint_gpe", "Amérique"), ("hint_gpe", "Asie"), ("hint_norp", "Portugais")])
_add("Le matcha est un thé vert réduit en poudre très prisé au Japon.", [
    ("hint_food", "matcha"), ("hint_food", "thé vert"), ("hint_gpe", "Japon")])
_add("Le parmesan est un fromage italien vieilli au moins 12 mois.", [
    ("hint_food", "parmesan"), ("hint_norp", "italien")])
_add("Le miso est une pâte de soja fermentée essentielle dans la cuisine japonaise.", [
    ("hint_food", "miso"), ("hint_food", "soja"), ("hint_norp", "japonaise")])
_add("Le tapioca est extrait de la racine de manioc.", [
    ("hint_food", "tapioca"), ("hint_food", "manioc")])

# ═══════════════════════════════════════════════════════════════
#  hint_object_name  (~20 phrases)
# ═══════════════════════════════════════════════════════════════

_add("Le Boeing 747 a révolutionné le transport aérien lors de son introduction en 1970.", [
    ("hint_object_name", "Boeing 747"), ("hint_time_date", "1970")])
_add("Le Concorde était le seul avion de ligne supersonique en service commercial.", [
    ("hint_object_name", "Concorde")])
_add("La Tour Eiffel a été construite pour l'Exposition universelle de 1889 à Paris.", [
    ("hint_object_name", "Tour Eiffel"), ("hint_time_date", "1889"), ("hint_gpe", "Paris")])
_add("Le Titanic a coulé lors de son voyage inaugural en 1912 après avoir heurté un iceberg.", [
    ("hint_object_name", "Titanic"), ("hint_time_date", "1912")])
_add("L'iPhone d'Apple a transformé l'industrie de la téléphonie mobile en 2007.", [
    ("hint_object_name", "iPhone"), ("hint_org_name", "Apple"), ("hint_time_date", "2007")])
_add("Le télescope James Webb a été lancé en décembre 2021 par la NASA.", [
    ("hint_object_name", "télescope James Webb"), ("hint_time_date", "décembre 2021"), ("hint_org_name", "NASA")])
_add("Le Spoutnik a été le premier satellite artificiel mis en orbite en 1957.", [
    ("hint_object_name", "Spoutnik"), ("hint_time_date", "1957")])
_add("La PlayStation de Sony domine le marché des consoles de jeux vidéo.", [
    ("hint_object_name", "PlayStation"), ("hint_org_name", "Sony")])
_add("Le pont du Golden Gate est l'un des monuments les plus photographiés de San Francisco.", [
    ("hint_object_name", "pont du Golden Gate"), ("hint_gpe", "San Francisco")])
_add("Le rover Curiosity explore la surface de Mars depuis 2012.", [
    ("hint_object_name", "Curiosity"), ("hint_gpe", "Mars"), ("hint_time_date", "2012")])

# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/boost_weak_classes.jsonl")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    random.seed(args.seed)
    random.shuffle(SENTENCES)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, sent in enumerate(SENTENCES):
            row = {"id": f"boost_{i:04d}", "text": sent["text"], "spans": sent["spans"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter
    lc = Counter()
    for s in SENTENCES:
        for sp in s["spans"]:
            lc[sp["label"]] += 1

    print(f"\n{len(SENTENCES)} phrases ecrites dans {args.output}")
    print(f"\nDistribution :")
    for label, count in sorted(lc.items(), key=lambda x: -x[1]):
        print(f"   {label:30s} {count:4d}")

if __name__ == "__main__":
    main()

