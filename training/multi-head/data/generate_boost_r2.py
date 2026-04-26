#!/usr/bin/env python3
"""Boost round 2 : remonter hint_law et hint_disease au-dessus de 250."""
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
#  hint_law  (~120 phrases pour passer de 128 à ~250)
# ═══════════════════════════════════════════════════════════════

_add("La loi Rothschild de 1973 a interdit à la Banque de France de prêter directement à l'État.", [
    ("hint_law", "loi Rothschild"), ("hint_time_date", "1973"), ("hint_org_name", "Banque de France")])
_add("Le traité de Neuilly de 1919 a sanctionné la Bulgarie après la Grande Guerre.", [
    ("hint_law", "traité de Neuilly"), ("hint_time_date", "1919"), ("hint_gpe", "Bulgarie")])
_add("La loi Falloux de 1850 a rétabli la liberté d'enseignement en France.", [
    ("hint_law", "loi Falloux"), ("hint_time_date", "1850"), ("hint_gpe", "France")])
_add("Le traité de Shimonoseki de 1895 a marqué la victoire du Japon sur la Chine.", [
    ("hint_law", "traité de Shimonoseki"), ("hint_time_date", "1895"), ("hint_gpe", "Japon"), ("hint_gpe", "Chine")])
_add("La loi Debré de 1959 a organisé les rapports entre l'État et l'enseignement privé.", [
    ("hint_law", "loi Debré"), ("hint_time_date", "1959")])
_add("Le décret Guizot de 1833 a rendu obligatoire la création d'écoles primaires.", [
    ("hint_law", "décret Guizot"), ("hint_time_date", "1833")])
_add("Le traité de Cateau-Cambrésis de 1559 mit fin aux guerres d'Italie.", [
    ("hint_law", "traité de Cateau-Cambrésis"), ("hint_time_date", "1559"), ("hint_gpe", "Italie")])
_add("La loi Ferry de 1882 a rendu l'enseignement primaire gratuit et obligatoire.", [
    ("hint_law", "loi Ferry"), ("hint_time_date", "1882")])
_add("Le traité de Locarno de 1925 a garanti les frontières occidentales de l'Allemagne.", [
    ("hint_law", "traité de Locarno"), ("hint_time_date", "1925"), ("hint_gpe", "Allemagne")])
_add("La loi Neuwirth de 1967 a autorisé la contraception en France.", [
    ("hint_law", "loi Neuwirth"), ("hint_time_date", "1967"), ("hint_gpe", "France")])
_add("Le traité de Passarowitz de 1718 mit fin à la guerre entre l'Autriche et l'Empire ottoman.", [
    ("hint_law", "traité de Passarowitz"), ("hint_time_date", "1718"), ("hint_gpe", "Autriche")])
_add("L'armistice de Rethondes du 11 novembre 1918 a été signé dans un wagon.", [
    ("hint_law", "armistice de Rethondes"), ("hint_time_date", "11 novembre 1918")])
_add("Le traité de Koutchouk-Kaïnardji de 1774 accorda à la Russie un accès à la mer Noire.", [
    ("hint_law", "traité de Koutchouk-Kaïnardji"), ("hint_time_date", "1774"), ("hint_gpe", "Russie")])
_add("La loi Waldeck-Rousseau de 1884 a légalisé les syndicats en France.", [
    ("hint_law", "loi Waldeck-Rousseau"), ("hint_time_date", "1884"), ("hint_gpe", "France")])
_add("Le traité de Cordoue de 1236 a établi la paix entre Castillans et Almohades.", [
    ("hint_law", "traité de Cordoue"), ("hint_time_date", "1236")])
_add("La loi Auroux de 1982 a renforcé les droits des travailleurs dans l'entreprise.", [
    ("hint_law", "loi Auroux"), ("hint_time_date", "1982")])
_add("Le traité de Brétigny de 1360 a mis fin à la première phase de la guerre de Cent Ans.", [
    ("hint_law", "traité de Brétigny"), ("hint_time_date", "1360")])
_add("La Constitution de Weimar de 1919 a fondé la première démocratie allemande.", [
    ("hint_law", "Constitution de Weimar"), ("hint_time_date", "1919"), ("hint_norp", "allemande")])
_add("Le Patriot Act a renforcé les pouvoirs de surveillance aux États-Unis après le 11 septembre.", [
    ("hint_law", "Patriot Act"), ("hint_gpe", "États-Unis")])
_add("L'édit de Milan de 313 a accordé la liberté de culte aux chrétiens dans l'Empire romain.", [
    ("hint_law", "édit de Milan"), ("hint_time_date", "313"), ("hint_norp", "chrétiens")])
_add("La loi Deferre de 1982 a lancé la décentralisation en France.", [
    ("hint_law", "loi Deferre"), ("hint_time_date", "1982"), ("hint_gpe", "France")])
_add("Le traité de Lunéville de 1801 a confirmé les acquisitions françaises en Rhénanie.", [
    ("hint_law", "traité de Lunéville"), ("hint_time_date", "1801"), ("hint_norp", "françaises")])
_add("La loi Royer de 1973 a encadré l'implantation des grandes surfaces commerciales.", [
    ("hint_law", "loi Royer"), ("hint_time_date", "1973")])
_add("Le Civil Rights Act de 1964 a interdit la discrimination raciale aux États-Unis.", [
    ("hint_law", "Civil Rights Act"), ("hint_time_date", "1964"), ("hint_gpe", "États-Unis")])
_add("L'édit de Thessalonique de 380 a fait du christianisme la religion officielle de l'Empire.", [
    ("hint_law", "édit de Thessalonique"), ("hint_time_date", "380"), ("hint_concept", "christianisme")])
_add("Le traité d'Andrinople de 1829 accorda l'autonomie à la Grèce.", [
    ("hint_law", "traité d'Andrinople"), ("hint_time_date", "1829"), ("hint_gpe", "Grèce")])
_add("La loi Lang de 1981 a instauré le prix unique du livre en France.", [
    ("hint_law", "loi Lang"), ("hint_time_date", "1981"), ("hint_gpe", "France")])
_add("Le traité INF de 1987 a éliminé les missiles nucléaires à portée intermédiaire.", [
    ("hint_law", "traité INF"), ("hint_time_date", "1987")])
_add("La loi Macron de 2015 a libéralisé le travail du dimanche et le transport par autocar.", [
    ("hint_law", "loi Macron"), ("hint_time_date", "2015")])
_add("Le traité de Berlin de 1878 a redessiné la carte des Balkans.", [
    ("hint_law", "traité de Berlin"), ("hint_time_date", "1878"), ("hint_gpe", "Balkans")])
_add("La loi Badinter de 1981 a aboli la peine de mort en France.", [
    ("hint_law", "loi Badinter"), ("hint_time_date", "1981"), ("hint_gpe", "France")])
_add("Le traité de Portsmouth de 1905 mit fin à la guerre russo-japonaise.", [
    ("hint_law", "traité de Portsmouth"), ("hint_time_date", "1905")])
_add("La loi Besson de 2000 a renforcé le droit au logement des gens du voyage.", [
    ("hint_law", "loi Besson"), ("hint_time_date", "2000")])
_add("Le traité de Riga de 1921 fixa la frontière entre la Pologne et la Russie soviétique.", [
    ("hint_law", "traité de Riga"), ("hint_time_date", "1921"), ("hint_gpe", "Pologne")])
_add("La loi Informatique et Libertés de 1978 a créé la CNIL.", [
    ("hint_law", "loi Informatique et Libertés"), ("hint_time_date", "1978"), ("hint_org_name", "CNIL")])
_add("Le traité ABM de 1972 limitait les systèmes antimissiles balistiques.", [
    ("hint_law", "traité ABM"), ("hint_time_date", "1972")])
_add("L'ordonnance de janvier 1959 a posé les bases de la Constitution financière française.", [
    ("hint_law", "ordonnance de janvier 1959"), ("hint_norp", "française")])
_add("Le traité de Tlatelolco de 1967 a interdit les armes nucléaires en Amérique latine.", [
    ("hint_law", "traité de Tlatelolco"), ("hint_time_date", "1967"), ("hint_gpe", "Amérique latine")])
_add("La loi Hamon de 2014 a renforcé les droits des consommateurs en matière de rétractation.", [
    ("hint_law", "loi Hamon"), ("hint_time_date", "2014")])
_add("Le Voting Rights Act de 1965 a garanti le droit de vote aux Afro-Américains.", [
    ("hint_law", "Voting Rights Act"), ("hint_time_date", "1965"), ("hint_norp", "Afro-Américains")])
_add("La charte de l'Atlantique de 1941 a défini les buts de guerre anglo-américains.", [
    ("hint_law", "charte de l'Atlantique"), ("hint_time_date", "1941")])
_add("Le traité de Waitangi de 1840 est le document fondateur de la Nouvelle-Zélande.", [
    ("hint_law", "traité de Waitangi"), ("hint_time_date", "1840"), ("hint_gpe", "Nouvelle-Zélande")])
_add("La loi Grenelle 2 de 2010 a fixé les engagements environnementaux de la France.", [
    ("hint_law", "loi Grenelle 2"), ("hint_time_date", "2010"), ("hint_gpe", "France")])
_add("Le traité de Rarotonga de 1985 a créé une zone dénucléarisée dans le Pacifique Sud.", [
    ("hint_law", "traité de Rarotonga"), ("hint_time_date", "1985")])
_add("La loi El Khomri de 2016 a assoupli le droit du travail en France.", [
    ("hint_law", "loi El Khomri"), ("hint_time_date", "2016"), ("hint_gpe", "France")])
_add("Le traité de Pelindaba de 1996 interdit les armes nucléaires sur le continent africain.", [
    ("hint_law", "traité de Pelindaba"), ("hint_time_date", "1996"), ("hint_norp", "africain")])
_add("La loi de finances rectificative a été adoptée en urgence par le Parlement.", [
    ("hint_law", "loi de finances rectificative"), ("hint_org_name", "Parlement")])
_add("Le traité de paix de San Francisco de 1951 a mis fin à l'occupation du Japon.", [
    ("hint_law", "traité de paix de San Francisco"), ("hint_time_date", "1951"), ("hint_gpe", "Japon")])
_add("La Convention des droits de l'enfant a été adoptée par l'ONU en 1989.", [
    ("hint_law", "Convention des droits de l'enfant"), ("hint_org_name", "ONU"), ("hint_time_date", "1989")])
_add("Le traité de Nystad de 1721 a consacré la Russie comme grande puissance européenne.", [
    ("hint_law", "traité de Nystad"), ("hint_time_date", "1721"), ("hint_gpe", "Russie")])
_add("La loi organique relative aux lois de finances de 2001 a modernisé le budget de l'État.", [
    ("hint_law", "loi organique relative aux lois de finances"), ("hint_time_date", "2001")])
_add("Le décret de Moscou de 1812 ordonnait l'incendie de la capitale russe.", [
    ("hint_law", "décret de Moscou"), ("hint_time_date", "1812"), ("hint_norp", "russe")])
_add("La loi Savary de 1984 a réformé l'enseignement supérieur français.", [
    ("hint_law", "loi Savary"), ("hint_time_date", "1984"), ("hint_norp", "français")])
_add("Le traité de Tilsit de 1807 a partagé l'Europe entre la France et la Russie.", [
    ("hint_law", "traité de Tilsit"), ("hint_time_date", "1807"), ("hint_gpe", "France"), ("hint_gpe", "Russie")])
_add("La loi Duflot de 2013 a encadré les loyers dans les zones tendues.", [
    ("hint_law", "loi Duflot"), ("hint_time_date", "2013")])
_add("Le traité de San Germain de 1919 a fixé les frontières de l'Autriche moderne.", [
    ("hint_law", "traité de San Germain"), ("hint_time_date", "1919"), ("hint_gpe", "Autriche")])
_add("La loi Pinel de 2014 a instauré un dispositif de défiscalisation immobilière.", [
    ("hint_law", "loi Pinel"), ("hint_time_date", "2014")])
_add("Le traité de La Haye de 1625 renouvela l'alliance entre la France et les Provinces-Unies.", [
    ("hint_law", "traité de La Haye"), ("hint_time_date", "1625"), ("hint_gpe", "France")])
_add("La loi de bioéthique de 2021 a élargi l'accès à la procréation médicalement assistée.", [
    ("hint_law", "loi de bioéthique"), ("hint_time_date", "2021")])
_add("Le traité de Münster mit fin à la guerre de Quatre-Vingts Ans entre l'Espagne et les Pays-Bas.", [
    ("hint_law", "traité de Münster"), ("hint_gpe", "Espagne"), ("hint_gpe", "Pays-Bas")])
_add("La loi pour la confiance dans l'économie numérique date de 2004.", [
    ("hint_law", "loi pour la confiance dans l'économie numérique"), ("hint_time_date", "2004")])
_add("Le traité de Géranium de 1648 confirma l'indépendance de la Confédération suisse.", [
    ("hint_law", "traité de Géranium"), ("hint_time_date", "1648"), ("hint_norp", "suisse")])
_add("La loi Raffarin de 2005 a simplifié les régimes d'assurance-chômage.", [
    ("hint_law", "loi Raffarin"), ("hint_time_date", "2005")])
_add("Le décret impérial de Lyon de 1810 a réorganisé l'Université française.", [
    ("hint_law", "décret impérial de Lyon"), ("hint_time_date", "1810"), ("hint_norp", "française")])
_add("Le traité de Fontainebleau de 1814 régla les conditions de l'abdication de Napoléon.", [
    ("hint_law", "traité de Fontainebleau"), ("hint_time_date", "1814"), ("hint_person_name", "Napoléon")])
_add("La Constitution espagnole de 1978 a instauré une monarchie parlementaire.", [
    ("hint_law", "Constitution espagnole"), ("hint_time_date", "1978")])
_add("Le traité de Nice de 2001 a réformé les institutions avant l'élargissement de l'UE.", [
    ("hint_law", "traité de Nice"), ("hint_time_date", "2001")])
_add("La loi Alur de 2014 a réformé le droit de l'urbanisme et du logement.", [
    ("hint_law", "loi Alur"), ("hint_time_date", "2014")])
_add("La Convention sur les armes chimiques interdit la production de gaz de combat.", [
    ("hint_law", "Convention sur les armes chimiques")])
_add("Le décret de Grasse de 1791 réorganisa les municipalités françaises.", [
    ("hint_law", "décret de Grasse"), ("hint_time_date", "1791"), ("hint_norp", "françaises")])
_add("Le traité ICAN a renforcé le régime de non-prolifération nucléaire.", [
    ("hint_law", "traité ICAN")])
_add("L'accord de libre-échange nord-américain a été remplacé par l'ACEUM en 2020.", [
    ("hint_law", "accord de libre-échange nord-américain"), ("hint_law", "ACEUM"), ("hint_time_date", "2020")])
_add("La loi relative à la solidarité et au renouvellement urbains date de 2000.", [
    ("hint_law", "loi relative à la solidarité et au renouvellement urbains"), ("hint_time_date", "2000")])
_add("Le traité de Saragosse de 1529 compléta le traité de Tordesillas pour le Pacifique.", [
    ("hint_law", "traité de Saragosse"), ("hint_time_date", "1529"), ("hint_law", "traité de Tordesillas")])

# ═══════════════════════════════════════════════════════════════
#  hint_disease  (~120 phrases pour passer de 134 à ~250)
# ═══════════════════════════════════════════════════════════════

_add("La borréliose est souvent confondue avec la maladie de Lyme.", [
    ("hint_disease", "borréliose"), ("hint_disease", "maladie de Lyme")])
_add("La candidose est une infection fongique fréquente chez les immunodéprimés.", [
    ("hint_disease", "candidose")])
_add("Le syndrome d'Ehlers-Danlos se caractérise par une hypermobilité articulaire.", [
    ("hint_disease", "syndrome d'Ehlers-Danlos")])
_add("La maladie de Horton provoque des maux de tête violents chez les personnes âgées.", [
    ("hint_disease", "maladie de Horton")])
_add("Le syndrome de Prader-Willi entraîne une sensation de faim permanente.", [
    ("hint_disease", "syndrome de Prader-Willi")])
_add("La maladie de Ménière est responsable de vertiges invalidants.", [
    ("hint_disease", "maladie de Ménière")])
_add("La péritonite est une urgence chirurgicale causée par une infection abdominale.", [
    ("hint_disease", "péritonite")])
_add("Le syndrome de Diogène se manifeste par une accumulation compulsive d'objets.", [
    ("hint_disease", "syndrome de Diogène")])
_add("La sarcoïdose est une maladie inflammatoire qui touche principalement les poumons.", [
    ("hint_disease", "sarcoïdose")])
_add("Le syndrome hémolytique et urémique est une complication grave de certaines infections.", [
    ("hint_disease", "syndrome hémolytique et urémique")])
_add("La maladie de Whipple est une infection bactérienne rare du tube digestif.", [
    ("hint_disease", "maladie de Whipple")])
_add("Le syndrome de Reye est une complication rare mais grave liée à la prise d'aspirine chez l'enfant.", [
    ("hint_disease", "syndrome de Reye")])
_add("La maladie de Verneuil provoque des abcès douloureux sous la peau.", [
    ("hint_disease", "maladie de Verneuil")])
_add("Le syndrome des jambes sans repos perturbe gravement le sommeil.", [
    ("hint_disease", "syndrome des jambes sans repos")])
_add("La maladie de Perthes touche la hanche des enfants entre 3 et 12 ans.", [
    ("hint_disease", "maladie de Perthes")])
_add("Le syndrome de Korsakoff est une perte de mémoire liée à l'alcoolisme chronique.", [
    ("hint_disease", "syndrome de Korsakoff")])
_add("La maladie de Dupuytren provoque une rétraction progressive des doigts.", [
    ("hint_disease", "maladie de Dupuytren")])
_add("Le choléra a provoqué plusieurs épidémies au XIXe siècle en Europe.", [
    ("hint_disease", "choléra"), ("hint_time_date", "XIXe siècle"), ("hint_gpe", "Europe")])
_add("La grippe asiatique de 1957 a tué environ deux millions de personnes.", [
    ("hint_disease", "grippe asiatique"), ("hint_time_date", "1957"), ("hint_quantity", "deux millions")])
_add("La maladie de Behçet provoque des ulcères buccaux et génitaux récurrents.", [
    ("hint_disease", "maladie de Behçet")])
_add("Le syndrome de Brugada est une anomalie cardiaque pouvant causer la mort subite.", [
    ("hint_disease", "syndrome de Brugada")])
_add("La maladie de Buerger touche les artères des mains et des pieds chez les fumeurs.", [
    ("hint_disease", "maladie de Buerger")])
_add("L'herpès est une infection virale récurrente transmise par contact cutané.", [
    ("hint_disease", "herpès")])
_add("La varicelle est une maladie infantile très contagieuse.", [
    ("hint_disease", "varicelle")])
_add("Les oreillons peuvent entraîner une surdité en cas de complication.", [
    ("hint_disease", "oreillons")])
_add("La rubéole est dangereuse pour le fœtus si la mère est infectée pendant la grossesse.", [
    ("hint_disease", "rubéole")])
_add("La maladie de Still est une forme rare d'arthrite juvénile systémique.", [
    ("hint_disease", "maladie de Still")])
_add("Le syndrome de Raynaud provoque un blanchiment des doigts au froid.", [
    ("hint_disease", "syndrome de Raynaud")])
_add("La maladie de Kaposi est un cancer associé au sida.", [
    ("hint_disease", "maladie de Kaposi"), ("hint_disease", "sida")])
_add("Le syndrome de Williams est un trouble génétique rare associé à un visage d'elfe.", [
    ("hint_disease", "syndrome de Williams")])
_add("La maladie de Bowen est une forme précoce de cancer cutané.", [
    ("hint_disease", "maladie de Bowen")])
_add("Le cancer du pancréas est l'un des cancers les plus meurtriers.", [
    ("hint_disease", "cancer du pancréas")])
_add("Le cancer de l'ovaire est souvent diagnostiqué à un stade avancé.", [
    ("hint_disease", "cancer de l'ovaire")])
_add("Le cancer du foie est fréquemment associé à une hépatite chronique.", [
    ("hint_disease", "cancer du foie"), ("hint_disease", "hépatite")])
_add("Le cancer de la vessie touche plus souvent les fumeurs.", [
    ("hint_disease", "cancer de la vessie")])
_add("Le cancer du rein est souvent découvert de façon fortuite à l'imagerie.", [
    ("hint_disease", "cancer du rein")])
_add("Le cancer de l'estomac est en recul dans les pays industrialisés.", [
    ("hint_disease", "cancer de l'estomac")])
_add("Le myélome multiple est un cancer de la moelle osseuse.", [
    ("hint_disease", "myélome multiple")])
_add("Le glioblastome est la tumeur cérébrale maligne la plus fréquente chez l'adulte.", [
    ("hint_disease", "glioblastome")])
_add("La maladie de Hirschsprung est une malformation congénitale de l'intestin.", [
    ("hint_disease", "maladie de Hirschsprung")])
_add("Le syndrome de Bloom est une maladie génétique prédisposant aux cancers.", [
    ("hint_disease", "syndrome de Bloom")])
_add("La maladie de Wegener provoque une inflammation des vaisseaux sanguins.", [
    ("hint_disease", "maladie de Wegener")])
_add("La maladie de Addison est une insuffisance surrénalienne chronique.", [
    ("hint_disease", "maladie de Addison")])
_add("Le syndrome de Cushing est caractérisé par un excès de cortisol.", [
    ("hint_disease", "syndrome de Cushing")])
_add("La maladie de Conn provoque une hypertension liée à un excès d'aldostérone.", [
    ("hint_disease", "maladie de Conn")])
_add("La fièvre de Lassa est une fièvre hémorragique endémique en Afrique de l'Ouest.", [
    ("hint_disease", "fièvre de Lassa"), ("hint_gpe", "Afrique de l'Ouest")])
_add("La fièvre de la vallée du Rift touche le bétail et peut se transmettre à l'homme.", [
    ("hint_disease", "fièvre de la vallée du Rift")])
_add("La peste pneumonique est la forme la plus grave et la plus contagieuse de la peste.", [
    ("hint_disease", "peste pneumonique")])
_add("La fièvre Q est une zoonose transmise par les ruminants domestiques.", [
    ("hint_disease", "fièvre Q")])
_add("La maladie du sommeil menace encore des populations rurales en Afrique subsaharienne.", [
    ("hint_disease", "maladie du sommeil"), ("hint_gpe", "Afrique subsaharienne")])
_add("L'ulcère de Buruli est une maladie tropicale négligée causée par une mycobactérie.", [
    ("hint_disease", "ulcère de Buruli")])
_add("Le syndrome de Zollinger-Ellison provoque une hypersécrétion d'acide gastrique.", [
    ("hint_disease", "syndrome de Zollinger-Ellison")])
_add("La maladie de Pompe est une glycogénose traitable par enzymothérapie substitutive.", [
    ("hint_disease", "maladie de Pompe")])
_add("Le syndrome de Lennox-Gastaut est une forme sévère d'épilepsie de l'enfant.", [
    ("hint_disease", "syndrome de Lennox-Gastaut"), ("hint_disease", "épilepsie")])
_add("La maladie de Niemann-Pick est une maladie lysosomale d'accumulation lipidique.", [
    ("hint_disease", "maladie de Niemann-Pick")])
_add("La fibrose pulmonaire réduit progressivement la capacité respiratoire.", [
    ("hint_disease", "fibrose pulmonaire")])
_add("L'apnée du sommeil provoque des arrêts respiratoires pendant la nuit.", [
    ("hint_disease", "apnée du sommeil")])
_add("La maladie de Biermer est une anémie due à un déficit en vitamine B12.", [
    ("hint_disease", "maladie de Biermer")])
_add("Le psoriasis en gouttes apparaît souvent après une infection streptococcique.", [
    ("hint_disease", "psoriasis en gouttes")])
_add("La dermatite atopique touche jusqu'à 20 % des enfants dans les pays occidentaux.", [
    ("hint_disease", "dermatite atopique"), ("hint_percentage", "20 %")])
_add("Le vitiligo provoque une dépigmentation de la peau par destruction des mélanocytes.", [
    ("hint_disease", "vitiligo")])
_add("La maladie de Graves est la cause la plus fréquente d'hyperthyroïdie auto-immune.", [
    ("hint_disease", "maladie de Graves")])
_add("Le syndrome des ovaires polykystiques touche environ 10 % des femmes en âge de procréer.", [
    ("hint_disease", "syndrome des ovaires polykystiques"), ("hint_percentage", "10 %")])
_add("La maladie de Tay-Sachs est une maladie lysosomale fatale du nourrisson.", [
    ("hint_disease", "maladie de Tay-Sachs")])
_add("Le syndrome de Wiskott-Aldrich est un déficit immunitaire héréditaire lié à l'X.", [
    ("hint_disease", "syndrome de Wiskott-Aldrich")])
_add("La maladie de von Willebrand est le trouble de la coagulation héréditaire le plus fréquent.", [
    ("hint_disease", "maladie de von Willebrand")])
_add("Le syndrome de Cri du chat est causé par une délétion du chromosome 5.", [
    ("hint_disease", "syndrome de Cri du chat")])
_add("La maladie de Ménétrier provoque un épaississement de la muqueuse gastrique.", [
    ("hint_disease", "maladie de Ménétrier")])

# ═══════════════════════════════════════════════════════════════
#  hint_work_of_art  (~70 phrases en plus pour passer de 136 à ~200)
# ═══════════════════════════════════════════════════════════════

_add("Le Nom de la rose d'Umberto Eco est un polar médiéval érudit.", [
    ("hint_work_of_art", "Nom de la rose"), ("hint_person_name", "Umberto Eco")])
_add("La Lettre écarlate de Hawthorne explore la culpabilité puritaine en Nouvelle-Angleterre.", [
    ("hint_work_of_art", "Lettre écarlate"), ("hint_person_name", "Hawthorne")])
_add("Candide de Voltaire est une satire philosophique du meilleur des mondes possibles.", [
    ("hint_work_of_art", "Candide"), ("hint_person_name", "Voltaire")])
_add("Les Choses de Georges Perec décrivent l'aliénation par la société de consommation.", [
    ("hint_work_of_art", "Choses"), ("hint_person_name", "Georges Perec")])
_add("Le Désespéré de Léon Bloy est un roman autobiographique d'une violence rare.", [
    ("hint_work_of_art", "Désespéré"), ("hint_person_name", "Léon Bloy")])
_add("Beloved de Toni Morrison est hanté par le souvenir de l'esclavage.", [
    ("hint_work_of_art", "Beloved"), ("hint_person_name", "Toni Morrison")])
_add("Le Tunnel de Ernesto Sabato est un roman existentialiste argentin.", [
    ("hint_work_of_art", "Tunnel"), ("hint_person_name", "Ernesto Sabato"), ("hint_norp", "argentin")])
_add("La Route de Cormac McCarthy décrit un monde post-apocalyptique dévastateur.", [
    ("hint_work_of_art", "Route"), ("hint_person_name", "Cormac McCarthy")])
_add("Neige de Pamuk mêle politique et poésie dans une petite ville turque.", [
    ("hint_work_of_art", "Neige"), ("hint_person_name", "Pamuk"), ("hint_norp", "turque")])
_add("Le Guépard de Lampedusa raconte le déclin de l'aristocratie sicilienne.", [
    ("hint_work_of_art", "Guépard"), ("hint_person_name", "Lampedusa"), ("hint_norp", "sicilienne")])
_add("Les Buddenbrook de Thomas Mann décrivent la chute d'une famille bourgeoise de Lübeck.", [
    ("hint_work_of_art", "Buddenbrook"), ("hint_person_name", "Thomas Mann"), ("hint_gpe", "Lübeck")])
_add("Rashōmon d'Akira Kurosawa a fait découvrir le cinéma japonais au monde.", [
    ("hint_work_of_art", "Rashōmon"), ("hint_person_name", "Akira Kurosawa"), ("hint_norp", "japonais")])
_add("La Maison aux esprits d'Isabel Allende est un classique du réalisme magique.", [
    ("hint_work_of_art", "Maison aux esprits"), ("hint_person_name", "Isabel Allende"), ("hint_concept", "réalisme magique")])
_add("Pedro Páramo de Juan Rulfo a révolutionné la littérature latino-américaine.", [
    ("hint_work_of_art", "Pedro Páramo"), ("hint_person_name", "Juan Rulfo")])
_add("Solaris de Stanisław Lem interroge les limites de la communication avec l'inconnu.", [
    ("hint_work_of_art", "Solaris"), ("hint_person_name", "Stanisław Lem")])
_add("Le Silence des agneaux de Jonathan Demme a valu un Oscar à Jodie Foster.", [
    ("hint_work_of_art", "Silence des agneaux"), ("hint_person_name", "Jonathan Demme"), ("hint_person_name", "Jodie Foster")])
_add("Mulholland Drive de David Lynch est un puzzle onirique sur Hollywood.", [
    ("hint_work_of_art", "Mulholland Drive"), ("hint_person_name", "David Lynch")])
_add("La Chartreuse de Parme de Stendhal se déroule dans l'Italie post-napoléonienne.", [
    ("hint_work_of_art", "Chartreuse de Parme"), ("hint_person_name", "Stendhal"), ("hint_gpe", "Italie")])
_add("Germinal de Zola a été adapté au cinéma par Claude Berri en 1993.", [
    ("hint_work_of_art", "Germinal"), ("hint_person_name", "Zola"), ("hint_person_name", "Claude Berri"), ("hint_time_date", "1993")])
_add("Le Rouge et le Noir de Stendhal est un roman d'apprentissage sous la Restauration.", [
    ("hint_work_of_art", "Rouge et le Noir"), ("hint_person_name", "Stendhal")])
_add("Les Rougon-Macquart de Zola forment un cycle de vingt romans naturalistes.", [
    ("hint_work_of_art", "Rougon-Macquart"), ("hint_person_name", "Zola"), ("hint_concept", "naturalistes")])
_add("Taxi Driver de Scorsese a révélé Robert De Niro au grand public en 1976.", [
    ("hint_work_of_art", "Taxi Driver"), ("hint_person_name", "Scorsese"), ("hint_person_name", "Robert De Niro"), ("hint_time_date", "1976")])
_add("Le Trône de fer de George R.R. Martin a inspiré la série Game of Thrones.", [
    ("hint_work_of_art", "Trône de fer"), ("hint_person_name", "George R.R. Martin"), ("hint_work_of_art", "Game of Thrones")])
_add("Le Joueur de Dostoïevski est largement autobiographique.", [
    ("hint_work_of_art", "Joueur"), ("hint_person_name", "Dostoïevski")])
_add("Les Sonnets de Shakespeare comptent parmi les plus beaux poèmes d'amour de la langue anglaise.", [
    ("hint_work_of_art", "Sonnets"), ("hint_person_name", "Shakespeare"), ("hint_language", "anglaise")])
_add("Le Fantôme de l'Opéra de Gaston Leroux a été adapté en comédie musicale par Andrew Lloyd Webber.", [
    ("hint_work_of_art", "Fantôme de l'Opéra"), ("hint_person_name", "Gaston Leroux"), ("hint_person_name", "Andrew Lloyd Webber")])
_add("Shogun de James Clavell se déroule dans le Japon féodal du XVIIe siècle.", [
    ("hint_work_of_art", "Shogun"), ("hint_person_name", "James Clavell"), ("hint_gpe", "Japon"), ("hint_time_date", "XVIIe siècle")])
_add("Watchmen d'Alan Moore est un comic qui a redéfini le genre des super-héros.", [
    ("hint_work_of_art", "Watchmen"), ("hint_person_name", "Alan Moore")])
_add("Persépolis de Marjane Satrapi raconte la Révolution iranienne à travers les yeux d'une enfant.", [
    ("hint_work_of_art", "Persépolis"), ("hint_person_name", "Marjane Satrapi"), ("hint_event_named", "Révolution iranienne")])
_add("Maus d'Art Spiegelman raconte la Shoah à travers une bande dessinée animalière.", [
    ("hint_work_of_art", "Maus"), ("hint_person_name", "Art Spiegelman"), ("hint_event_named", "Shoah")])

# ═══════════════════════════════════════════════════════════════
#  hint_concept  (~50 phrases en plus pour passer de 161 à ~210)
# ═══════════════════════════════════════════════════════════════

_add("Le providentialisme considère que l'histoire est guidée par la volonté divine.", [
    ("hint_concept", "providentialisme")])
_add("Le volontarisme politique affirme que la volonté humaine peut transformer la société.", [
    ("hint_concept", "volontarisme politique")])
_add("Le gradualisme privilégie les réformes progressives plutôt que la rupture révolutionnaire.", [
    ("hint_concept", "gradualisme")])
_add("Le réformisme cherche à transformer la société dans le cadre des institutions existantes.", [
    ("hint_concept", "réformisme")])
_add("L'étatisme accorde un rôle central à l'État dans l'organisation de l'économie.", [
    ("hint_concept", "étatisme")])
_add("Le dirigisme économique concentre les décisions stratégiques entre les mains de l'État.", [
    ("hint_concept", "dirigisme économique")])
_add("Le libéralisme classique de Locke et Smith défend la propriété et le libre marché.", [
    ("hint_concept", "libéralisme classique"), ("hint_person_name", "Locke"), ("hint_person_name", "Smith")])
_add("Le néoréalisme en relations internationales analyse le système interétatique par la structure.", [
    ("hint_concept", "néoréalisme")])
_add("Le fonctionnalisme de Malinowski étudie les institutions par leur rôle dans la société.", [
    ("hint_concept", "fonctionnalisme"), ("hint_person_name", "Malinowski")])
_add("Le diffusionnisme explique le développement culturel par la transmission entre sociétés.", [
    ("hint_concept", "diffusionnisme")])
_add("L'évolutionnisme social classait les sociétés sur une échelle de progrès.", [
    ("hint_concept", "évolutionnisme social")])
_add("Le culturalisme de Ruth Benedict étudie les cultures comme des totalités cohérentes.", [
    ("hint_concept", "culturalisme"), ("hint_person_name", "Ruth Benedict")])
_add("Le cognitivisme en psychologie étudie les processus mentaux comme un traitement d'information.", [
    ("hint_concept", "cognitivisme")])
_add("Le connectivisme de Siemens propose une théorie de l'apprentissage à l'ère numérique.", [
    ("hint_concept", "connectivisme"), ("hint_person_name", "Siemens")])
_add("Le computationnalisme assimile l'esprit à une machine de Turing.", [
    ("hint_concept", "computationnalisme")])
_add("Le réductionnisme explique les phénomènes complexes par leurs composants élémentaires.", [
    ("hint_concept", "réductionnisme")])
_add("L'émergentisme soutient que le tout possède des propriétés absentes de ses parties.", [
    ("hint_concept", "émergentisme")])
_add("Le déontologisme kantien fonde la morale sur le respect inconditionnel du devoir.", [
    ("hint_concept", "déontologisme kantien")])
_add("Le perfectionnisme moral vise l'excellence humaine comme fin de l'éthique.", [
    ("hint_concept", "perfectionnisme moral")])
_add("Le personnalisme de Mounier met la personne au centre de la réflexion politique.", [
    ("hint_concept", "personnalisme"), ("hint_person_name", "Mounier")])
_add("Le solidarisme de Léon Bourgeois a inspiré la politique sociale de la IIIe République.", [
    ("hint_concept", "solidarisme"), ("hint_person_name", "Léon Bourgeois")])
_add("Le distributisme de Chesterton et Belloc prône la large répartition de la propriété.", [
    ("hint_concept", "distributisme"), ("hint_person_name", "Chesterton"), ("hint_person_name", "Belloc")])
_add("L'ordolibéralisme allemand veut un État fort garant de la concurrence.", [
    ("hint_concept", "ordolibéralisme"), ("hint_norp", "allemand")])
_add("Le supplétivisme en linguistique désigne le remplacement d'une forme par une racine différente.", [
    ("hint_concept", "supplétivisme")])
_add("Le phonocentrisme critiqué par Derrida privilégie la parole sur l'écriture.", [
    ("hint_concept", "phonocentrisme"), ("hint_person_name", "Derrida")])

# ═══════════════════════════════════════════════════════════════
#  hint_language  (~40 phrases en plus)
# ═══════════════════════════════════════════════════════════════

_add("Le tchétchène est une langue caucasienne du nord-est parlée en Russie.", [
    ("hint_language", "tchétchène"), ("hint_gpe", "Russie")])
_add("Le basaa est une langue bantoue parlée au Cameroun.", [
    ("hint_language", "basaa"), ("hint_gpe", "Cameroun")])
_add("Le tatar est une langue turcique parlée au Tatarstan en Russie.", [
    ("hint_language", "tatar"), ("hint_gpe", "Tatarstan"), ("hint_gpe", "Russie")])
_add("Le bachkir est une langue turcique parlée dans l'Oural russe.", [
    ("hint_language", "bachkir"), ("hint_norp", "russe")])
_add("Le tchouvache est la seule langue turcique du groupe ogour encore vivante.", [
    ("hint_language", "tchouvache")])
_add("Le same est une langue finno-ougrienne parlée par le peuple Sami en Scandinavie.", [
    ("hint_language", "same"), ("hint_norp", "Sami")])
_add("L'aïnou est une langue isolée en voie d'extinction au Japon.", [
    ("hint_language", "aïnou"), ("hint_gpe", "Japon")])
_add("Le créole martiniquais mêle des éléments du français et de langues africaines.", [
    ("hint_language", "créole martiniquais"), ("hint_language", "français")])
_add("Le bislama est un pidgin anglais devenu langue officielle du Vanuatu.", [
    ("hint_language", "bislama"), ("hint_language", "anglais"), ("hint_gpe", "Vanuatu")])
_add("Le papiamento est un créole ibérique parlé à Curaçao et à Aruba.", [
    ("hint_language", "papiamento"), ("hint_gpe", "Curaçao"), ("hint_gpe", "Aruba")])
_add("Le sranan tongo est un créole anglais parlé au Suriname.", [
    ("hint_language", "sranan tongo"), ("hint_language", "anglais"), ("hint_gpe", "Suriname")])
_add("Le pirahã est une langue amazonienne connue pour sa structure phonologique minimaliste.", [
    ("hint_language", "pirahã")])
_add("Le toki pona est une langue construite minimaliste comptant à peine 120 mots.", [
    ("hint_language", "toki pona"), ("hint_count", "120")])
_add("Le volapük est une langue construite antérieure à l'espéranto.", [
    ("hint_language", "volapük"), ("hint_language", "espéranto")])
_add("Le klingon est une langue fictive créée pour l'univers de Star Trek.", [
    ("hint_language", "klingon"), ("hint_work_of_art", "Star Trek")])
_add("Le sindarin est une langue elfique créée par Tolkien pour Le Seigneur des anneaux.", [
    ("hint_language", "sindarin"), ("hint_person_name", "Tolkien"), ("hint_work_of_art", "Seigneur des anneaux")])

# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/boost_weak_r2.jsonl")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    random.shuffle(SENTENCES)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, sent in enumerate(SENTENCES):
            row = {"id": f"boost2_{i:04d}", "text": sent["text"], "spans": sent["spans"]}
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

