#!/usr/bin/env python3
"""
Génère des phrases annotées pour les 5 labels ABSTRACT.
Détection automatique des offsets via str.find() pour éviter les erreurs.

Usage:
  python generate_abstract_data.py --output data/abstract_sentences.jsonl
"""
from __future__ import annotations
import json, argparse, random
from typing import List, Dict, Tuple

SENTENCES: List[Dict] = []

def _add(text: str, annotations: List[Tuple[str, str]]):
    spans = []
    used_positions = set()
    for label, surface in annotations:
        # Chercher la première occurrence non encore utilisée
        search_from = 0
        while True:
            idx = text.find(surface, search_from)
            assert idx != -1, f"'{surface}' not found in: {text}"
            if idx not in used_positions:
                used_positions.add(idx)
                break
            search_from = idx + 1
        spans.append({"label": label, "start": idx, "end": idx + len(surface), "text": surface})
    SENTENCES.append({"text": text, "spans": spans})

# ═══════════════════════════════════════════════════════════════
#  hint_law  (40 phrases)
# ═══════════════════════════════════════════════════════════════

_add("Le Traité de Versailles fut signé en 1919 dans la galerie des Glaces.", [
    ("hint_law", "Traité de Versailles"), ("hint_time_date", "1919"), ("hint_fac_name", "galerie des Glaces")])
_add("L'édit de Nantes garantissait la liberté de culte aux protestants.", [
    ("hint_law", "édit de Nantes"), ("hint_norp", "protestants")])
_add("La Constitution de la Ve République a été adoptée en 1958.", [
    ("hint_law", "Constitution de la Ve République"), ("hint_time_date", "1958")])
_add("Le Code civil napoléonien reste le fondement du droit français.", [
    ("hint_law", "Code civil napoléonien"), ("hint_norp", "français")])
_add("Les accords de Matignon ont mis fin à la crise en Nouvelle-Calédonie.", [
    ("hint_law", "accords de Matignon"), ("hint_gpe", "Nouvelle-Calédonie")])
_add("La Convention de Genève protège les prisonniers de guerre.", [
    ("hint_law", "Convention de Genève"), ("hint_event_nominal", "guerre")])
_add("Le protocole de Kyoto visait à réduire les émissions de gaz à effet de serre.", [
    ("hint_law", "protocole de Kyoto")])
_add("La Déclaration des droits de l'homme et du citoyen date de 1789.", [
    ("hint_law", "Déclaration des droits de l'homme et du citoyen"), ("hint_time_date", "1789")])
_add("Le traité de Maastricht a créé l'Union européenne.", [
    ("hint_law", "traité de Maastricht"), ("hint_org_name", "Union européenne")])
_add("La loi Hadopi a été votée pour lutter contre le piratage.", [("hint_law", "loi Hadopi")])
_add("Les accords d'Évian ont marqué la fin de la guerre d'Algérie.", [
    ("hint_law", "accords d'Évian"), ("hint_event_nominal", "guerre"), ("hint_gpe", "Algérie")])
_add("Le décret d'abolition de l'esclavage fut signé par Victor Schœlcher.", [
    ("hint_law", "décret d'abolition de l'esclavage"), ("hint_person_name", "Victor Schœlcher")])
_add("L'accord de Paris sur le climat a été ratifié par 196 pays.", [
    ("hint_law", "accord de Paris"), ("hint_count", "196")])
_add("La charte des Nations unies a été signée à San Francisco en 1945.", [
    ("hint_law", "charte des Nations unies"), ("hint_gpe", "San Francisco"), ("hint_time_date", "1945")])
_add("Le règlement général sur la protection des données est entré en vigueur en 2018.", [
    ("hint_law", "règlement général sur la protection des données"), ("hint_time_date", "2018")])
_add("La directive européenne sur le droit d'auteur a suscité des controverses.", [
    ("hint_law", "directive européenne sur le droit d'auteur")])
_add("L'ordonnance de Villers-Cotterêts a imposé le français dans l'administration.", [
    ("hint_law", "ordonnance de Villers-Cotterêts"), ("hint_language", "français")])
_add("Le concordat de 1801 régissait les relations entre l'État et l'Église.", [
    ("hint_law", "concordat de 1801")])
_add("La loi Taubira reconnaît la traite négrière comme crime contre l'humanité.", [
    ("hint_law", "loi Taubira")])
_add("Le pacte de Varsovie réunissait les pays du bloc soviétique.", [
    ("hint_law", "pacte de Varsovie"), ("hint_norp", "soviétique")])
_add("La loi sur la séparation des Églises et de l'État date de 1905.", [
    ("hint_law", "loi sur la séparation des Églises et de l'État"), ("hint_time_date", "1905")])
_add("Le traité de Westphalie a posé les bases du droit international moderne.", [
    ("hint_law", "traité de Westphalie")])
_add("La convention de Montego Bay régit le droit de la mer.", [
    ("hint_law", "convention de Montego Bay")])
_add("Le statut de Rome a créé la Cour pénale internationale en 1998.", [
    ("hint_law", "statut de Rome"), ("hint_org_name", "Cour pénale internationale"), ("hint_time_date", "1998")])
_add("La loi Évin interdit la publicité pour le tabac en France.", [
    ("hint_law", "loi Évin"), ("hint_gpe", "France")])
_add("Le traité de Lisbonne a réformé les institutions européennes.", [
    ("hint_law", "traité de Lisbonne"), ("hint_norp", "européennes")])
_add("La Magna Carta de 1215 est considérée comme le fondement des libertés anglaises.", [
    ("hint_law", "Magna Carta"), ("hint_time_date", "1215"), ("hint_norp", "anglaises")])
_add("Le Bill of Rights américain garantit dix amendements fondamentaux.", [
    ("hint_law", "Bill of Rights"), ("hint_norp", "américain")])
_add("Les accords de Camp David ont été signés par Sadate et Begin.", [
    ("hint_law", "accords de Camp David"), ("hint_person_name", "Sadate"), ("hint_person_name", "Begin")])
_add("Le traité de non-prolifération nucléaire est entré en vigueur en 1970.", [
    ("hint_law", "traité de non-prolifération nucléaire"), ("hint_time_date", "1970")])
_add("La Convention européenne des droits de l'homme protège les citoyens du continent.", [
    ("hint_law", "Convention européenne des droits de l'homme")])
_add("Le Code du travail régit les relations entre employeurs et salariés.", [
    ("hint_law", "Code du travail"), ("hint_group_role", "employeurs"), ("hint_group_role", "salariés")])
_add("La loi Toubon défend l'usage du français dans la vie publique.", [
    ("hint_law", "loi Toubon"), ("hint_language", "français")])
_add("Le Habeas Corpus Act de 1679 protège contre les détentions arbitraires.", [
    ("hint_law", "Habeas Corpus Act"), ("hint_time_date", "1679")])
_add("L'accord de Schengen a supprimé les frontières entre les pays signataires.", [
    ("hint_law", "accord de Schengen")])
_add("Le traité d'Utrecht a redessiné la carte de l'Europe en 1713.", [
    ("hint_law", "traité d'Utrecht"), ("hint_gpe", "Europe"), ("hint_time_date", "1713")])
_add("La loi Gayssot punit la négation des crimes contre l'humanité.", [("hint_law", "loi Gayssot")])
_add("Le pacte Briand-Kellogg de 1928 visait à mettre la guerre hors la loi.", [
    ("hint_law", "pacte Briand-Kellogg"), ("hint_time_date", "1928"), ("hint_event_nominal", "guerre")])
_add("La directive Bolkestein sur les services a fait débat au Parlement européen.", [
    ("hint_law", "directive Bolkestein"), ("hint_org_name", "Parlement européen")])
_add("Les accords de Grenelle de 1968 ont mis fin à la crise sociale en France.", [
    ("hint_law", "accords de Grenelle"), ("hint_time_date", "1968"), ("hint_gpe", "France")])
_add("Le traité de Tordesillas de 1494 partageait le monde entre l'Espagne et le Portugal.", [
    ("hint_law", "traité de Tordesillas"), ("hint_time_date", "1494"), ("hint_gpe", "Espagne"), ("hint_gpe", "Portugal")])

# ═══════════════════════════════════════════════════════════════
#  hint_work_of_art  (100 phrases)
# ═══════════════════════════════════════════════════════════════

_add("La Joconde est exposée au musée du Louvre à Paris.", [
    ("hint_work_of_art", "Joconde"), ("hint_fac_name", "musée du Louvre"), ("hint_gpe", "Paris")])
_add("Victor Hugo a écrit Les Misérables en 1862.", [
    ("hint_person_name", "Victor Hugo"), ("hint_work_of_art", "Les Misérables"), ("hint_time_date", "1862")])
_add("Le Petit Prince de Saint-Exupéry est le livre le plus traduit au monde.", [
    ("hint_work_of_art", "Petit Prince"), ("hint_person_name", "Saint-Exupéry")])
_add("La Symphonie nº 9 de Beethoven est un chef-d'œuvre de la musique classique.", [
    ("hint_work_of_art", "Symphonie nº 9"), ("hint_person_name", "Beethoven")])
_add("Le film Star Wars a révolutionné le cinéma de science-fiction.", [
    ("hint_work_of_art", "Star Wars")])
_add("Guernica de Picasso dénonce les horreurs de la guerre civile espagnole.", [
    ("hint_work_of_art", "Guernica"), ("hint_person_name", "Picasso"), ("hint_event_nominal", "guerre"), ("hint_norp", "espagnole")])
_add("Marcel Proust est l'auteur de À la recherche du temps perdu.", [
    ("hint_person_name", "Marcel Proust"), ("hint_work_of_art", "À la recherche du temps perdu")])
_add("La Flûte enchantée de Mozart fut créée à Vienne en 1791.", [
    ("hint_work_of_art", "Flûte enchantée"), ("hint_person_name", "Mozart"), ("hint_gpe", "Vienne"), ("hint_time_date", "1791")])
_add("Don Quichotte de Cervantès est considéré comme le premier roman moderne.", [
    ("hint_work_of_art", "Don Quichotte"), ("hint_person_name", "Cervantès")])
_add("Le tableau Les Demoiselles d'Avignon a marqué la naissance du cubisme.", [
    ("hint_work_of_art", "Les Demoiselles d'Avignon"), ("hint_concept", "cubisme")])
_add("La série Game of Thrones a battu des records d'audience.", [
    ("hint_work_of_art", "Game of Thrones")])
_add("L'Iliade d'Homère raconte la guerre de Troie.", [
    ("hint_work_of_art", "Iliade"), ("hint_person_name", "Homère"), ("hint_event_nominal", "guerre"), ("hint_gpe", "Troie")])
_add("Le Sacre du printemps de Stravinsky provoqua un scandale à sa création.", [
    ("hint_work_of_art", "Sacre du printemps"), ("hint_person_name", "Stravinsky")])
_add("La Vénus de Milo est une sculpture grecque exposée au Louvre.", [
    ("hint_work_of_art", "Vénus de Milo"), ("hint_norp", "grecque"), ("hint_fac_name", "Louvre")])
_add("Tintin est un personnage créé par Hergé dans les années 1920.", [
    ("hint_work_of_art", "Tintin"), ("hint_person_name", "Hergé"), ("hint_time_date", "années 1920")])
_add("Le Requiem de Verdi est une des plus grandes œuvres chorales.", [
    ("hint_work_of_art", "Requiem"), ("hint_person_name", "Verdi")])
_add("Cent ans de solitude de Gabriel García Márquez est un roman majeur.", [
    ("hint_work_of_art", "Cent ans de solitude"), ("hint_person_name", "Gabriel García Márquez")])
_add("Le Radeau de la Méduse de Géricault est un tableau monumental.", [
    ("hint_work_of_art", "Radeau de la Méduse"), ("hint_person_name", "Géricault")])
_add("Le Seigneur des anneaux a été adapté au cinéma par Peter Jackson.", [
    ("hint_work_of_art", "Seigneur des anneaux"), ("hint_person_name", "Peter Jackson")])
_add("La Neuvième de Beethoven est l'hymne officiel de l'Union européenne.", [
    ("hint_work_of_art", "Neuvième"), ("hint_person_name", "Beethoven"), ("hint_org_name", "Union européenne")])
_add("Le Lac des cygnes de Tchaïkovski reste le ballet le plus célèbre au monde.", [
    ("hint_work_of_art", "Lac des cygnes"), ("hint_person_name", "Tchaïkovski")])
_add("Les Fleurs du mal de Baudelaire ont été publiées en 1857.", [
    ("hint_work_of_art", "Fleurs du mal"), ("hint_person_name", "Baudelaire"), ("hint_time_date", "1857")])
_add("La Nuit étoilée de Van Gogh est exposée au MoMA à New York.", [
    ("hint_work_of_art", "Nuit étoilée"), ("hint_person_name", "Van Gogh"), ("hint_fac_name", "MoMA"), ("hint_gpe", "New York")])
_add("Le Penseur de Rodin est une sculpture mondialement connue.", [
    ("hint_work_of_art", "Penseur"), ("hint_person_name", "Rodin")])
_add("Madame Bovary de Flaubert a scandalisé la société du XIXe siècle.", [
    ("hint_work_of_art", "Madame Bovary"), ("hint_person_name", "Flaubert"), ("hint_time_date", "XIXe siècle")])
_add("Le Bolero de Ravel est l'une des pièces les plus jouées au monde.", [
    ("hint_work_of_art", "Bolero"), ("hint_person_name", "Ravel")])
_add("Les Quatre Saisons de Vivaldi illustrent chaque période de l'année.", [
    ("hint_work_of_art", "Quatre Saisons"), ("hint_person_name", "Vivaldi")])
_add("Carmen de Bizet est l'opéra français le plus joué dans le monde.", [
    ("hint_work_of_art", "Carmen"), ("hint_person_name", "Bizet"), ("hint_norp", "français")])
_add("Le Cri de Munch exprime l'angoisse existentielle de l'homme moderne.", [
    ("hint_work_of_art", "Cri"), ("hint_person_name", "Munch")])
_add("Harry Potter de J.K. Rowling est devenu un phénomène littéraire mondial.", [
    ("hint_work_of_art", "Harry Potter"), ("hint_person_name", "J.K. Rowling")])
_add("Les Noces de Figaro de Mozart ont été créées à Vienne en 1786.", [
    ("hint_work_of_art", "Noces de Figaro"), ("hint_person_name", "Mozart"), ("hint_gpe", "Vienne"), ("hint_time_date", "1786")])
_add("Le David de Michel-Ange est conservé à la Galerie de l'Académie à Florence.", [
    ("hint_work_of_art", "David"), ("hint_person_name", "Michel-Ange"), ("hint_fac_name", "Galerie de l'Académie"), ("hint_gpe", "Florence")])
_add("La Cène de Léonard de Vinci orne le mur du réfectoire de Santa Maria delle Grazie.", [
    ("hint_work_of_art", "Cène"), ("hint_person_name", "Léonard de Vinci"), ("hint_fac_name", "Santa Maria delle Grazie")])
_add("Hamlet de Shakespeare pose la question de l'existence humaine.", [
    ("hint_work_of_art", "Hamlet"), ("hint_person_name", "Shakespeare")])
_add("Le Jardin des délices de Bosch est un triptyque mystérieux du XVe siècle.", [
    ("hint_work_of_art", "Jardin des délices"), ("hint_person_name", "Bosch"), ("hint_time_date", "XVe siècle")])
# --- nouvelles phrases work_of_art variées ---
_add("Le Parrain de Francis Ford Coppola est considéré comme un des meilleurs films de l'histoire.", [
    ("hint_work_of_art", "Parrain"), ("hint_person_name", "Francis Ford Coppola")])
_add("La Traviata de Verdi est jouée chaque année à l'Opéra de Vienne.", [
    ("hint_work_of_art", "Traviata"), ("hint_person_name", "Verdi"), ("hint_fac_name", "Opéra de Vienne")])
_add("Le Fantôme de l'Opéra est l'une des comédies musicales les plus longtemps jouées à Broadway.", [
    ("hint_work_of_art", "Fantôme de l'Opéra"), ("hint_fac_name", "Broadway")])
_add("Blade Runner de Ridley Scott a redéfini le genre de la science-fiction au cinéma.", [
    ("hint_work_of_art", "Blade Runner"), ("hint_person_name", "Ridley Scott")])
_add("Les Contemplations de Victor Hugo expriment la douleur de la perte.", [
    ("hint_work_of_art", "Contemplations"), ("hint_person_name", "Victor Hugo")])
_add("Astérix et Obélix sont des personnages créés par Goscinny et Uderzo.", [
    ("hint_work_of_art", "Astérix et Obélix"), ("hint_person_name", "Goscinny"), ("hint_person_name", "Uderzo")])
_add("Le Messie de Haendel est traditionnellement joué à Noël dans les pays anglophones.", [
    ("hint_work_of_art", "Messie"), ("hint_person_name", "Haendel")])
_add("Le Petit Nicolas de Sempé et Goscinny reste un classique de la littérature jeunesse.", [
    ("hint_work_of_art", "Petit Nicolas"), ("hint_person_name", "Sempé"), ("hint_person_name", "Goscinny")])
_add("L'Odyssée d'Homère a inspiré des milliers d'œuvres à travers les siècles.", [
    ("hint_work_of_art", "Odyssée"), ("hint_person_name", "Homère")])
_add("Le Baiser de Klimt est l'une des peintures les plus reproduites au monde.", [
    ("hint_work_of_art", "Baiser"), ("hint_person_name", "Klimt")])
_add("Moby Dick de Herman Melville raconte la traque obsessionnelle d'une baleine blanche.", [
    ("hint_work_of_art", "Moby Dick"), ("hint_person_name", "Herman Melville")])
_add("La Création d'Adam de Michel-Ange orne le plafond de la chapelle Sixtine.", [
    ("hint_work_of_art", "Création d'Adam"), ("hint_person_name", "Michel-Ange"), ("hint_fac_name", "chapelle Sixtine")])
_add("Les Liaisons dangereuses de Laclos ont été adaptées au cinéma plusieurs fois.", [
    ("hint_work_of_art", "Liaisons dangereuses"), ("hint_person_name", "Laclos")])
_add("La Marche impériale de John Williams est devenue un thème iconique de la culture pop.", [
    ("hint_work_of_art", "Marche impériale"), ("hint_person_name", "John Williams")])
_add("Le Procès de Kafka explore les méandres d'une bureaucratie absurde et oppressante.", [
    ("hint_work_of_art", "Procès"), ("hint_person_name", "Kafka")])
_add("Citizen Kane d'Orson Welles est souvent cité comme le meilleur film de tous les temps.", [
    ("hint_work_of_art", "Citizen Kane"), ("hint_person_name", "Orson Welles")])
_add("La Persistance de la mémoire de Dalí représente des montres molles dans un paysage onirique.", [
    ("hint_work_of_art", "Persistance de la mémoire"), ("hint_person_name", "Dalí")])
_add("Le Petit Chaperon rouge est un conte popularisé par Charles Perrault.", [
    ("hint_work_of_art", "Petit Chaperon rouge"), ("hint_person_name", "Charles Perrault")])
_add("Faust de Goethe interroge les limites de la connaissance humaine.", [
    ("hint_work_of_art", "Faust"), ("hint_person_name", "Goethe")])
_add("Le Désert des Tartares de Dino Buzzati explore l'attente et le vide existentiel.", [
    ("hint_work_of_art", "Désert des Tartares"), ("hint_person_name", "Dino Buzzati")])
_add("Matrix des sœurs Wachowski a popularisé le concept de simulation informatique.", [
    ("hint_work_of_art", "Matrix"), ("hint_person_name", "Wachowski")])
_add("2001, l'Odyssée de l'espace de Kubrick reste un monument du cinéma de science-fiction.", [
    ("hint_work_of_art", "2001, l'Odyssée de l'espace"), ("hint_person_name", "Kubrick")])
_add("Le Malade imaginaire de Molière est une comédie satirique sur la médecine.", [
    ("hint_work_of_art", "Malade imaginaire"), ("hint_person_name", "Molière")])
_add("Le Conte de deux cités de Dickens se déroule pendant la Révolution française.", [
    ("hint_work_of_art", "Conte de deux cités"), ("hint_person_name", "Dickens"), ("hint_event_named", "Révolution française")])
_add("Roméo et Juliette de Shakespeare est la tragédie romantique la plus célèbre.", [
    ("hint_work_of_art", "Roméo et Juliette"), ("hint_person_name", "Shakespeare")])
_add("Le Portrait de Dorian Gray d'Oscar Wilde explore la vanité et la corruption morale.", [
    ("hint_work_of_art", "Portrait de Dorian Gray"), ("hint_person_name", "Oscar Wilde")])
_add("L'Attrape-cœurs de Salinger est devenu le symbole de la rébellion adolescente.", [
    ("hint_work_of_art", "Attrape-cœurs"), ("hint_person_name", "Salinger")])
_add("Les Nymphéas de Monet sont exposés au musée de l'Orangerie à Paris.", [
    ("hint_work_of_art", "Nymphéas"), ("hint_person_name", "Monet"), ("hint_fac_name", "musée de l'Orangerie"), ("hint_gpe", "Paris")])
_add("Le Hobbit de Tolkien a été publié en 1937 comme livre pour enfants.", [
    ("hint_work_of_art", "Hobbit"), ("hint_person_name", "Tolkien"), ("hint_time_date", "1937")])
_add("Crime et Châtiment de Dostoïevski est un roman psychologique majeur.", [
    ("hint_work_of_art", "Crime et Châtiment"), ("hint_person_name", "Dostoïevski")])
_add("Pulp Fiction de Quentin Tarantino a redéfini le cinéma indépendant des années 1990.", [
    ("hint_work_of_art", "Pulp Fiction"), ("hint_person_name", "Quentin Tarantino"), ("hint_time_date", "années 1990")])
_add("Le Voyage au bout de la nuit de Céline a marqué la littérature du XXe siècle.", [
    ("hint_work_of_art", "Voyage au bout de la nuit"), ("hint_person_name", "Céline"), ("hint_time_date", "XXe siècle")])
_add("La Pietà de Michel-Ange est conservée dans la basilique Saint-Pierre à Rome.", [
    ("hint_work_of_art", "Pietà"), ("hint_person_name", "Michel-Ange"), ("hint_fac_name", "basilique Saint-Pierre"), ("hint_gpe", "Rome")])
_add("Anna Karénine de Tolstoï explore les tensions entre passion et convention sociale.", [
    ("hint_work_of_art", "Anna Karénine"), ("hint_person_name", "Tolstoï")])
_add("Le Meilleur des mondes d'Aldous Huxley décrit une société dystopique contrôlée par la science.", [
    ("hint_work_of_art", "Meilleur des mondes"), ("hint_person_name", "Aldous Huxley")])
_add("L'Étranger d'Albert Camus est un roman fondateur de l'absurde.", [
    ("hint_work_of_art", "Étranger"), ("hint_person_name", "Albert Camus")])
_add("Les Ménines de Vélasquez sont considérées comme l'un des tableaux les plus analysés de l'histoire.", [
    ("hint_work_of_art", "Ménines"), ("hint_person_name", "Vélasquez")])
_add("La Gioconda de Léonard de Vinci est le tableau le plus visité du Louvre.", [
    ("hint_work_of_art", "Gioconda"), ("hint_person_name", "Léonard de Vinci"), ("hint_fac_name", "Louvre")])
_add("Spirited Away du studio Ghibli a remporté l'Oscar du meilleur film d'animation.", [
    ("hint_work_of_art", "Spirited Away"), ("hint_org_name", "studio Ghibli")])
_add("Les Raisins de la colère de Steinbeck dépeint la misère des migrants américains.", [
    ("hint_work_of_art", "Raisins de la colère"), ("hint_person_name", "Steinbeck"), ("hint_norp", "américains")])
_add("La Bohème de Puccini se déroule dans le Paris du XIXe siècle.", [
    ("hint_work_of_art", "Bohème"), ("hint_person_name", "Puccini"), ("hint_gpe", "Paris"), ("hint_time_date", "XIXe siècle")])
_add("Le Vieil Homme et la Mer d'Hemingway a valu à son auteur le prix Pulitzer.", [
    ("hint_work_of_art", "Vieil Homme et la Mer"), ("hint_person_name", "Hemingway")])
_add("Le Château de Kafka est resté inachevé à la mort de l'auteur en 1924.", [
    ("hint_work_of_art", "Château"), ("hint_person_name", "Kafka"), ("hint_time_date", "1924")])
_add("La Divine Comédie de Dante a influencé toute la littérature occidentale.", [
    ("hint_work_of_art", "Divine Comédie"), ("hint_person_name", "Dante")])
_add("Apocalypse Now de Coppola est librement inspiré de Au cœur des ténèbres de Conrad.", [
    ("hint_work_of_art", "Apocalypse Now"), ("hint_person_name", "Coppola"), ("hint_work_of_art", "Au cœur des ténèbres"), ("hint_person_name", "Conrad")])
_add("Les Trois Mousquetaires d'Alexandre Dumas est un classique du roman d'aventures.", [
    ("hint_work_of_art", "Trois Mousquetaires"), ("hint_person_name", "Alexandre Dumas")])
_add("Le Sacre de Napoléon peint par David est exposé au Louvre.", [
    ("hint_work_of_art", "Sacre de Napoléon"), ("hint_person_name", "David"), ("hint_fac_name", "Louvre")])
_add("Inception de Christopher Nolan explore les rêves imbriqués les uns dans les autres.", [
    ("hint_work_of_art", "Inception"), ("hint_person_name", "Christopher Nolan")])
_add("Les Métamorphoses d'Ovide sont un poème fondateur de la mythologie romaine.", [
    ("hint_work_of_art", "Métamorphoses"), ("hint_person_name", "Ovide"), ("hint_norp", "romaine")])
_add("La Marseillaise est devenue l'hymne national français en 1795.", [
    ("hint_work_of_art", "Marseillaise"), ("hint_norp", "français"), ("hint_time_date", "1795")])

# ═══════════════════════════════════════════════════════════════
#  hint_concept  (100 phrases)
# ═══════════════════════════════════════════════════════════════

_add("La théorie de la relativité d'Einstein a bouleversé la physique.", [
    ("hint_concept", "théorie de la relativité"), ("hint_person_name", "Einstein")])
_add("Le marxisme a influencé de nombreux mouvements sociaux au XXe siècle.", [
    ("hint_concept", "marxisme"), ("hint_time_date", "XXe siècle")])
_add("Le darwinisme explique l'évolution des espèces par la sélection naturelle.", [
    ("hint_concept", "darwinisme")])
_add("La psychanalyse a été fondée par Sigmund Freud à Vienne.", [
    ("hint_concept", "psychanalyse"), ("hint_person_name", "Sigmund Freud"), ("hint_gpe", "Vienne")])
_add("Le libéralisme économique prône la libre concurrence.", [
    ("hint_concept", "libéralisme économique")])
_add("La mécanique quantique décrit le comportement des particules subatomiques.", [
    ("hint_concept", "mécanique quantique")])
_add("Le structuralisme a dominé les sciences humaines dans les années 1960.", [
    ("hint_concept", "structuralisme"), ("hint_time_date", "années 1960")])
_add("L'existentialisme de Sartre a marqué la pensée française d'après-guerre.", [
    ("hint_concept", "existentialisme"), ("hint_person_name", "Sartre"), ("hint_norp", "française")])
_add("Le keynésianisme recommande l'intervention de l'État dans l'économie.", [
    ("hint_concept", "keynésianisme")])
_add("La laïcité est un principe fondamental de la République française.", [
    ("hint_concept", "laïcité"), ("hint_norp", "française")])
_add("Le féminisme revendique l'égalité des droits entre les sexes.", [
    ("hint_concept", "féminisme")])
_add("L'empirisme considère que toute connaissance vient de l'expérience.", [
    ("hint_concept", "empirisme")])
_add("Le positivisme d'Auguste Comte a fondé la sociologie moderne.", [
    ("hint_concept", "positivisme"), ("hint_person_name", "Auguste Comte")])
_add("La théorie des cordes tente d'unifier la physique quantique et la relativité.", [
    ("hint_concept", "théorie des cordes")])
_add("Le capitalisme et le socialisme s'affrontent depuis le XIXe siècle.", [
    ("hint_concept", "capitalisme"), ("hint_concept", "socialisme"), ("hint_time_date", "XIXe siècle")])
_add("L'intelligence artificielle repose sur l'apprentissage automatique.", [
    ("hint_concept", "intelligence artificielle"), ("hint_concept", "apprentissage automatique")])
_add("Le stoïcisme prônait la maîtrise de soi et l'acceptation du destin.", [
    ("hint_concept", "stoïcisme")])
_add("La théorie du Big Bang explique l'origine de l'univers.", [
    ("hint_concept", "théorie du Big Bang")])
_add("Le nihilisme de Nietzsche a remis en question les valeurs traditionnelles.", [
    ("hint_concept", "nihilisme"), ("hint_person_name", "Nietzsche")])
_add("L'utilitarisme de Bentham évalue les actions par leurs conséquences.", [
    ("hint_concept", "utilitarisme"), ("hint_person_name", "Bentham")])
# --- nouvelles phrases concept variées ---
_add("Le rationalisme de Descartes a posé les fondements de la philosophie moderne.", [
    ("hint_concept", "rationalisme"), ("hint_person_name", "Descartes")])
_add("L'anarchisme prône l'abolition de toute forme d'autorité étatique.", [
    ("hint_concept", "anarchisme")])
_add("Le romantisme a révolutionné la littérature et les arts au XIXe siècle.", [
    ("hint_concept", "romantisme"), ("hint_time_date", "XIXe siècle")])
_add("Le déterminisme soutient que tout événement est causalement nécessaire.", [
    ("hint_concept", "déterminisme")])
_add("Le pragmatisme de William James met l'accent sur les conséquences pratiques.", [
    ("hint_concept", "pragmatisme"), ("hint_person_name", "William James")])
_add("L'humanisme de la Renaissance a placé l'homme au centre de la réflexion.", [
    ("hint_concept", "humanisme"), ("hint_event_named", "Renaissance")])
_add("Le behaviorisme étudie le comportement observable sans recourir à l'introspection.", [
    ("hint_concept", "behaviorisme")])
_add("La phénoménologie de Husserl cherche à décrire les structures de la conscience.", [
    ("hint_concept", "phénoménologie"), ("hint_person_name", "Husserl")])
_add("Le matérialisme dialectique est au fondement de la pensée marxiste.", [
    ("hint_concept", "matérialisme dialectique"), ("hint_norp", "marxiste")])
_add("Le constructivisme en éducation insiste sur le rôle actif de l'apprenant.", [
    ("hint_concept", "constructivisme")])
_add("La déconstruction de Derrida a transformé la critique littéraire.", [
    ("hint_concept", "déconstruction"), ("hint_person_name", "Derrida")])
_add("Le transhumanisme envisage l'amélioration de l'homme par la technologie.", [
    ("hint_concept", "transhumanisme")])
_add("L'altermondialisme s'oppose à la mondialisation néolibérale.", [
    ("hint_concept", "altermondialisme"), ("hint_concept", "mondialisation néolibérale")])
_add("Le surréalisme d'André Breton a bouleversé l'art et la littérature.", [
    ("hint_concept", "surréalisme"), ("hint_person_name", "André Breton")])
_add("Le relativisme culturel considère qu'aucune culture n'est supérieure à une autre.", [
    ("hint_concept", "relativisme culturel")])
_add("Le communisme prétend abolir la propriété privée des moyens de production.", [
    ("hint_concept", "communisme")])
_add("Le fascisme est apparu en Italie dans les années 1920 sous Mussolini.", [
    ("hint_concept", "fascisme"), ("hint_gpe", "Italie"), ("hint_time_date", "années 1920"), ("hint_person_name", "Mussolini")])
_add("Le populisme exploite le mécontentement des classes populaires contre les élites.", [
    ("hint_concept", "populisme")])
_add("Le néolibéralisme domine les politiques économiques depuis les années 1980.", [
    ("hint_concept", "néolibéralisme"), ("hint_time_date", "années 1980")])
_add("Le monothéisme est apparu au Moyen-Orient il y a plusieurs millénaires.", [
    ("hint_concept", "monothéisme"), ("hint_gpe", "Moyen-Orient")])
_add("Le polythéisme caractérisait les religions de la Grèce et de la Rome antiques.", [
    ("hint_concept", "polythéisme"), ("hint_gpe", "Grèce"), ("hint_gpe", "Rome")])
_add("L'impressionnisme a transformé la peinture française à la fin du XIXe siècle.", [
    ("hint_concept", "impressionnisme"), ("hint_norp", "française"), ("hint_time_date", "XIXe siècle")])
_add("Le réalisme en littérature cherche à représenter le monde tel qu'il est.", [
    ("hint_concept", "réalisme")])
_add("Le naturalisme de Zola prolonge le réalisme par une approche scientifique.", [
    ("hint_concept", "naturalisme"), ("hint_person_name", "Zola")])
_add("Le classicisme a dominé les arts en France sous le règne de Louis XIV.", [
    ("hint_concept", "classicisme"), ("hint_gpe", "France"), ("hint_person_name", "Louis XIV")])
_add("Le dadaïsme est né à Zurich pendant la Première Guerre mondiale.", [
    ("hint_concept", "dadaïsme"), ("hint_gpe", "Zurich"), ("hint_event_named", "Première Guerre mondiale")])
_add("Le minimalisme en musique se caractérise par la répétition de motifs simples.", [
    ("hint_concept", "minimalisme")])
_add("L'expressionnisme allemand a marqué le cinéma du début du XXe siècle.", [
    ("hint_concept", "expressionnisme"), ("hint_norp", "allemand"), ("hint_time_date", "XXe siècle")])
_add("Le colonialisme a profondément transformé les sociétés africaines et asiatiques.", [
    ("hint_concept", "colonialisme"), ("hint_norp", "africaines"), ("hint_norp", "asiatiques")])
_add("L'impérialisme européen a redessiné les frontières du monde au XIXe siècle.", [
    ("hint_concept", "impérialisme"), ("hint_norp", "européen"), ("hint_time_date", "XIXe siècle")])
_add("Le nationalisme a été un moteur des révolutions du XIXe siècle en Europe.", [
    ("hint_concept", "nationalisme"), ("hint_time_date", "XIXe siècle"), ("hint_gpe", "Europe")])
_add("Le fédéralisme organise le partage du pouvoir entre l'État central et les régions.", [
    ("hint_concept", "fédéralisme")])
_add("Le mercantilisme favorisait l'accumulation de métaux précieux par l'État.", [
    ("hint_concept", "mercantilisme")])
_add("Le monétarisme de Milton Friedman a influencé les politiques de la Fed.", [
    ("hint_concept", "monétarisme"), ("hint_person_name", "Milton Friedman"), ("hint_org_name", "Fed")])
_add("Le protectionnisme vise à protéger l'industrie nationale par des droits de douane.", [
    ("hint_concept", "protectionnisme")])
_add("L'épicurisme recherche le bonheur à travers l'absence de souffrance.", [
    ("hint_concept", "épicurisme")])
_add("Le scepticisme remet en cause la possibilité d'atteindre une vérité certaine.", [
    ("hint_concept", "scepticisme")])
_add("Le confucianisme a façonné la pensée morale et politique de la Chine.", [
    ("hint_concept", "confucianisme"), ("hint_gpe", "Chine")])
_add("Le taoïsme prône l'harmonie avec la nature et le non-agir.", [
    ("hint_concept", "taoïsme")])
_add("Le shintoïsme est la religion autochtone du Japon.", [
    ("hint_concept", "shintoïsme"), ("hint_gpe", "Japon")])
_add("Le soufisme est la dimension mystique de l'islam.", [
    ("hint_concept", "soufisme"), ("hint_concept", "islam")])
_add("Le protestantisme est né de la Réforme lancée par Martin Luther.", [
    ("hint_concept", "protestantisme"), ("hint_event_named", "Réforme"), ("hint_person_name", "Martin Luther")])
_add("Le catholicisme reste la confession la plus répandue en Amérique latine.", [
    ("hint_concept", "catholicisme"), ("hint_gpe", "Amérique latine")])
_add("Le bouddhisme enseigne les quatre nobles vérités et le chemin de la libération.", [
    ("hint_concept", "bouddhisme")])
_add("L'hindouisme est la troisième religion du monde par le nombre de fidèles.", [
    ("hint_concept", "hindouisme")])
_add("Le véganisme va au-delà du végétarisme en excluant tout produit animal.", [
    ("hint_concept", "véganisme"), ("hint_concept", "végétarisme")])
_add("L'abolitionnisme a mené à l'interdiction de l'esclavage aux États-Unis.", [
    ("hint_concept", "abolitionnisme"), ("hint_gpe", "États-Unis")])
_add("Le pacifisme refuse le recours à la violence comme moyen politique.", [
    ("hint_concept", "pacifisme")])
_add("Le multilatéralisme privilégie la coopération entre plusieurs États.", [
    ("hint_concept", "multilatéralisme")])
_add("La souveraineté nationale est un concept central du droit international.", [
    ("hint_concept", "souveraineté nationale")])
_add("Le panafricanisme vise à unir les peuples du continent africain.", [
    ("hint_concept", "panafricanisme"), ("hint_norp", "africain")])
_add("Le sionisme a abouti à la création de l'État d'Israël en 1948.", [
    ("hint_concept", "sionisme"), ("hint_gpe", "Israël"), ("hint_time_date", "1948")])
_add("L'eugénisme a été utilisé pour justifier des politiques raciales au XXe siècle.", [
    ("hint_concept", "eugénisme"), ("hint_time_date", "XXe siècle")])
_add("Le darwinisme social a détourné la théorie de Darwin pour justifier les inégalités.", [
    ("hint_concept", "darwinisme social"), ("hint_person_name", "Darwin")])
_add("Le biomimétisme s'inspire des solutions de la nature pour innover.", [
    ("hint_concept", "biomimétisme")])
_add("L'économie circulaire cherche à éliminer le concept de déchet.", [
    ("hint_concept", "économie circulaire")])
_add("La décroissance remet en cause le dogme de la croissance économique infinie.", [
    ("hint_concept", "décroissance")])
_add("Le développement durable concilie progrès économique, social et environnemental.", [
    ("hint_concept", "développement durable")])
_add("Le deep learning a révolutionné la reconnaissance d'images et la traduction automatique.", [
    ("hint_concept", "deep learning")])
_add("La blockchain est une technologie de registre distribué et décentralisé.", [
    ("hint_concept", "blockchain")])
_add("L'open source encourage le partage libre du code informatique.", [
    ("hint_concept", "open source")])

# ═══════════════════════════════════════════════════════════════
#  hint_disease  (40 phrases)
# ═══════════════════════════════════════════════════════════════

_add("La pandémie de Covid-19 a paralysé l'économie mondiale en 2020.", [
    ("hint_event_nominal", "pandémie"), ("hint_disease", "Covid-19"), ("hint_time_date", "2020")])
_add("La grippe espagnole a fait des millions de morts en 1918.", [
    ("hint_disease", "grippe espagnole"), ("hint_time_date", "1918")])
_add("La maladie d'Alzheimer touche des millions de personnes âgées.", [("hint_disease", "maladie d'Alzheimer")])
_add("Le paludisme reste endémique dans de nombreuses régions tropicales.", [("hint_disease", "paludisme")])
_add("Le virus Ebola a causé une épidémie meurtrière en Afrique de l'Ouest.", [
    ("hint_disease", "virus Ebola"), ("hint_event_nominal", "épidémie"), ("hint_gpe", "Afrique de l'Ouest")])
_add("La tuberculose était surnommée la peste blanche au XIXe siècle.", [
    ("hint_disease", "tuberculose"), ("hint_time_date", "XIXe siècle")])
_add("Le diabète de type 2 est lié à l'obésité et à la sédentarité.", [("hint_disease", "diabète de type 2")])
_add("La peste noire a décimé un tiers de la population européenne.", [
    ("hint_disease", "peste noire"), ("hint_norp", "européenne")])
_add("Le choléra a ravagé Paris lors de l'épidémie de 1832.", [
    ("hint_disease", "choléra"), ("hint_gpe", "Paris"), ("hint_event_nominal", "épidémie"), ("hint_time_date", "1832")])
_add("La variole a été éradiquée grâce à une campagne mondiale de vaccination.", [("hint_disease", "variole")])
_add("Le syndrome de Down est la trisomie la plus fréquente.", [("hint_disease", "syndrome de Down")])
_add("La maladie de Parkinson affecte le système nerveux central.", [("hint_disease", "maladie de Parkinson")])
_add("L'épidémie de SRAS en 2003 a mis en alerte les systèmes de santé.", [
    ("hint_event_nominal", "épidémie"), ("hint_disease", "SRAS"), ("hint_time_date", "2003")])
_add("Le cancer du poumon est la première cause de mortalité par cancer.", [("hint_disease", "cancer du poumon")])
_add("La rougeole connaît une recrudescence en Europe faute de vaccination.", [
    ("hint_disease", "rougeole"), ("hint_gpe", "Europe")])
_add("La dengue est transmise par le moustique Aedes aegypti.", [("hint_disease", "dengue")])
_add("Le virus Zika a provoqué des malformations chez les nouveau-nés au Brésil.", [
    ("hint_disease", "virus Zika"), ("hint_gpe", "Brésil")])
_add("La sclérose en plaques est une maladie auto-immune du système nerveux.", [("hint_disease", "sclérose en plaques")])
_add("Le VIH a été identifié pour la première fois en 1983 par l'équipe de Montagnier.", [
    ("hint_disease", "VIH"), ("hint_time_date", "1983"), ("hint_person_name", "Montagnier")])
_add("La fièvre jaune sévit encore dans les zones tropicales d'Afrique et d'Amérique.", [
    ("hint_disease", "fièvre jaune"), ("hint_gpe", "Afrique"), ("hint_gpe", "Amérique")])
_add("La méningite bactérienne nécessite un traitement antibiotique en urgence.", [("hint_disease", "méningite bactérienne")])
_add("Le tétanos est évité grâce au rappel vaccinal tous les dix ans.", [
    ("hint_disease", "tétanos"), ("hint_time_duration", "dix ans")])
_add("La lèpre, aussi appelée maladie de Hansen, est traitée par antibiotiques.", [
    ("hint_disease", "lèpre"), ("hint_disease", "maladie de Hansen")])
_add("L'hépatite B se transmet par le sang et les fluides corporels.", [("hint_disease", "hépatite B")])
_add("La diphtérie a quasiment disparu grâce à la vaccination systématique.", [("hint_disease", "diphtérie")])
_add("Le syndrome de Guillain-Barré provoque une paralysie ascendante.", [("hint_disease", "syndrome de Guillain-Barré")])
_add("La coqueluche reste dangereuse pour les nourrissons non vaccinés.", [("hint_disease", "coqueluche")])
_add("L'asthme touche environ 300 millions de personnes dans le monde.", [
    ("hint_disease", "asthme"), ("hint_quantity", "300 millions")])
_add("La mucoviscidose est la maladie génétique la plus fréquente en Europe.", [
    ("hint_disease", "mucoviscidose"), ("hint_gpe", "Europe")])
_add("Le lupus érythémateux est une maladie auto-immune systémique.", [("hint_disease", "lupus érythémateux")])
_add("Le choléra a tué des milliers de personnes à Londres au XIXe siècle.", [
    ("hint_disease", "choléra"), ("hint_gpe", "Londres"), ("hint_time_date", "XIXe siècle")])
_add("La bilharziose touche plus de 200 millions de personnes en Afrique.", [
    ("hint_disease", "bilharziose"), ("hint_quantity", "200 millions"), ("hint_gpe", "Afrique")])
_add("Le syndrome métabolique augmente le risque de maladie cardiovasculaire.", [("hint_disease", "syndrome métabolique")])
_add("La trypanosomiase africaine est transmise par la mouche tsé-tsé.", [("hint_disease", "trypanosomiase africaine")])
_add("La maladie de Crohn provoque une inflammation chronique de l'intestin.", [("hint_disease", "maladie de Crohn")])
_add("L'endométriose touche environ une femme sur dix en âge de procréer.", [("hint_disease", "endométriose")])
_add("La maladie de Lyme est transmise par les tiques infectées.", [("hint_disease", "maladie de Lyme")])
_add("Le SIDA a tué plus de 36 millions de personnes depuis le début de l'épidémie.", [
    ("hint_disease", "SIDA"), ("hint_quantity", "36 millions"), ("hint_event_nominal", "épidémie")])
_add("La grippe aviaire H5N1 inquiète les autorités sanitaires mondiales.", [("hint_disease", "grippe aviaire H5N1")])
_add("La fibromyalgie provoque des douleurs diffuses et une fatigue chronique.", [("hint_disease", "fibromyalgie")])

# ═══════════════════════════════════════════════════════════════
#  hint_language  (40 phrases)
# ═══════════════════════════════════════════════════════════════

_add("Le français est parlé sur les cinq continents.", [("hint_language", "français")])
_add("Le mandarin est la langue la plus parlée au monde.", [("hint_language", "mandarin")])
_add("Le latin a donné naissance aux langues romanes.", [("hint_language", "latin")])
_add("L'espéranto a été créé en 1887 par Zamenhof.", [
    ("hint_language", "espéranto"), ("hint_time_date", "1887"), ("hint_person_name", "Zamenhof")])
_add("Le basque est l'une des plus anciennes langues d'Europe.", [
    ("hint_language", "basque"), ("hint_gpe", "Europe")])
_add("L'arabe est la langue liturgique de l'islam.", [
    ("hint_language", "arabe"), ("hint_concept", "islam")])
_add("Le swahili est une langue bantoue parlée en Afrique de l'Est.", [
    ("hint_language", "swahili"), ("hint_gpe", "Afrique de l'Est")])
_add("Le créole haïtien est né du contact entre le français et les langues africaines.", [
    ("hint_language", "créole haïtien"), ("hint_language", "français")])
_add("Le sanskrit est la langue classique de l'Inde.", [
    ("hint_language", "sanskrit"), ("hint_gpe", "Inde")])
_add("L'occitan était autrefois la langue dominante du sud de la France.", [
    ("hint_language", "occitan"), ("hint_gpe", "France")])
_add("Le gaélique irlandais est une langue celtique en déclin.", [("hint_language", "gaélique irlandais")])
_add("Le japonais utilise trois systèmes d'écriture différents.", [("hint_language", "japonais")])
_add("Le catalan est co-officiel en Catalogne avec l'espagnol.", [
    ("hint_language", "catalan"), ("hint_gpe", "Catalogne"), ("hint_language", "espagnol")])
_add("L'hébreu a été revitalisé comme langue parlée au XIXe siècle.", [
    ("hint_language", "hébreu"), ("hint_time_date", "XIXe siècle")])
_add("Le wolof est la langue la plus utilisée au Sénégal.", [
    ("hint_language", "wolof"), ("hint_gpe", "Sénégal")])
_add("Le breton est une langue régionale menacée de disparition.", [("hint_language", "breton")])
_add("Le portugais est la sixième langue la plus parlée dans le monde.", [("hint_language", "portugais")])
_add("Le quechua était la langue de l'Empire inca.", [
    ("hint_language", "quechua"), ("hint_org_name", "Empire inca")])
_add("L'allemand est la langue maternelle la plus répandue en Europe.", [
    ("hint_language", "allemand"), ("hint_gpe", "Europe")])
_add("Le hindi et l'ourdou sont mutuellement intelligibles à l'oral.", [
    ("hint_language", "hindi"), ("hint_language", "ourdou")])
_add("Le tagalog est la base du filipino, la langue nationale des Philippines.", [
    ("hint_language", "tagalog"), ("hint_language", "filipino"), ("hint_gpe", "Philippines")])
_add("Le turc est une langue agglutinante parlée par 80 millions de personnes.", [
    ("hint_language", "turc"), ("hint_quantity", "80 millions")])
_add("Le finnois et l'estonien sont des langues finno-ougriennes apparentées.", [
    ("hint_language", "finnois"), ("hint_language", "estonien")])
_add("Le yiddish a été la lingua franca des communautés juives d'Europe.", [
    ("hint_language", "yiddish"), ("hint_norp", "juives"), ("hint_gpe", "Europe")])
_add("Le coréen utilise un alphabet unique appelé hangul.", [("hint_language", "coréen")])
_add("Le tamoul est l'une des langues classiques les plus anciennes au monde.", [("hint_language", "tamoul")])
_add("Le roumain est la seule langue romane d'Europe de l'Est.", [
    ("hint_language", "roumain"), ("hint_gpe", "Europe de l'Est")])
_add("Le thaï est une langue tonale à l'écriture complexe.", [("hint_language", "thaï")])
_add("Le persan est parlé en Iran, en Afghanistan et au Tadjikistan.", [
    ("hint_language", "persan"), ("hint_gpe", "Iran"), ("hint_gpe", "Afghanistan"), ("hint_gpe", "Tadjikistan")])
_add("Le néerlandais est la langue officielle des Pays-Bas et de la Belgique flamande.", [
    ("hint_language", "néerlandais"), ("hint_gpe", "Pays-Bas"), ("hint_gpe", "Belgique flamande")])
_add("Le guarani est co-officiel avec l'espagnol au Paraguay.", [
    ("hint_language", "guarani"), ("hint_language", "espagnol"), ("hint_gpe", "Paraguay")])
_add("Le tibétain est une langue sino-tibétaine écrite en alphabet dérivé du sanskrit.", [
    ("hint_language", "tibétain"), ("hint_language", "sanskrit")])
_add("Le norvégien existe sous deux formes écrites : le bokmål et le nynorsk.", [
    ("hint_language", "norvégien"), ("hint_language", "bokmål"), ("hint_language", "nynorsk")])
_add("Le maltais est la seule langue sémitique écrite en alphabet latin.", [
    ("hint_language", "maltais"), ("hint_language", "latin")])
_add("Le corse est reconnu comme langue régionale de France.", [
    ("hint_language", "corse"), ("hint_gpe", "France")])
_add("Le bengali est la langue officielle du Bangladesh et du Bengale-Occidental.", [
    ("hint_language", "bengali"), ("hint_gpe", "Bangladesh"), ("hint_gpe", "Bengale-Occidental")])
_add("Le tchèque et le slovaque sont des langues slaves mutuellement compréhensibles.", [
    ("hint_language", "tchèque"), ("hint_language", "slovaque")])
_add("Le gallois connaît un renouveau grâce aux politiques linguistiques du Pays de Galles.", [
    ("hint_language", "gallois"), ("hint_gpe", "Pays de Galles")])
_add("L'indonésien est la langue véhiculaire de l'archipel le plus peuplé du monde.", [
    ("hint_language", "indonésien")])
_add("Le grec ancien est la langue d'Homère, de Platon et d'Aristote.", [
    ("hint_language", "grec ancien"), ("hint_person_name", "Homère"), ("hint_person_name", "Platon"), ("hint_person_name", "Aristote")])

# ═══════════════════════════════════════════════════════════════
#  MIX — phrases croisant plusieurs labels ABSTRACT
# ═══════════════════════════════════════════════════════════════

_add("La théorie de l'évolution de Darwin a été publiée dans De l'origine des espèces en 1859.", [
    ("hint_concept", "théorie de l'évolution"), ("hint_person_name", "Darwin"),
    ("hint_work_of_art", "De l'origine des espèces"), ("hint_time_date", "1859")])
_add("La Constitution américaine garantit la liberté d'expression en anglais.", [
    ("hint_law", "Constitution américaine"), ("hint_language", "anglais")])
_add("Le SARS-CoV-2 a été séquencé pour la première fois à Wuhan en janvier 2020.", [
    ("hint_disease", "SARS-CoV-2"), ("hint_gpe", "Wuhan"), ("hint_time_date", "janvier 2020")])
_add("Albert Camus a reçu le prix Nobel pour L'Étranger et La Peste.", [
    ("hint_person_name", "Albert Camus"), ("hint_work_of_art", "L'Étranger"), ("hint_work_of_art", "La Peste")])
_add("Le romantisme s'est développé en réaction au classicisme des Lumières.", [
    ("hint_concept", "romantisme"), ("hint_concept", "classicisme")])
_add("Le traité de Rome a fondé la Communauté économique européenne en 1957.", [
    ("hint_law", "traité de Rome"), ("hint_org_name", "Communauté économique européenne"), ("hint_time_date", "1957")])
_add("Kant a développé l'idéalisme transcendantal dans la Critique de la raison pure.", [
    ("hint_person_name", "Kant"), ("hint_concept", "idéalisme transcendantal"),
    ("hint_work_of_art", "Critique de la raison pure")])
_add("La loi Veil de 1975 a légalisé l'interruption volontaire de grossesse en France.", [
    ("hint_law", "loi Veil"), ("hint_time_date", "1975"), ("hint_gpe", "France")])
_add("Le réalisme magique de García Márquez a transformé la littérature latino-américaine.", [
    ("hint_concept", "réalisme magique"), ("hint_person_name", "García Márquez")])
_add("La peste bubonique a ravagé l'Europe au XIVe siècle et inspiré le Décaméron de Boccace.", [
    ("hint_disease", "peste bubonique"), ("hint_gpe", "Europe"), ("hint_time_date", "XIVe siècle"),
    ("hint_work_of_art", "Décaméron"), ("hint_person_name", "Boccace")])
_add("Le Code Justinien a codifié le droit romain au VIe siècle.", [
    ("hint_law", "Code Justinien"), ("hint_norp", "romain"), ("hint_time_date", "VIe siècle")])
_add("La Renaissance italienne a vu naître l'humanisme et des chefs-d'œuvre comme la Chapelle Sixtine.", [
    ("hint_event_named", "Renaissance"), ("hint_norp", "italienne"),
    ("hint_concept", "humanisme"), ("hint_work_of_art", "Chapelle Sixtine")])
_add("Le Coran est écrit en arabe classique et constitue le texte fondateur de l'islam.", [
    ("hint_work_of_art", "Coran"), ("hint_language", "arabe classique"), ("hint_concept", "islam")])
_add("La Convention de Berne protège les droits d'auteur des œuvres littéraires et artistiques.", [
    ("hint_law", "Convention de Berne")])
_add("Le choléra décrit dans Le Hussard sur le toit de Giono ravage la Provence.", [
    ("hint_disease", "choléra"), ("hint_work_of_art", "Hussard sur le toit"),
    ("hint_person_name", "Giono"), ("hint_gpe", "Provence")])
_add("Le décret Crémieux de 1870 accorda la citoyenneté française aux juifs d'Algérie.", [
    ("hint_law", "décret Crémieux"), ("hint_time_date", "1870"), ("hint_norp", "française"),
    ("hint_norp", "juifs"), ("hint_gpe", "Algérie")])
_add("Le bouddhisme et l'hindouisme ont profondément marqué les cultures asiatiques.", [
    ("hint_concept", "bouddhisme"), ("hint_concept", "hindouisme"), ("hint_norp", "asiatiques")])
_add("Nietzsche a proclamé la mort de Dieu dans Ainsi parlait Zarathoustra.", [
    ("hint_person_name", "Nietzsche"), ("hint_work_of_art", "Ainsi parlait Zarathoustra")])
_add("La Bible a été traduite en plus de 700 langues à travers le monde.", [
    ("hint_work_of_art", "Bible"), ("hint_count", "700")])
_add("Le Contrat social de Rousseau a inspiré les rédacteurs de la Déclaration des droits de l'homme.", [
    ("hint_work_of_art", "Contrat social"), ("hint_person_name", "Rousseau"),
    ("hint_law", "Déclaration des droits de l'homme")])
_add("Thomas More a inventé le concept d'utopie dans son ouvrage du même nom.", [
    ("hint_person_name", "Thomas More"), ("hint_concept", "utopie")])
_add("Le Capital de Marx est le texte fondateur du communisme moderne.", [
    ("hint_work_of_art", "Capital"), ("hint_person_name", "Marx"), ("hint_concept", "communisme")])
_add("La grippe de Hong Kong de 1968 a fait environ un million de morts dans le monde.", [
    ("hint_disease", "grippe de Hong Kong"), ("hint_time_date", "1968"), ("hint_quantity", "un million")])
_add("Les Principia de Newton ont posé les bases de la mécanique classique.", [
    ("hint_work_of_art", "Principia"), ("hint_person_name", "Newton"), ("hint_concept", "mécanique classique")])
_add("Le traité de Verdun de 843 a partagé l'Empire carolingien entre les petits-fils de Charlemagne.", [
    ("hint_law", "traité de Verdun"), ("hint_time_date", "843"), ("hint_person_name", "Charlemagne")])
_add("La Bhagavad-Gita est un texte sacré de l'hindouisme écrit en sanskrit.", [
    ("hint_work_of_art", "Bhagavad-Gita"), ("hint_concept", "hindouisme"), ("hint_language", "sanskrit")])
_add("Le cubisme de Picasso et Braque a rompu avec la perspective traditionnelle.", [
    ("hint_concept", "cubisme"), ("hint_person_name", "Picasso"), ("hint_person_name", "Braque")])
_add("La variole a été éradiquée en 1980 grâce au programme de l'OMS.", [
    ("hint_disease", "variole"), ("hint_time_date", "1980"), ("hint_org_name", "OMS")])
_add("Le Manifeste du Parti communiste de Marx et Engels a été publié en 1848.", [
    ("hint_work_of_art", "Manifeste du Parti communiste"), ("hint_person_name", "Marx"),
    ("hint_person_name", "Engels"), ("hint_time_date", "1848")])
_add("L'Encyclopédie de Diderot et d'Alembert incarne l'esprit des Lumières.", [
    ("hint_work_of_art", "Encyclopédie"), ("hint_person_name", "Diderot"), ("hint_person_name", "d'Alembert")])

# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genere les donnees ABSTRACT")
    parser.add_argument("--output", default="data/abstract_sentences.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    random.shuffle(SENTENCES)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, sent in enumerate(SENTENCES):
            row = {"id": f"abstract_{i:04d}", "text": sent["text"], "spans": sent["spans"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter
    label_counts = Counter()
    for sent in SENTENCES:
        for sp in sent["spans"]:
            label_counts[sp["label"]] += 1

    print(f"\n{len(SENTENCES)} phrases ecrites dans {args.output}")
    print(f"\nDistribution des labels :")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"   {label:30s} {count:4d}")

if __name__ == "__main__":
    main()

