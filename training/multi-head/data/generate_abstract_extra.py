#!/usr/bin/env python3
"""
Génération massive de phrases supplémentaires pour les labels ABSTRACT.
Complète generate_abstract_data.py pour atteindre ~300+ par fine label.
"""
from __future__ import annotations
import json, argparse, random
from typing import List, Dict, Tuple

SENTENCES: List[Dict] = []

def _add(text: str, annotations: List[Tuple[str, str]]):
    spans = []
    used_positions = set()
    for label, surface in annotations:
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
#  hint_law  (~60 phrases supplémentaires)
# ═══════════════════════════════════════════════════════════════

_add("La loi Avia sur la haine en ligne a été partiellement censurée par le Conseil constitutionnel.", [
    ("hint_law", "loi Avia"), ("hint_org_name", "Conseil constitutionnel")])
_add("Le traité de Brest-Litovsk de 1918 mit fin à la participation russe à la Première Guerre mondiale.", [
    ("hint_law", "traité de Brest-Litovsk"), ("hint_time_date", "1918"), ("hint_norp", "russe"), ("hint_event_named", "Première Guerre mondiale")])
_add("L'acte unique européen de 1986 a relancé la construction européenne.", [
    ("hint_law", "acte unique européen"), ("hint_time_date", "1986")])
_add("La loi Littoral de 1986 protège les côtes françaises de l'urbanisation.", [
    ("hint_law", "loi Littoral"), ("hint_time_date", "1986"), ("hint_norp", "françaises")])
_add("Le traité de Tilsit a scellé l'alliance entre Napoléon et le tsar Alexandre.", [
    ("hint_law", "traité de Tilsit"), ("hint_person_name", "Napoléon"), ("hint_person_name", "Alexandre")])
_add("Les accords d'Abraham ont normalisé les relations entre Israël et plusieurs pays arabes.", [
    ("hint_law", "accords d'Abraham"), ("hint_gpe", "Israël"), ("hint_norp", "arabes")])
_add("La loi SRU impose aux communes un quota de logements sociaux.", [("hint_law", "loi SRU")])
_add("Le traité de San Stefano de 1878 accorda l'indépendance à la Bulgarie.", [
    ("hint_law", "traité de San Stefano"), ("hint_time_date", "1878"), ("hint_gpe", "Bulgarie")])
_add("La directive REACH encadre l'utilisation des substances chimiques en Europe.", [
    ("hint_law", "directive REACH"), ("hint_gpe", "Europe")])
_add("Le Code pénal français a été profondément réformé en 1994.", [
    ("hint_law", "Code pénal"), ("hint_norp", "français"), ("hint_time_date", "1994")])
_add("Le décret de Messidor fixait le calendrier républicain en France.", [
    ("hint_law", "décret de Messidor"), ("hint_gpe", "France")])
_add("La Convention de Vienne régit le droit des traités internationaux depuis 1969.", [
    ("hint_law", "Convention de Vienne"), ("hint_time_date", "1969")])
_add("L'armistice du 11 novembre 1918 mit fin aux combats de la Grande Guerre.", [
    ("hint_law", "armistice du 11 novembre 1918"), ("hint_event_named", "Grande Guerre")])
_add("Le traité de Nankin de 1842 imposa l'ouverture de cinq ports chinois au commerce britannique.", [
    ("hint_law", "traité de Nankin"), ("hint_time_date", "1842"), ("hint_norp", "chinois"), ("hint_norp", "britannique")])
_add("La loi Blanquer a réformé l'organisation de l'éducation nationale en France.", [
    ("hint_law", "loi Blanquer"), ("hint_gpe", "France")])
_add("Le concordat d'Amboise régla les rapports entre la monarchie et la papauté.", [
    ("hint_law", "concordat d'Amboise")])
_add("Les conventions de La Haye codifient les lois et coutumes de la guerre.", [
    ("hint_law", "conventions de La Haye"), ("hint_event_nominal", "guerre")])
_add("Le traité de Rapallo de 1922 rapprocha l'Allemagne et la Russie soviétique.", [
    ("hint_law", "traité de Rapallo"), ("hint_time_date", "1922"), ("hint_gpe", "Allemagne")])
_add("La loi Pacte de 2019 a simplifié la création d'entreprise en France.", [
    ("hint_law", "loi Pacte"), ("hint_time_date", "2019"), ("hint_gpe", "France")])
_add("Le traité de Sèvres de 1920 prévoyait le démembrement de l'Empire ottoman.", [
    ("hint_law", "traité de Sèvres"), ("hint_time_date", "1920")])
_add("L'édit de Fontainebleau de 1685 révoqua l'édit de Nantes et provoqua l'exil des huguenots.", [
    ("hint_law", "édit de Fontainebleau"), ("hint_time_date", "1685"), ("hint_law", "édit de Nantes"), ("hint_norp", "huguenots")])
_add("La loi Sapin 2 renforce la transparence et la lutte contre la corruption.", [
    ("hint_law", "loi Sapin 2")])
_add("Le traité de Lausanne de 1923 fixa les frontières de la Turquie moderne.", [
    ("hint_law", "traité de Lausanne"), ("hint_time_date", "1923"), ("hint_gpe", "Turquie")])
_add("Le pacte germano-soviétique de 1939 stupéfia les chancelleries européennes.", [
    ("hint_law", "pacte germano-soviétique"), ("hint_time_date", "1939"), ("hint_norp", "européennes")])
_add("La charte de l'environnement a été intégrée à la Constitution française en 2005.", [
    ("hint_law", "charte de l'environnement"), ("hint_time_date", "2005")])
_add("Le traité de Trianon de 1920 réduisit considérablement le territoire hongrois.", [
    ("hint_law", "traité de Trianon"), ("hint_time_date", "1920"), ("hint_norp", "hongrois")])
_add("Les accords de Minsk visaient à mettre fin au conflit dans l'est de l'Ukraine.", [
    ("hint_law", "accords de Minsk"), ("hint_gpe", "Ukraine")])
_add("La loi Dalo garantit le droit au logement opposable en France.", [
    ("hint_law", "loi Dalo"), ("hint_gpe", "France")])
_add("Le traité de Lisbonne a doté l'Union européenne d'une personnalité juridique.", [
    ("hint_law", "traité de Lisbonne"), ("hint_org_name", "Union européenne")])
_add("L'ordonnance de Moulins de 1566 renforça le pouvoir judiciaire royal.", [
    ("hint_law", "ordonnance de Moulins"), ("hint_time_date", "1566")])
_add("Le traité d'Aix-la-Chapelle de 2019 renforce la coopération franco-allemande.", [
    ("hint_law", "traité d'Aix-la-Chapelle"), ("hint_time_date", "2019")])
_add("La loi Climat et Résilience de 2021 fixe des objectifs ambitieux de réduction des émissions.", [
    ("hint_law", "loi Climat et Résilience"), ("hint_time_date", "2021")])
_add("Le pacte de stabilité et de croissance encadre les budgets des pays de la zone euro.", [
    ("hint_law", "pacte de stabilité et de croissance")])
_add("Les accords de Dayton de 1995 mirent fin à la guerre en Bosnie.", [
    ("hint_law", "accords de Dayton"), ("hint_time_date", "1995"), ("hint_event_nominal", "guerre"), ("hint_gpe", "Bosnie")])
_add("La Convention d'Istanbul lutte contre la violence faite aux femmes.", [
    ("hint_law", "Convention d'Istanbul")])
_add("Le traité de Nymphenburg de 1741 allia la Bavière à la France contre l'Autriche.", [
    ("hint_law", "traité de Nymphenburg"), ("hint_time_date", "1741"), ("hint_gpe", "Bavière"), ("hint_gpe", "France"), ("hint_gpe", "Autriche")])
_add("La loi de programmation militaire définit les grands axes de la défense nationale.", [
    ("hint_law", "loi de programmation militaire")])
_add("Le décret-loi de 1939 interdit les organisations communistes en France.", [
    ("hint_law", "décret-loi de 1939"), ("hint_norp", "communistes"), ("hint_gpe", "France")])
_add("Le traité de Cambrai de 1529 est aussi appelé la Paix des Dames.", [
    ("hint_law", "traité de Cambrai"), ("hint_time_date", "1529")])
_add("La directive Natura 2000 protège les habitats naturels de l'Union européenne.", [
    ("hint_law", "directive Natura 2000"), ("hint_org_name", "Union européenne")])
_add("La loi Leonetti-Claeys de 2016 encadre la fin de vie en France.", [
    ("hint_law", "loi Leonetti-Claeys"), ("hint_time_date", "2016"), ("hint_gpe", "France")])
_add("Le traité de paix de Hubertusburg de 1763 mit fin à la guerre de Sept Ans.", [
    ("hint_law", "traité de paix de Hubertusburg"), ("hint_time_date", "1763")])
_add("La loi NOME de 2010 a libéralisé le marché de l'électricité en France.", [
    ("hint_law", "loi NOME"), ("hint_time_date", "2010"), ("hint_gpe", "France")])
_add("L'édit de Caracalla de 212 accorda la citoyenneté romaine à tous les hommes libres de l'Empire.", [
    ("hint_law", "édit de Caracalla"), ("hint_time_date", "212"), ("hint_norp", "romaine")])

# ═══════════════════════════════════════════════════════════════
#  hint_work_of_art  (~80 phrases supplémentaires)
# ═══════════════════════════════════════════════════════════════

_add("Les Misérables ont été adaptés en comédie musicale à Londres.", [
    ("hint_work_of_art", "Misérables"), ("hint_gpe", "Londres")])
_add("Le Déjeuner sur l'herbe de Manet fit scandale au Salon des refusés en 1863.", [
    ("hint_work_of_art", "Déjeuner sur l'herbe"), ("hint_person_name", "Manet"), ("hint_time_date", "1863")])
_add("War and Peace de Tolstoï est une fresque monumentale de la Russie napoléonienne.", [
    ("hint_work_of_art", "War and Peace"), ("hint_person_name", "Tolstoï")])
_add("Le Barbier de Séville de Rossini est un opéra bouffe en deux actes.", [
    ("hint_work_of_art", "Barbier de Séville"), ("hint_person_name", "Rossini")])
_add("Germinal de Zola dépeint les conditions de vie des mineurs du Nord.", [
    ("hint_work_of_art", "Germinal"), ("hint_person_name", "Zola")])
_add("Le Joueur d'échecs de Stefan Zweig est une nouvelle sur la folie obsessionnelle.", [
    ("hint_work_of_art", "Joueur d'échecs"), ("hint_person_name", "Stefan Zweig")])
_add("La Naissance de Vénus de Botticelli est un chef-d'œuvre de la Renaissance florentine.", [
    ("hint_work_of_art", "Naissance de Vénus"), ("hint_person_name", "Botticelli"), ("hint_event_named", "Renaissance")])
_add("Le Chien andalou de Buñuel et Dalí est un court-métrage surréaliste de 1929.", [
    ("hint_work_of_art", "Chien andalou"), ("hint_person_name", "Buñuel"), ("hint_person_name", "Dalí"), ("hint_time_date", "1929")])
_add("Les Confessions de Rousseau sont une autobiographie qui a marqué la littérature.", [
    ("hint_work_of_art", "Confessions"), ("hint_person_name", "Rousseau")])
_add("Le Avengers est l'une des franchises cinématographiques les plus lucratives.", [
    ("hint_work_of_art", "Avengers")])
_add("Les Fables de La Fontaine sont enseignées dans toutes les écoles françaises.", [
    ("hint_work_of_art", "Fables"), ("hint_person_name", "La Fontaine"), ("hint_norp", "françaises")])
_add("Le Joueur de flûte de Hamelin est un conte populaire d'origine allemande.", [
    ("hint_work_of_art", "Joueur de flûte de Hamelin"), ("hint_norp", "allemande")])
_add("Shining de Stanley Kubrick est considéré comme un chef-d'œuvre du film d'horreur.", [
    ("hint_work_of_art", "Shining"), ("hint_person_name", "Stanley Kubrick")])
_add("Bonjour tristesse de Françoise Sagan a été publié quand l'autrice avait 18 ans.", [
    ("hint_work_of_art", "Bonjour tristesse"), ("hint_person_name", "Françoise Sagan")])
_add("Le Cantique des cantiques est l'un des textes les plus poétiques de la Bible.", [
    ("hint_work_of_art", "Cantique des cantiques"), ("hint_work_of_art", "Bible")])
_add("Psycho d'Alfred Hitchcock a révolutionné le thriller psychologique.", [
    ("hint_work_of_art", "Psycho"), ("hint_person_name", "Alfred Hitchcock")])
_add("Le Tour du monde en quatre-vingts jours de Jules Verne reste un classique de l'aventure.", [
    ("hint_work_of_art", "Tour du monde en quatre-vingts jours"), ("hint_person_name", "Jules Verne")])
_add("Nosferatu de Murnau est le premier film de vampires de l'histoire du cinéma.", [
    ("hint_work_of_art", "Nosferatu"), ("hint_person_name", "Murnau")])
_add("Le Prince de Machiavel est un traité politique écrit au XVIe siècle.", [
    ("hint_work_of_art", "Prince"), ("hint_person_name", "Machiavel"), ("hint_time_date", "XVIe siècle")])
_add("La Mouette de Tchekhov est une pièce de théâtre en quatre actes.", [
    ("hint_work_of_art", "Mouette"), ("hint_person_name", "Tchekhov")])
_add("Breaking Bad est souvent citée comme la meilleure série télévisée de l'histoire.", [
    ("hint_work_of_art", "Breaking Bad")])
_add("Les Frères Karamazov de Dostoïevski explorent le conflit entre foi et raison.", [
    ("hint_work_of_art", "Frères Karamazov"), ("hint_person_name", "Dostoïevski")])
_add("Le Corbeau et le Renard est la fable la plus connue de La Fontaine.", [
    ("hint_work_of_art", "Corbeau et le Renard"), ("hint_person_name", "La Fontaine")])
_add("Parasite de Bong Joon-ho a remporté la Palme d'or à Cannes en 2019.", [
    ("hint_work_of_art", "Parasite"), ("hint_person_name", "Bong Joon-ho"), ("hint_gpe", "Cannes"), ("hint_time_date", "2019")])
_add("Le Livre de la jungle de Rudyard Kipling a été adapté par les studios Disney.", [
    ("hint_work_of_art", "Livre de la jungle"), ("hint_person_name", "Rudyard Kipling"), ("hint_org_name", "Disney")])
_add("Le Monde de Sophie de Jostein Gaarder est un roman d'initiation à la philosophie.", [
    ("hint_work_of_art", "Monde de Sophie"), ("hint_person_name", "Jostein Gaarder")])
_add("Psychose de Hitchcock contient la scène de douche la plus célèbre du cinéma.", [
    ("hint_work_of_art", "Psychose"), ("hint_person_name", "Hitchcock")])
_add("L'Art de la guerre de Sun Tzu est un traité militaire chinois du Ve siècle av. J.-C.", [
    ("hint_work_of_art", "Art de la guerre"), ("hint_person_name", "Sun Tzu"), ("hint_norp", "chinois")])
_add("La Cantatrice chauve d'Ionesco est une pièce fondatrice du théâtre de l'absurde.", [
    ("hint_work_of_art", "Cantatrice chauve"), ("hint_person_name", "Ionesco")])
_add("Le Mépris de Jean-Luc Godard est tourné dans la villa Malaparte à Capri.", [
    ("hint_work_of_art", "Mépris"), ("hint_person_name", "Jean-Luc Godard"), ("hint_gpe", "Capri")])
_add("Beloved de Toni Morrison a remporté le prix Pulitzer en 1988.", [
    ("hint_work_of_art", "Beloved"), ("hint_person_name", "Toni Morrison"), ("hint_time_date", "1988")])
_add("L'Oiseau de feu de Stravinsky est un ballet créé pour les Ballets russes.", [
    ("hint_work_of_art", "Oiseau de feu"), ("hint_person_name", "Stravinsky")])
_add("Ulysse de James Joyce est considéré comme le sommet du modernisme littéraire.", [
    ("hint_work_of_art", "Ulysse"), ("hint_person_name", "James Joyce")])
_add("Les Voyages de Gulliver de Jonathan Swift est une satire de la société anglaise.", [
    ("hint_work_of_art", "Voyages de Gulliver"), ("hint_person_name", "Jonathan Swift"), ("hint_norp", "anglaise")])
_add("Le Parrain 2 de Coppola est souvent considéré comme supérieur au premier volet.", [
    ("hint_work_of_art", "Parrain 2"), ("hint_person_name", "Coppola")])
_add("L'Alchimiste de Paulo Coelho a été traduit en plus de 80 langues.", [
    ("hint_work_of_art", "Alchimiste"), ("hint_person_name", "Paulo Coelho")])
_add("La Symphonie du Nouveau Monde de Dvořák s'inspire de la musique afro-américaine.", [
    ("hint_work_of_art", "Symphonie du Nouveau Monde"), ("hint_person_name", "Dvořák")])
_add("Le Dernier des Mohicans de Fenimore Cooper est un classique du roman américain.", [
    ("hint_work_of_art", "Dernier des Mohicans"), ("hint_person_name", "Fenimore Cooper"), ("hint_norp", "américain")])
_add("L'Homme qui rit de Victor Hugo a inspiré le personnage du Joker.", [
    ("hint_work_of_art", "Homme qui rit"), ("hint_person_name", "Victor Hugo")])
_add("Notre-Dame de Paris de Victor Hugo a contribué à sauver la cathédrale de la démolition.", [
    ("hint_work_of_art", "Notre-Dame de Paris"), ("hint_person_name", "Victor Hugo")])
_add("Le Roi Lear de Shakespeare met en scène la folie d'un monarque vieillissant.", [
    ("hint_work_of_art", "Roi Lear"), ("hint_person_name", "Shakespeare")])
_add("Interstellar de Christopher Nolan explore les paradoxes du voyage dans le temps.", [
    ("hint_work_of_art", "Interstellar"), ("hint_person_name", "Christopher Nolan")])
_add("La Ferme des animaux de George Orwell est une allégorie de la révolution russe.", [
    ("hint_work_of_art", "Ferme des animaux"), ("hint_person_name", "George Orwell"), ("hint_norp", "russe")])
_add("La Dolce Vita de Fellini a donné son nom à un style de vie insouciant.", [
    ("hint_work_of_art", "Dolce Vita"), ("hint_person_name", "Fellini")])
_add("Le Fantôme de Canterville d'Oscar Wilde mêle humour et fantastique.", [
    ("hint_work_of_art", "Fantôme de Canterville"), ("hint_person_name", "Oscar Wilde")])
_add("Frankenstein de Mary Shelley est souvent considéré comme le premier roman de science-fiction.", [
    ("hint_work_of_art", "Frankenstein"), ("hint_person_name", "Mary Shelley")])
_add("Le Livre des merveilles de Marco Polo a fasciné les Européens du Moyen Âge.", [
    ("hint_work_of_art", "Livre des merveilles"), ("hint_person_name", "Marco Polo"), ("hint_norp", "Européens")])
_add("Vertigo d'Alfred Hitchcock est un chef-d'œuvre du suspense psychologique.", [
    ("hint_work_of_art", "Vertigo"), ("hint_person_name", "Alfred Hitchcock")])
_add("Le Songe d'une nuit d'été de Shakespeare est une comédie féerique.", [
    ("hint_work_of_art", "Songe d'une nuit d'été"), ("hint_person_name", "Shakespeare")])
_add("One Piece d'Eiichiro Oda est le manga le plus vendu de tous les temps.", [
    ("hint_work_of_art", "One Piece"), ("hint_person_name", "Eiichiro Oda")])
_add("Naruto de Masashi Kishimoto a popularisé la culture du manga en Occident.", [
    ("hint_work_of_art", "Naruto"), ("hint_person_name", "Masashi Kishimoto")])
_add("Le Souper d'Emmaüs du Caravage joue magistralement avec la lumière.", [
    ("hint_work_of_art", "Souper d'Emmaüs"), ("hint_person_name", "Caravage")])
_add("La Cinquième Symphonie de Beethoven débute par le motif le plus célèbre de la musique.", [
    ("hint_work_of_art", "Cinquième Symphonie"), ("hint_person_name", "Beethoven")])
_add("Dune de Frank Herbert est un pilier de la science-fiction littéraire.", [
    ("hint_work_of_art", "Dune"), ("hint_person_name", "Frank Herbert")])
_add("Le Loup des steppes de Hermann Hesse explore la dualité de la nature humaine.", [
    ("hint_work_of_art", "Loup des steppes"), ("hint_person_name", "Hermann Hesse")])
_add("Les Aventures de Pinocchio de Collodi est un classique de la littérature italienne.", [
    ("hint_work_of_art", "Aventures de Pinocchio"), ("hint_person_name", "Collodi"), ("hint_norp", "italienne")])
_add("Le Médecin malgré lui de Molière est une comédie satirique sur les charlatans.", [
    ("hint_work_of_art", "Médecin malgré lui"), ("hint_person_name", "Molière")])
_add("Amélie Poulain de Jean-Pierre Jeunet a charmé les spectateurs du monde entier.", [
    ("hint_work_of_art", "Amélie Poulain"), ("hint_person_name", "Jean-Pierre Jeunet")])
_add("La Liste de Schindler de Spielberg retrace le sauvetage de 1200 juifs pendant la Shoah.", [
    ("hint_work_of_art", "Liste de Schindler"), ("hint_person_name", "Spielberg"), ("hint_norp", "juifs")])
_add("Vol au-dessus d'un nid de coucou a remporté les cinq Oscars majeurs en 1976.", [
    ("hint_work_of_art", "Vol au-dessus d'un nid de coucou"), ("hint_time_date", "1976")])
_add("Le Horla de Maupassant est une nouvelle fantastique sur la folie.", [
    ("hint_work_of_art", "Horla"), ("hint_person_name", "Maupassant")])
_add("Bel-Ami de Maupassant dépeint l'ascension sociale d'un arriviste dans le Paris du XIXe.", [
    ("hint_work_of_art", "Bel-Ami"), ("hint_person_name", "Maupassant"), ("hint_gpe", "Paris")])
_add("Titanic de James Cameron est le film qui a fait pleurer des millions de spectateurs.", [
    ("hint_work_of_art", "Titanic"), ("hint_person_name", "James Cameron")])
_add("1984 de George Orwell décrit une société totalitaire où la pensée est contrôlée.", [
    ("hint_work_of_art", "1984"), ("hint_person_name", "George Orwell")])
_add("Le Conte de fées de Perrault a popularisé Cendrillon et La Belle au bois dormant.", [
    ("hint_person_name", "Perrault"), ("hint_work_of_art", "Cendrillon"), ("hint_work_of_art", "Belle au bois dormant")])

# ═══════════════════════════════════════════════════════════════
#  hint_concept  (~80 phrases supplémentaires)
# ═══════════════════════════════════════════════════════════════

_add("Le matérialisme historique est la base de l'analyse sociale marxiste.", [
    ("hint_concept", "matérialisme historique"), ("hint_norp", "marxiste")])
_add("La philosophie des Lumières a inspiré les révolutions du XVIIIe siècle.", [
    ("hint_concept", "philosophie des Lumières"), ("hint_time_date", "XVIIIe siècle")])
_add("Le relativisme moral soutient que les jugements éthiques dépendent du contexte culturel.", [
    ("hint_concept", "relativisme moral")])
_add("Le décolonialisme remet en question les héritages intellectuels de la colonisation.", [
    ("hint_concept", "décolonialisme")])
_add("Le néoplatonisme a influencé la pensée chrétienne et islamique médiévale.", [
    ("hint_concept", "néoplatonisme"), ("hint_norp", "chrétienne"), ("hint_norp", "islamique")])
_add("Le libéralisme politique défend les libertés individuelles et l'État de droit.", [
    ("hint_concept", "libéralisme politique")])
_add("Le socialisme utopique de Fourier proposait une organisation communautaire de la société.", [
    ("hint_concept", "socialisme utopique"), ("hint_person_name", "Fourier")])
_add("L'individualisme met l'accent sur l'autonomie et la responsabilité personnelle.", [
    ("hint_concept", "individualisme")])
_add("Le holisme considère qu'un système est plus que la somme de ses parties.", [
    ("hint_concept", "holisme")])
_add("Le fonctionnalisme en sociologie analyse les institutions par leur rôle social.", [
    ("hint_concept", "fonctionnalisme")])
_add("L'animisme attribue une âme aux éléments de la nature.", [("hint_concept", "animisme")])
_add("Le pantheisme identifie Dieu à l'ensemble de la nature et de l'univers.", [
    ("hint_concept", "pantheisme")])
_add("Le dualisme cartésien distingue le corps et l'esprit comme deux substances séparées.", [
    ("hint_concept", "dualisme cartésien")])
_add("Le solipsisme doute de l'existence de tout ce qui est extérieur à l'esprit.", [
    ("hint_concept", "solipsisme")])
_add("La théorie des jeux analyse les interactions stratégiques entre agents rationnels.", [
    ("hint_concept", "théorie des jeux")])
_add("Le malthusianisme prédit que la croissance démographique dépassera les ressources.", [
    ("hint_concept", "malthusianisme")])
_add("Le souverainisme défend la primauté de l'État-nation sur les instances supranationales.", [
    ("hint_concept", "souverainisme")])
_add("Le néoconservatisme a influencé la politique étrangère américaine après 2001.", [
    ("hint_concept", "néoconservatisme"), ("hint_norp", "américaine"), ("hint_time_date", "2001")])
_add("Le véganisme rejette toute forme d'exploitation animale.", [
    ("hint_concept", "véganisme")])
_add("L'anticléricalisme s'oppose à l'influence du clergé dans la vie publique.", [
    ("hint_concept", "anticléricalisme")])
_add("Le cosmopolitisme considère que chaque être humain est citoyen du monde.", [
    ("hint_concept", "cosmopolitisme")])
_add("L'utilitarisme de Mill raffine celui de Bentham en distinguant les plaisirs.", [
    ("hint_concept", "utilitarisme"), ("hint_person_name", "Mill"), ("hint_person_name", "Bentham")])
_add("Le perspectivisme de Nietzsche affirme qu'il n'existe pas de vérité absolue.", [
    ("hint_concept", "perspectivisme"), ("hint_person_name", "Nietzsche")])
_add("L'herméneutique cherche à interpréter les textes en tenant compte de leur contexte.", [
    ("hint_concept", "herméneutique")])
_add("Le matérialisme philosophique réduit la réalité à la matière et à ses mouvements.", [
    ("hint_concept", "matérialisme philosophique")])
_add("Le pluralisme politique accepte la coexistence de visions du monde divergentes.", [
    ("hint_concept", "pluralisme politique")])
_add("L'empirisme logique du Cercle de Vienne a profondément marqué la philosophie analytique.", [
    ("hint_concept", "empirisme logique"), ("hint_org_name", "Cercle de Vienne")])
_add("Le situationnisme de Debord critique la société du spectacle et de la consommation.", [
    ("hint_concept", "situationnisme"), ("hint_person_name", "Debord")])
_add("Le jacobinisme défend un État centralisé et une souveraineté populaire forte.", [
    ("hint_concept", "jacobinisme")])
_add("Le maoïsme a adapté le marxisme-léninisme au contexte paysan chinois.", [
    ("hint_concept", "maoïsme"), ("hint_concept", "marxisme-léninisme"), ("hint_norp", "chinois")])
_add("Le post-modernisme remet en cause les grands récits de la modernité.", [
    ("hint_concept", "post-modernisme")])
_add("L'écoféminisme relie la domination de la nature à l'oppression des femmes.", [
    ("hint_concept", "écoféminisme")])
_add("Le keynésianisme recommande la relance budgétaire en période de récession.", [
    ("hint_concept", "keynésianisme"), ("hint_event_nominal", "récession")])
_add("Le taylorisme organise le travail par la division scientifique des tâches.", [
    ("hint_concept", "taylorisme")])
_add("Le fordisme a généralisé la production de masse et la consommation de masse.", [
    ("hint_concept", "fordisme")])
_add("Le toyotisme privilégie la production en flux tendu et la qualité totale.", [
    ("hint_concept", "toyotisme")])
_add("Le luddisme désigne la résistance des ouvriers anglais à la mécanisation.", [
    ("hint_concept", "luddisme"), ("hint_norp", "anglais")])
_add("Le malthusianisme économique freine volontairement la production pour maintenir les prix.", [
    ("hint_concept", "malthusianisme économique")])
_add("La physiocratie considérait l'agriculture comme la seule source de richesse.", [
    ("hint_concept", "physiocratie")])
_add("Le néo-institutionnalisme analyse le rôle des institutions dans les dynamiques économiques.", [
    ("hint_concept", "néo-institutionnalisme")])
_add("L'intersectionnalité étudie l'imbrication des différentes formes de discrimination.", [
    ("hint_concept", "intersectionnalité")])
_add("Le structuralisme de Lévi-Strauss a révolutionné l'anthropologie.", [
    ("hint_concept", "structuralisme"), ("hint_person_name", "Lévi-Strauss")])
_add("La psychologie cognitive étudie les processus mentaux comme la mémoire et l'attention.", [
    ("hint_concept", "psychologie cognitive")])
_add("Le connexionnisme modélise la cognition par des réseaux de neurones artificiels.", [
    ("hint_concept", "connexionnisme")])
_add("L'éthique conséquentialiste juge les actes uniquement par leurs résultats.", [
    ("hint_concept", "éthique conséquentialiste")])
_add("La déontologie de Kant fonde la morale sur le devoir et la règle universelle.", [
    ("hint_concept", "déontologie"), ("hint_person_name", "Kant")])
_add("Le contractualisme de Rawls propose une justice fondée sur le voile d'ignorance.", [
    ("hint_concept", "contractualisme"), ("hint_person_name", "Rawls")])
_add("L'agilité en développement logiciel privilégie l'itération rapide et le feedback.", [
    ("hint_concept", "agilité")])
_add("Le paradigme orienté objet structure le code autour de classes et d'instances.", [
    ("hint_concept", "paradigme orienté objet")])
_add("La neutralité du net garantit l'égalité de traitement des flux de données.", [
    ("hint_concept", "neutralité du net")])
_add("Le droit naturel postule l'existence de droits universels antérieurs à toute loi.", [
    ("hint_concept", "droit naturel")])

# ═══════════════════════════════════════════════════════════════
#  hint_disease  (~40 phrases supplémentaires)
# ═══════════════════════════════════════════════════════════════

_add("La maladie à virus Marburg est une fièvre hémorragique proche d'Ebola.", [
    ("hint_disease", "maladie à virus Marburg")])
_add("Le MERS-CoV a touché principalement le Moyen-Orient à partir de 2012.", [
    ("hint_disease", "MERS-CoV"), ("hint_gpe", "Moyen-Orient"), ("hint_time_date", "2012")])
_add("La leishmaniose est transmise par la piqûre de phlébotomes infectés.", [
    ("hint_disease", "leishmaniose")])
_add("La chikungunya provoque de fortes douleurs articulaires chez les personnes infectées.", [
    ("hint_disease", "chikungunya")])
_add("Le glaucome est une maladie de l'œil qui peut conduire à la cécité.", [
    ("hint_disease", "glaucome")])
_add("La maladie de Huntington est une affection neurodégénérative héréditaire.", [
    ("hint_disease", "maladie de Huntington")])
_add("La drépanocytose est la maladie génétique la plus répandue dans le monde.", [
    ("hint_disease", "drépanocytose")])
_add("L'autisme est un trouble du neurodéveloppement qui affecte la communication.", [
    ("hint_disease", "autisme")])
_add("La maladie cœliaque impose un régime strict sans gluten.", [
    ("hint_disease", "maladie cœliaque")])
_add("Le syndrome de Marfan est une maladie génétique du tissu conjonctif.", [
    ("hint_disease", "syndrome de Marfan")])
_add("L'ostéoporose fragilise les os et augmente le risque de fractures.", [
    ("hint_disease", "ostéoporose")])
_add("La schizophrénie touche environ 1 % de la population mondiale.", [
    ("hint_disease", "schizophrénie"), ("hint_percentage", "1 %")])
_add("La maladie de Chagas est endémique en Amérique latine.", [
    ("hint_disease", "maladie de Chagas"), ("hint_gpe", "Amérique latine")])
_add("Le psoriasis est une maladie inflammatoire chronique de la peau.", [
    ("hint_disease", "psoriasis")])
_add("La maladie de Charcot provoque une dégénérescence progressive des motoneurones.", [
    ("hint_disease", "maladie de Charcot")])
_add("Le syndrome de Turner ne touche que les personnes de sexe féminin.", [
    ("hint_disease", "syndrome de Turner")])
_add("La myopathie de Duchenne est une dystrophie musculaire progressive et invalidante.", [
    ("hint_disease", "myopathie de Duchenne")])
_add("L'anorexie mentale est un trouble alimentaire pouvant mettre la vie en danger.", [
    ("hint_disease", "anorexie mentale")])
_add("La maladie de Ménière provoque des vertiges, des acouphènes et une perte auditive.", [
    ("hint_disease", "maladie de Ménière")])
_add("La narcolepsie est un trouble du sommeil caractérisé par des accès de somnolence.", [
    ("hint_disease", "narcolepsie")])
_add("L'hémophilie empêche la coagulation normale du sang.", [
    ("hint_disease", "hémophilie")])
_add("Le paludisme tue encore plus de 600 000 personnes par an, surtout en Afrique.", [
    ("hint_disease", "paludisme"), ("hint_quantity", "600 000"), ("hint_gpe", "Afrique")])
_add("La poliomyélite a presque disparu grâce aux campagnes de vaccination de l'OMS.", [
    ("hint_disease", "poliomyélite"), ("hint_org_name", "OMS")])
_add("Le zona est une réactivation du virus de la varicelle chez l'adulte.", [
    ("hint_disease", "zona"), ("hint_disease", "varicelle")])
_add("La maladie de Wilson provoque une accumulation toxique de cuivre dans l'organisme.", [
    ("hint_disease", "maladie de Wilson")])
_add("Le syndrome de Cushing résulte d'un excès chronique de cortisol.", [
    ("hint_disease", "syndrome de Cushing")])
_add("La grippe saisonnière cause entre 290 000 et 650 000 décès par an dans le monde.", [
    ("hint_disease", "grippe saisonnière")])
_add("Le mpox, anciennement appelé variole du singe, a connu une flambée mondiale en 2022.", [
    ("hint_disease", "mpox"), ("hint_time_date", "2022")])
_add("La maladie de Gaucher est une maladie lysosomale traitable par enzymothérapie.", [
    ("hint_disease", "maladie de Gaucher")])
_add("Le syndrome de fatigue chronique reste une maladie mal comprise.", [
    ("hint_disease", "syndrome de fatigue chronique")])

# ═══════════════════════════════════════════════════════════════
#  hint_language  (~40 phrases supplémentaires)
# ═══════════════════════════════════════════════════════════════

_add("Le vietnamien utilise un alphabet latin enrichi de diacritiques.", [
    ("hint_language", "vietnamien"), ("hint_language", "latin")])
_add("Le khmer est la langue officielle du Cambodge.", [
    ("hint_language", "khmer"), ("hint_gpe", "Cambodge")])
_add("L'amharique est la langue de travail du gouvernement éthiopien.", [
    ("hint_language", "amharique"), ("hint_norp", "éthiopien")])
_add("Le berbère est reconnu comme langue officielle en Algérie et au Maroc.", [
    ("hint_language", "berbère"), ("hint_gpe", "Algérie"), ("hint_gpe", "Maroc")])
_add("Le somali est parlé en Somalie, à Djibouti et dans l'est de l'Éthiopie.", [
    ("hint_language", "somali"), ("hint_gpe", "Somalie"), ("hint_gpe", "Djibouti")])
_add("Le lituanien est considéré comme l'une des langues indo-européennes les plus archaïques.", [
    ("hint_language", "lituanien")])
_add("L'afrikaans est une langue germanique parlée en Afrique du Sud.", [
    ("hint_language", "afrikaans"), ("hint_gpe", "Afrique du Sud")])
_add("Le birman est la langue officielle du Myanmar.", [
    ("hint_language", "birman"), ("hint_gpe", "Myanmar")])
_add("Le mongol est écrit en écriture cyrillique en Mongolie.", [
    ("hint_language", "mongol"), ("hint_gpe", "Mongolie")])
_add("Le pachtou est l'une des deux langues officielles de l'Afghanistan.", [
    ("hint_language", "pachtou"), ("hint_gpe", "Afghanistan")])
_add("Le serbe peut s'écrire en alphabet latin ou cyrillique.", [
    ("hint_language", "serbe"), ("hint_language", "latin")])
_add("Le haoussa est une langue de commerce majeure en Afrique de l'Ouest.", [
    ("hint_language", "haoussa"), ("hint_gpe", "Afrique de l'Ouest")])
_add("Le zoulou est la langue maternelle la plus répandue en Afrique du Sud.", [
    ("hint_language", "zoulou"), ("hint_gpe", "Afrique du Sud")])
_add("Le nahuatl était la langue de l'Empire aztèque au Mexique.", [
    ("hint_language", "nahuatl"), ("hint_gpe", "Mexique")])
_add("L'ourdou est la langue nationale du Pakistan, proche du hindi.", [
    ("hint_language", "ourdou"), ("hint_gpe", "Pakistan"), ("hint_language", "hindi")])
_add("Le croate, le bosnien et le serbe sont mutuellement compréhensibles.", [
    ("hint_language", "croate"), ("hint_language", "bosnien"), ("hint_language", "serbe")])
_add("Le géorgien utilise un alphabet unique vieux de plus de 1500 ans.", [
    ("hint_language", "géorgien")])
_add("Le romani est la langue des communautés roms dispersées en Europe.", [
    ("hint_language", "romani"), ("hint_norp", "roms"), ("hint_gpe", "Europe")])
_add("L'aymara est une langue amérindienne parlée en Bolivie et au Pérou.", [
    ("hint_language", "aymara"), ("hint_gpe", "Bolivie"), ("hint_gpe", "Pérou")])
_add("Le slovène est la langue officielle de la Slovénie depuis son indépendance.", [
    ("hint_language", "slovène"), ("hint_gpe", "Slovénie")])
_add("Le kurde est parlé par environ 30 millions de locuteurs répartis entre plusieurs pays.", [
    ("hint_language", "kurde"), ("hint_quantity", "30 millions")])
_add("Le provençal est un dialecte de l'occitan parlé dans le sud-est de la France.", [
    ("hint_language", "provençal"), ("hint_language", "occitan"), ("hint_gpe", "France")])
_add("Le chinois cantonais est la langue dominante à Hong Kong et à Macao.", [
    ("hint_language", "chinois cantonais"), ("hint_gpe", "Hong Kong"), ("hint_gpe", "Macao")])
_add("Le mapudungun est la langue ancestrale du peuple mapuche au Chili.", [
    ("hint_language", "mapudungun"), ("hint_norp", "mapuche"), ("hint_gpe", "Chili")])
_add("Le luxembourgeois est devenu langue nationale du Luxembourg en 1984.", [
    ("hint_language", "luxembourgeois"), ("hint_gpe", "Luxembourg"), ("hint_time_date", "1984")])

# ═══════════════════════════════════════════════════════════════
#  MIX supplémentaire
# ═══════════════════════════════════════════════════════════════

_add("Le positivisme juridique de Kelsen distingue strictement le droit et la morale.", [
    ("hint_concept", "positivisme juridique"), ("hint_person_name", "Kelsen")])
_add("Le Déclin et la chute de l'Empire romain de Gibbon est une œuvre monumentale en six volumes.", [
    ("hint_work_of_art", "Déclin et la chute de l'Empire romain"), ("hint_person_name", "Gibbon")])
_add("La loi de Moore prédit le doublement de la densité des transistors tous les deux ans.", [
    ("hint_concept", "loi de Moore"), ("hint_time_duration", "deux ans")])
_add("Le traité de Methuen de 1703 établit une alliance commerciale entre le Portugal et l'Angleterre.", [
    ("hint_law", "traité de Methuen"), ("hint_time_date", "1703"), ("hint_gpe", "Portugal"), ("hint_gpe", "Angleterre")])
_add("Le concept de soft power a été théorisé par Joseph Nye dans les années 1990.", [
    ("hint_concept", "soft power"), ("hint_person_name", "Joseph Nye"), ("hint_time_date", "années 1990")])
_add("La tuberculose décrite dans La Dame aux camélias de Dumas fils était alors incurable.", [
    ("hint_disease", "tuberculose"), ("hint_work_of_art", "Dame aux camélias"), ("hint_person_name", "Dumas fils")])
_add("Le traité de Nankin imposa à la Chine la cession de Hong Kong aux Britanniques.", [
    ("hint_law", "traité de Nankin"), ("hint_gpe", "Chine"), ("hint_gpe", "Hong Kong"), ("hint_norp", "Britanniques")])
_add("L'Esprit des lois de Montesquieu a influencé la séparation des pouvoirs dans les démocraties modernes.", [
    ("hint_work_of_art", "Esprit des lois"), ("hint_person_name", "Montesquieu"), ("hint_concept", "séparation des pouvoirs")])
_add("Le pacte Molotov-Ribbentrop de 1939 a permis le partage de la Pologne entre l'URSS et l'Allemagne.", [
    ("hint_law", "pacte Molotov-Ribbentrop"), ("hint_time_date", "1939"), ("hint_gpe", "Pologne")])
_add("Le Discours de la méthode de Descartes a posé les bases du rationalisme moderne.", [
    ("hint_work_of_art", "Discours de la méthode"), ("hint_person_name", "Descartes"), ("hint_concept", "rationalisme")])

# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/abstract_sentences_extra.jsonl")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    random.seed(args.seed)
    random.shuffle(SENTENCES)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, sent in enumerate(SENTENCES):
            row = {"id": f"abstract_extra_{i:04d}", "text": sent["text"], "spans": sent["spans"]}
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

