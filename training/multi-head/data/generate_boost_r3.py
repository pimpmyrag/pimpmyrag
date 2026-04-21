#!/usr/bin/env python3
"""Boost round 3 : toutes les classes < 250 remontées."""
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
#  hint_work_of_art  (~85 phrases, 167 → ~250)
# ═══════════════════════════════════════════════════════════════

_add("Mille et une nuits est un recueil de contes orientaux transmis depuis le IXe siècle.", [
    ("hint_work_of_art", "Mille et une nuits"), ("hint_time_date", "IXe siècle")])
_add("Le Monde selon Garp de John Irving est un roman picaresque américain.", [
    ("hint_work_of_art", "Monde selon Garp"), ("hint_person_name", "John Irving"), ("hint_norp", "américain")])
_add("Sous le soleil de Satan de Bernanos a remporté la Palme d'or sous la direction de Pialat.", [
    ("hint_work_of_art", "Sous le soleil de Satan"), ("hint_person_name", "Bernanos"), ("hint_person_name", "Pialat")])
_add("La Montagne magique de Thomas Mann se déroule dans un sanatorium des Alpes suisses.", [
    ("hint_work_of_art", "Montagne magique"), ("hint_person_name", "Thomas Mann")])
_add("Le Tambour de Günter Grass est un roman fondateur de la littérature allemande d'après-guerre.", [
    ("hint_work_of_art", "Tambour"), ("hint_person_name", "Günter Grass"), ("hint_norp", "allemande")])
_add("Blade Runner 2049 de Denis Villeneuve a été salué pour sa photographie exceptionnelle.", [
    ("hint_work_of_art", "Blade Runner 2049"), ("hint_person_name", "Denis Villeneuve")])
_add("La Cantatrice chauve d'Ionesco se joue sans interruption au théâtre de la Huchette depuis 1957.", [
    ("hint_work_of_art", "Cantatrice chauve"), ("hint_person_name", "Ionesco"), ("hint_fac_name", "théâtre de la Huchette"), ("hint_time_date", "1957")])
_add("Le Grand Meaulnes d'Alain-Fournier est un classique de la littérature adolescente.", [
    ("hint_work_of_art", "Grand Meaulnes"), ("hint_person_name", "Alain-Fournier")])
_add("Thérèse Raquin de Zola est un roman sombre sur la passion et le remords.", [
    ("hint_work_of_art", "Thérèse Raquin"), ("hint_person_name", "Zola")])
_add("Douze hommes en colère de Sidney Lumet est un huis clos judiciaire captivant.", [
    ("hint_work_of_art", "Douze hommes en colère"), ("hint_person_name", "Sidney Lumet")])
_add("Le Silence de la mer de Vercors a été écrit clandestinement sous l'Occupation.", [
    ("hint_work_of_art", "Silence de la mer"), ("hint_person_name", "Vercors")])
_add("Cyrano de Bergerac d'Edmond Rostand est la pièce de théâtre la plus jouée en France.", [
    ("hint_work_of_art", "Cyrano de Bergerac"), ("hint_person_name", "Edmond Rostand"), ("hint_gpe", "France")])
_add("La Condition humaine de Malraux se déroule pendant l'insurrection de Shanghai.", [
    ("hint_work_of_art", "Condition humaine"), ("hint_person_name", "Malraux"), ("hint_gpe", "Shanghai")])
_add("L'Amour aux temps du choléra de García Márquez mêle romance et épidémie.", [
    ("hint_work_of_art", "Amour aux temps du choléra"), ("hint_person_name", "García Márquez")])
_add("Le Pianiste de Roman Polanski retrace la survie d'un musicien juif dans le ghetto de Varsovie.", [
    ("hint_work_of_art", "Pianiste"), ("hint_person_name", "Roman Polanski"), ("hint_norp", "juif"), ("hint_gpe", "Varsovie")])
_add("Brokeback Mountain d'Ang Lee a brisé des tabous à sa sortie en 2005.", [
    ("hint_work_of_art", "Brokeback Mountain"), ("hint_person_name", "Ang Lee"), ("hint_time_date", "2005")])
_add("La Vie est belle de Roberto Benigni mêle comédie et horreur de l'Holocauste.", [
    ("hint_work_of_art", "Vie est belle"), ("hint_person_name", "Roberto Benigni")])
_add("Le Désert rouge d'Antonioni est un chef-d'œuvre de l'aliénation moderne.", [
    ("hint_work_of_art", "Désert rouge"), ("hint_person_name", "Antonioni")])
_add("Les Sept Samouraïs de Kurosawa a influencé des générations de réalisateurs.", [
    ("hint_work_of_art", "Sept Samouraïs"), ("hint_person_name", "Kurosawa")])
_add("Vent d'est de Jean-Luc Godard explore la politique à travers le cinéma expérimental.", [
    ("hint_work_of_art", "Vent d'est"), ("hint_person_name", "Jean-Luc Godard")])
_add("L'Insoutenable Légèreté de l'être de Milan Kundera interroge la liberté et le destin.", [
    ("hint_work_of_art", "Insoutenable Légèreté de l'être"), ("hint_person_name", "Milan Kundera")])
_add("Le Festin nu de William Burroughs a choqué par sa prose hallucinée.", [
    ("hint_work_of_art", "Festin nu"), ("hint_person_name", "William Burroughs")])
_add("Nostalghia de Tarkovski est une méditation sur l'exil et la mémoire.", [
    ("hint_work_of_art", "Nostalghia"), ("hint_person_name", "Tarkovski")])
_add("Sur la route de Jack Kerouac est le manifeste de la Beat Generation.", [
    ("hint_work_of_art", "Sur la route"), ("hint_person_name", "Jack Kerouac")])
_add("Autant en emporte le vent de Margaret Mitchell reste un classique du roman historique.", [
    ("hint_work_of_art", "Autant en emporte le vent"), ("hint_person_name", "Margaret Mitchell")])
_add("La Strada de Fellini a remporté l'Oscar du meilleur film étranger en 1957.", [
    ("hint_work_of_art", "Strada"), ("hint_person_name", "Fellini"), ("hint_time_date", "1957")])
_add("Les Harmonies Werckmeister de Béla Tarr est un film contemplatif de sept plans-séquences.", [
    ("hint_work_of_art", "Harmonies Werckmeister"), ("hint_person_name", "Béla Tarr")])
_add("Le Château ambulant de Miyazaki est un conte féerique inspiré du roman de Diana Wynne Jones.", [
    ("hint_work_of_art", "Château ambulant"), ("hint_person_name", "Miyazaki"), ("hint_person_name", "Diana Wynne Jones")])
_add("Les Temps modernes de Charlie Chaplin dénoncent l'aliénation industrielle.", [
    ("hint_work_of_art", "Temps modernes"), ("hint_person_name", "Charlie Chaplin")])
_add("Metropolis de Fritz Lang est un film muet visionnaire de 1927.", [
    ("hint_work_of_art", "Metropolis"), ("hint_person_name", "Fritz Lang"), ("hint_time_date", "1927")])
_add("Le Tombeau des lucioles de Takahata est l'un des films d'animation les plus bouleversants.", [
    ("hint_work_of_art", "Tombeau des lucioles"), ("hint_person_name", "Takahata")])
_add("Voyage au bout de l'enfer de Michael Cimino est un film marquant sur la guerre du Vietnam.", [
    ("hint_work_of_art", "Voyage au bout de l'enfer"), ("hint_person_name", "Michael Cimino")])
_add("Le Miroir de Tarkovski est un film autobiographique tissé de souvenirs d'enfance.", [
    ("hint_work_of_art", "Miroir"), ("hint_person_name", "Tarkovski")])
_add("La Nausée de Sartre est le roman fondateur de l'existentialisme littéraire.", [
    ("hint_work_of_art", "Nausée"), ("hint_person_name", "Sartre"), ("hint_concept", "existentialisme")])
_add("Le Bruit et la Fureur de Faulkner utilise quatre narrateurs pour raconter la même histoire.", [
    ("hint_work_of_art", "Bruit et la Fureur"), ("hint_person_name", "Faulkner")])
_add("Gatsby le Magnifique de Fitzgerald est un portrait du rêve américain brisé.", [
    ("hint_work_of_art", "Gatsby le Magnifique"), ("hint_person_name", "Fitzgerald"), ("hint_norp", "américain")])
_add("L'Attrape-rêves de Stephen King est un roman mêlant amitié, mémoire et horreur.", [
    ("hint_work_of_art", "Attrape-rêves"), ("hint_person_name", "Stephen King")])
_add("Tenet de Christopher Nolan joue avec l'inversion du temps.", [
    ("hint_work_of_art", "Tenet"), ("hint_person_name", "Christopher Nolan")])
_add("Le Septième Sceau d'Ingmar Bergman met en scène une partie d'échecs contre la Mort.", [
    ("hint_work_of_art", "Septième Sceau"), ("hint_person_name", "Ingmar Bergman")])
_add("La Chute d'Albert Camus est un monologue fiévreux dans les bars d'Amsterdam.", [
    ("hint_work_of_art", "Chute"), ("hint_person_name", "Albert Camus"), ("hint_gpe", "Amsterdam")])
_add("Les Âmes mortes de Gogol sont une satire de la Russie tsariste.", [
    ("hint_work_of_art", "Âmes mortes"), ("hint_person_name", "Gogol")])
_add("Le Mur de Sartre regroupe cinq nouvelles sur les limites de la liberté.", [
    ("hint_work_of_art", "Mur"), ("hint_person_name", "Sartre")])
_add("Alphaville de Godard est un film noir de science-fiction tourné dans le Paris réel.", [
    ("hint_work_of_art", "Alphaville"), ("hint_person_name", "Godard"), ("hint_gpe", "Paris")])
_add("Le Comte de Monte-Cristo d'Alexandre Dumas est un roman d'aventures et de vengeance.", [
    ("hint_work_of_art", "Comte de Monte-Cristo"), ("hint_person_name", "Alexandre Dumas")])
_add("Tokyo Story d'Ozu est un chef-d'œuvre du cinéma japonais sur la famille et le temps.", [
    ("hint_work_of_art", "Tokyo Story"), ("hint_person_name", "Ozu"), ("hint_norp", "japonais")])
_add("Le Bleu du ciel de Bataille mêle érotisme et politique dans l'Europe des années 1930.", [
    ("hint_work_of_art", "Bleu du ciel"), ("hint_person_name", "Bataille"), ("hint_time_date", "années 1930")])
_add("Les Enfants du paradis de Marcel Carné est considéré comme le plus grand film français.", [
    ("hint_work_of_art", "Enfants du paradis"), ("hint_person_name", "Marcel Carné"), ("hint_norp", "français")])
_add("Rashomon d'Akira Kurosawa propose quatre versions contradictoires d'un même crime.", [
    ("hint_work_of_art", "Rashomon"), ("hint_person_name", "Akira Kurosawa")])
_add("Siddhartha de Hermann Hesse est un roman initiatique inspiré du bouddhisme.", [
    ("hint_work_of_art", "Siddhartha"), ("hint_person_name", "Hermann Hesse"), ("hint_concept", "bouddhisme")])
_add("Le Moine de Matthew Lewis est un roman gothique anglais qui a scandalisé son époque.", [
    ("hint_work_of_art", "Moine"), ("hint_person_name", "Matthew Lewis"), ("hint_norp", "anglais")])
_add("Shoah de Claude Lanzmann est un documentaire monumental de neuf heures.", [
    ("hint_work_of_art", "Shoah"), ("hint_person_name", "Claude Lanzmann")])
_add("Le Rivage des Syrtes de Julien Gracq a décliné le prix Goncourt en 1951.", [
    ("hint_work_of_art", "Rivage des Syrtes"), ("hint_person_name", "Julien Gracq"), ("hint_time_date", "1951")])
_add("Les Cerfs-volants de Kaboul de Khaled Hosseini se déroule dans l'Afghanistan des talibans.", [
    ("hint_work_of_art", "Cerfs-volants de Kaboul"), ("hint_person_name", "Khaled Hosseini"), ("hint_gpe", "Afghanistan")])
_add("Plateforme de Michel Houellebecq a suscité la polémique à sa sortie en 2001.", [
    ("hint_work_of_art", "Plateforme"), ("hint_person_name", "Michel Houellebecq"), ("hint_time_date", "2001")])
_add("Au bonheur des dames de Zola décrit la naissance des grands magasins parisiens.", [
    ("hint_work_of_art", "Au bonheur des dames"), ("hint_person_name", "Zola"), ("hint_norp", "parisiens")])
_add("Le Bal de Irène Némirovsky a été publié à titre posthume en 2004.", [
    ("hint_work_of_art", "Bal"), ("hint_person_name", "Irène Némirovsky"), ("hint_time_date", "2004")])
_add("Twin Peaks de David Lynch a révolutionné les séries télévisées dans les années 1990.", [
    ("hint_work_of_art", "Twin Peaks"), ("hint_person_name", "David Lynch"), ("hint_time_date", "années 1990")])
_add("Grave of the Fireflies est souvent diffusé au Japon lors des commémorations du 15 août.", [
    ("hint_work_of_art", "Grave of the Fireflies"), ("hint_gpe", "Japon")])
_add("L'Ange bleu de Josef von Sternberg a lancé la carrière de Marlene Dietrich.", [
    ("hint_work_of_art", "Ange bleu"), ("hint_person_name", "Josef von Sternberg"), ("hint_person_name", "Marlene Dietrich")])
_add("Suite française d'Irène Némirovsky a été retrouvé dans une valise soixante ans après sa mort.", [
    ("hint_work_of_art", "Suite française"), ("hint_person_name", "Irène Némirovsky")])
_add("Le Clézio a reçu le Nobel pour Désert, un roman sur les Touaregs.", [
    ("hint_person_name", "Le Clézio"), ("hint_work_of_art", "Désert"), ("hint_norp", "Touaregs")])
_add("La Chevauchée fantastique de John Ford a codifié le western classique.", [
    ("hint_work_of_art", "Chevauchée fantastique"), ("hint_person_name", "John Ford")])
_add("Le Cid de Corneille est une tragi-comédie fondatrice du théâtre classique français.", [
    ("hint_work_of_art", "Cid"), ("hint_person_name", "Corneille"), ("hint_norp", "français")])
_add("Le Cuirassé Potemkine d'Eisenstein est un chef-d'œuvre du montage cinématographique.", [
    ("hint_work_of_art", "Cuirassé Potemkine"), ("hint_person_name", "Eisenstein")])
_add("Amadeus de Milos Forman mêle génie et jalousie dans la Vienne de Mozart.", [
    ("hint_work_of_art", "Amadeus"), ("hint_person_name", "Milos Forman"), ("hint_gpe", "Vienne"), ("hint_person_name", "Mozart")])
_add("Le Misanthrope de Molière explore les contradictions d'un homme honnête en société.", [
    ("hint_work_of_art", "Misanthrope"), ("hint_person_name", "Molière")])
_add("Nocturne de Chopin est l'une des pièces pour piano les plus jouées au monde.", [
    ("hint_work_of_art", "Nocturne"), ("hint_person_name", "Chopin")])
_add("Cléopâtre de Joseph Mankiewicz est resté longtemps le film le plus cher de l'histoire.", [
    ("hint_work_of_art", "Cléopâtre"), ("hint_person_name", "Joseph Mankiewicz")])

# ═══════════════════════════════════════════════════════════════
#  hint_concept  (~65 phrases, 186 → ~250)
# ═══════════════════════════════════════════════════════════════

_add("Le républicanisme défend la chose publique et la participation citoyenne.", [
    ("hint_concept", "républicanisme")])
_add("Le constitutionnalisme soumet le pouvoir politique au respect d'une loi fondamentale.", [
    ("hint_concept", "constitutionnalisme")])
_add("Le légitimisme défendait les droits de la branche aînée des Bourbons au trône de France.", [
    ("hint_concept", "légitimisme"), ("hint_gpe", "France")])
_add("L'orléanisme représentait la monarchie constitutionnelle libérale en France.", [
    ("hint_concept", "orléanisme"), ("hint_gpe", "France")])
_add("Le blanquisme prônait la prise du pouvoir par un petit groupe de révolutionnaires.", [
    ("hint_concept", "blanquisme")])
_add("Le proudhonisme oppose le mutualisme ouvrier à la propriété capitaliste.", [
    ("hint_concept", "proudhonisme"), ("hint_concept", "mutualisme")])
_add("Le bakouninisme a diffusé l'anarchisme collectiviste en Europe méridionale.", [
    ("hint_concept", "bakouninisme"), ("hint_concept", "anarchisme collectiviste")])
_add("Le syndicalisme révolutionnaire cherche à abolir le salariat par la grève générale.", [
    ("hint_concept", "syndicalisme révolutionnaire")])
_add("L'anarcho-syndicalisme combine l'action syndicale directe et l'autogestion ouvrière.", [
    ("hint_concept", "anarcho-syndicalisme")])
_add("Le municipalisme libertaire de Bookchin propose une démocratie directe locale.", [
    ("hint_concept", "municipalisme libertaire"), ("hint_person_name", "Bookchin")])
_add("Le communautarisme accorde une place centrale à l'appartenance communautaire.", [
    ("hint_concept", "communautarisme")])
_add("Le laïcisme strict refuse toute visibilité du religieux dans l'espace public.", [
    ("hint_concept", "laïcisme")])
_add("Le concordisme cherche à concilier les textes religieux et les découvertes scientifiques.", [
    ("hint_concept", "concordisme")])
_add("Le créationnisme s'oppose à la théorie de l'évolution en défendant une lecture littérale de la Bible.", [
    ("hint_concept", "créationnisme")])
_add("L'intelligent design présente la complexité du vivant comme preuve d'une conception intentionnelle.", [
    ("hint_concept", "intelligent design")])
_add("Le matérialisme éliminatif nie l'existence des états mentaux tels que les croyances.", [
    ("hint_concept", "matérialisme éliminatif")])
_add("Le compatibilisme concilie le libre arbitre avec le déterminisme causal.", [
    ("hint_concept", "compatibilisme"), ("hint_concept", "déterminisme")])
_add("L'incompatibilisme soutient que le libre arbitre et le déterminisme sont contradictoires.", [
    ("hint_concept", "incompatibilisme")])
_add("Le libertarianisme défend les droits individuels absolus contre l'intervention de l'État.", [
    ("hint_concept", "libertarianisme")])
_add("Le communisme libertaire combine l'abolition de l'État et la propriété collective.", [
    ("hint_concept", "communisme libertaire")])
_add("L'accélérationnisme pousse la logique capitaliste à son extrême pour provoquer son dépassement.", [
    ("hint_concept", "accélérationnisme")])
_add("Le catastrophisme en géologie attribue les formations terrestres à des événements violents.", [
    ("hint_concept", "catastrophisme")])
_add("L'uniformitarisme de Lyell a posé les bases de la géologie moderne.", [
    ("hint_concept", "uniformitarisme"), ("hint_person_name", "Lyell")])
_add("Le vitalisme attribue aux êtres vivants une force vitale distincte des lois physiques.", [
    ("hint_concept", "vitalisme")])
_add("Le mécanisme cartésien explique le fonctionnement du corps par les lois de la physique.", [
    ("hint_concept", "mécanisme cartésien")])
_add("Le parallélisme psychophysique affirme que corps et esprit évoluent de façon indépendante.", [
    ("hint_concept", "parallélisme psychophysique")])
_add("L'occasionnalisme de Malebranche attribue à Dieu la causalité entre corps et esprit.", [
    ("hint_concept", "occasionnalisme"), ("hint_person_name", "Malebranche")])
_add("Le nominalisme nie l'existence des universaux en dehors des mots qui les désignent.", [
    ("hint_concept", "nominalisme")])
_add("Le réalisme des universaux affirme que les catégories abstraites existent indépendamment de l'esprit.", [
    ("hint_concept", "réalisme des universaux")])
_add("Le conceptualisme considère que les universaux existent comme concepts dans l'esprit.", [
    ("hint_concept", "conceptualisme")])
_add("Le pragmatisme de Peirce évalue les idées par leurs conséquences pratiques observables.", [
    ("hint_concept", "pragmatisme"), ("hint_person_name", "Peirce")])
_add("L'instrumentalisme de Dewey conçoit les idées comme des outils de résolution de problèmes.", [
    ("hint_concept", "instrumentalisme"), ("hint_person_name", "Dewey")])
_add("Le falsificationnisme de Popper juge une théorie scientifique par sa réfutabilité.", [
    ("hint_concept", "falsificationnisme"), ("hint_person_name", "Popper")])
_add("Le paradigme kuhnien décrit les révolutions scientifiques comme des changements de vision du monde.", [
    ("hint_concept", "paradigme kuhnien")])
_add("Le programme fort de Bloor applique les méthodes sociologiques à l'explication de la science.", [
    ("hint_concept", "programme fort"), ("hint_person_name", "Bloor")])
_add("L'anarchisme épistémologique de Feyerabend refuse toute méthode scientifique unique.", [
    ("hint_concept", "anarchisme épistémologique"), ("hint_person_name", "Feyerabend")])
_add("Le behaviorisme logique de Ryle réduit les états mentaux à des dispositions comportementales.", [
    ("hint_concept", "behaviorisme logique"), ("hint_person_name", "Ryle")])
_add("Le structuralisme génétique de Goldmann relie les formes littéraires aux structures sociales.", [
    ("hint_concept", "structuralisme génétique"), ("hint_person_name", "Goldmann")])
_add("Le néomarxisme de l'École de Francfort critique la culture de masse et la raison instrumentale.", [
    ("hint_concept", "néomarxisme"), ("hint_org_name", "École de Francfort")])
_add("La théorie critique d'Adorno et Horkheimer analyse les mécanismes de domination culturelle.", [
    ("hint_concept", "théorie critique"), ("hint_person_name", "Adorno"), ("hint_person_name", "Horkheimer")])

# ═══════════════════════════════════════════════════════════════
#  hint_language  (~80 phrases, 173 → ~250)
# ═══════════════════════════════════════════════════════════════

_add("Le dimli est un dialecte du zazaki parlé dans l'est de la Turquie.", [
    ("hint_language", "dimli"), ("hint_language", "zazaki"), ("hint_gpe", "Turquie")])
_add("Le minnan est un dialecte chinois parlé à Taïwan et dans le Fujian.", [
    ("hint_language", "minnan"), ("hint_gpe", "Taïwan"), ("hint_gpe", "Fujian")])
_add("Le hakka est un dialecte chinois parlé par 30 millions de locuteurs dans le monde.", [
    ("hint_language", "hakka"), ("hint_quantity", "30 millions")])
_add("Le shanghainais est un dialecte wu en voie d'érosion face au mandarin.", [
    ("hint_language", "shanghainais"), ("hint_language", "mandarin")])
_add("Le tibétain standard est la langue officielle de la Région autonome du Tibet.", [
    ("hint_language", "tibétain standard")])
_add("Le dzongkha est la langue officielle du Bhoutan.", [
    ("hint_language", "dzongkha"), ("hint_gpe", "Bhoutan")])
_add("Le comorien est une langue bantoue parlée aux Comores et à Mayotte.", [
    ("hint_language", "comorien"), ("hint_gpe", "Comores"), ("hint_gpe", "Mayotte")])
_add("Le shimaore est le dialecte comorien parlé à Mayotte.", [
    ("hint_language", "shimaore"), ("hint_language", "comorien"), ("hint_gpe", "Mayotte")])
_add("Le fang est une langue bantoue parlée au Gabon, au Cameroun et en Guinée équatoriale.", [
    ("hint_language", "fang"), ("hint_gpe", "Gabon"), ("hint_gpe", "Cameroun")])
_add("Le twi est le dialecte le plus parlé de l'akan au Ghana.", [
    ("hint_language", "twi"), ("hint_language", "akan"), ("hint_gpe", "Ghana")])
_add("Le sotho du Nord est une langue officielle d'Afrique du Sud.", [
    ("hint_language", "sotho du Nord"), ("hint_gpe", "Afrique du Sud")])
_add("Le tigrinya est la principale langue d'Érythrée.", [
    ("hint_language", "tigrinya"), ("hint_gpe", "Érythrée")])
_add("L'oromo est la langue la plus parlée d'Éthiopie.", [
    ("hint_language", "oromo"), ("hint_gpe", "Éthiopie")])
_add("Le dioula est une langue mandingue véhiculaire en Côte d'Ivoire.", [
    ("hint_language", "dioula"), ("hint_gpe", "Côte d'Ivoire")])
_add("Le mandingue est un continuum linguistique couvrant l'Afrique de l'Ouest.", [
    ("hint_language", "mandingue"), ("hint_gpe", "Afrique de l'Ouest")])
_add("Le soninké est la langue du peuple soninké, présent au Mali et en Mauritanie.", [
    ("hint_language", "soninké"), ("hint_gpe", "Mali"), ("hint_gpe", "Mauritanie")])
_add("Le sérère est une langue atlantique parlée au Sénégal.", [
    ("hint_language", "sérère"), ("hint_gpe", "Sénégal")])
_add("Le diola est parlé en Casamance et en Gambie.", [
    ("hint_language", "diola"), ("hint_gpe", "Casamance"), ("hint_gpe", "Gambie")])
_add("Le ewé est une langue gbe parlée au Togo et au Ghana.", [
    ("hint_language", "ewé"), ("hint_gpe", "Togo"), ("hint_gpe", "Ghana")])
_add("Le fon est la langue la plus parlée au Bénin.", [
    ("hint_language", "fon"), ("hint_gpe", "Bénin")])
_add("Le kanouri est une langue saharienne parlée au Nigeria et au Niger.", [
    ("hint_language", "kanouri"), ("hint_gpe", "Nigeria"), ("hint_gpe", "Niger")])
_add("Le songhaï est une famille de langues parlées le long du fleuve Niger.", [
    ("hint_language", "songhaï")])
_add("Le zarma est un dialecte songhaï parlé au Niger.", [
    ("hint_language", "zarma"), ("hint_language", "songhaï"), ("hint_gpe", "Niger")])
_add("Le shona est la langue la plus parlée au Zimbabwe.", [
    ("hint_language", "shona"), ("hint_gpe", "Zimbabwe")])
_add("Le bemba est une langue bantoue majeure de la Zambie.", [
    ("hint_language", "bemba"), ("hint_gpe", "Zambie")])
_add("Le lunda est parlé en Zambie, en Angola et en RDC.", [
    ("hint_language", "lunda"), ("hint_gpe", "Zambie"), ("hint_gpe", "Angola")])
_add("Le chewa est la langue la plus parlée au Malawi et en Zambie.", [
    ("hint_language", "chewa"), ("hint_gpe", "Malawi"), ("hint_gpe", "Zambie")])
_add("Le sepedi est l'une des onze langues officielles de l'Afrique du Sud.", [
    ("hint_language", "sepedi"), ("hint_gpe", "Afrique du Sud")])
_add("Le venda est une langue bantoue parlée dans le nord de l'Afrique du Sud.", [
    ("hint_language", "venda"), ("hint_gpe", "Afrique du Sud")])
_add("Le tsonga est une langue bantoue parlée en Afrique australe.", [
    ("hint_language", "tsonga")])
_add("Le swazi est la langue officielle de l'Eswatini.", [
    ("hint_language", "swazi"), ("hint_gpe", "Eswatini")])
_add("Le sesotho sa Leboa est l'un des dialectes du sotho septentrional.", [
    ("hint_language", "sesotho sa Leboa")])
_add("Le kirghize utilise l'alphabet cyrillique depuis l'époque soviétique.", [
    ("hint_language", "kirghize"), ("hint_norp", "soviétique")])
_add("Le dari est la variante afghane du persan.", [
    ("hint_language", "dari"), ("hint_norp", "afghane"), ("hint_language", "persan")])
_add("Le baloutche est une langue iranienne parlée au Pakistan, en Iran et en Afghanistan.", [
    ("hint_language", "baloutche"), ("hint_gpe", "Pakistan"), ("hint_gpe", "Iran"), ("hint_gpe", "Afghanistan")])
_add("Le sindhi est une langue indo-aryenne parlée au Pakistan et en Inde.", [
    ("hint_language", "sindhi"), ("hint_gpe", "Pakistan"), ("hint_gpe", "Inde")])
_add("Le goudjrati est la langue maternelle de Gandhi.", [
    ("hint_language", "goudjrati"), ("hint_person_name", "Gandhi")])
_add("Le konkani est une langue officielle de l'État de Goa en Inde.", [
    ("hint_language", "konkani"), ("hint_gpe", "Goa"), ("hint_gpe", "Inde")])
_add("Le dogri est l'une des 22 langues officielles de l'Inde.", [
    ("hint_language", "dogri"), ("hint_gpe", "Inde")])
_add("Le maithili est parlé par 34 millions de locuteurs dans le Bihar indien.", [
    ("hint_language", "maithili"), ("hint_quantity", "34 millions")])
_add("Le santali est une langue austroasiatique parlée en Inde et au Bangladesh.", [
    ("hint_language", "santali"), ("hint_gpe", "Inde"), ("hint_gpe", "Bangladesh")])
_add("Le manipuri est une langue sino-tibétaine parlée dans le nord-est de l'Inde.", [
    ("hint_language", "manipuri"), ("hint_gpe", "Inde")])
_add("Le bodo est reconnu comme langue officielle de l'Inde depuis 2003.", [
    ("hint_language", "bodo"), ("hint_gpe", "Inde"), ("hint_time_date", "2003")])

# ═══════════════════════════════════════════════════════════════
#  hint_law  (~55 phrases, 196 → ~250)
# ═══════════════════════════════════════════════════════════════

_add("La loi Molière vise à imposer le français sur les chantiers publics.", [
    ("hint_law", "loi Molière"), ("hint_language", "français")])
_add("Le traité de Torún de 1466 fit passer la Prusse sous suzeraineté polonaise.", [
    ("hint_law", "traité de Torún"), ("hint_time_date", "1466"), ("hint_norp", "polonaise")])
_add("La Convention de Carthagène protège la biodiversité marine des Caraïbes.", [
    ("hint_law", "Convention de Carthagène")])
_add("Le décret de Canopus de 238 av. J.-C. est l'un des plus anciens décrets égyptiens conservés.", [
    ("hint_law", "décret de Canopus"), ("hint_norp", "égyptiens")])
_add("La loi Grammont de 1850 est la première loi française de protection animale.", [
    ("hint_law", "loi Grammont"), ("hint_time_date", "1850"), ("hint_norp", "française")])
_add("Le traité de Stolbovo de 1617 mit fin au conflit entre la Suède et la Russie.", [
    ("hint_law", "traité de Stolbovo"), ("hint_time_date", "1617"), ("hint_gpe", "Suède"), ("hint_gpe", "Russie")])
_add("La loi Élan de 2018 a réformé le logement et l'aménagement numérique.", [
    ("hint_law", "loi Élan"), ("hint_time_date", "2018")])
_add("Le traité de Nyons de 1289 rattacha le Dauphiné à la France.", [
    ("hint_law", "traité de Nyons"), ("hint_time_date", "1289"), ("hint_gpe", "France")])
_add("La charte Manden de 1236 est considérée comme l'une des premières déclarations des droits.", [
    ("hint_law", "charte Manden"), ("hint_time_date", "1236")])
_add("Le décret Millerand de 1899 accorda la journée de huit heures aux ouvriers des arsenaux.", [
    ("hint_law", "décret Millerand"), ("hint_time_date", "1899")])
_add("La loi de 1901 sur les associations reste le cadre juridique du monde associatif français.", [
    ("hint_law", "loi de 1901"), ("hint_norp", "français")])
_add("Le traité de Bassein de 1802 soumit les Marathes à la tutelle britannique.", [
    ("hint_law", "traité de Bassein"), ("hint_time_date", "1802"), ("hint_norp", "britannique")])
_add("La loi Doubin de 1989 a imposé la transparence dans les contrats de franchise.", [
    ("hint_law", "loi Doubin"), ("hint_time_date", "1989")])
_add("Le traité de Fredrikshamn de 1809 céda la Finlande à la Russie.", [
    ("hint_law", "traité de Fredrikshamn"), ("hint_time_date", "1809"), ("hint_gpe", "Finlande"), ("hint_gpe", "Russie")])
_add("La loi sur l'eau de 1992 a instauré les agences de l'eau en France.", [
    ("hint_law", "loi sur l'eau"), ("hint_time_date", "1992"), ("hint_gpe", "France")])
_add("Le décret de Compiègne de 877 fixa les règles de succession carolingienne.", [
    ("hint_law", "décret de Compiègne"), ("hint_time_date", "877")])
_add("La loi Barnier de 1995 a renforcé la protection de l'environnement en France.", [
    ("hint_law", "loi Barnier"), ("hint_time_date", "1995"), ("hint_gpe", "France")])
_add("Le traité de Gandamak de 1879 fit de l'Afghanistan un protectorat britannique.", [
    ("hint_law", "traité de Gandamak"), ("hint_time_date", "1879"), ("hint_gpe", "Afghanistan"), ("hint_norp", "britannique")])
_add("La loi Coppé-Zimmermann de 2011 a imposé des quotas de femmes dans les conseils d'administration.", [
    ("hint_law", "loi Coppé-Zimmermann"), ("hint_time_date", "2011")])
_add("Le traité d'Amiens de 1802 instaura une paix éphémère entre la France et le Royaume-Uni.", [
    ("hint_law", "traité d'Amiens"), ("hint_time_date", "1802"), ("hint_gpe", "France"), ("hint_gpe", "Royaume-Uni")])
_add("La convention de Sintra de 1808 provoqua un scandale en autorisant l'évacuation française du Portugal.", [
    ("hint_law", "convention de Sintra"), ("hint_time_date", "1808"), ("hint_norp", "française"), ("hint_gpe", "Portugal")])
_add("Le décret de Ventôse de 1794 promettait la redistribution des biens des émigrés.", [
    ("hint_law", "décret de Ventôse"), ("hint_time_date", "1794")])
_add("La loi Florange de 2014 a renforcé le droit de vote double des actionnaires fidèles.", [
    ("hint_law", "loi Florange"), ("hint_time_date", "2014")])
_add("Le Clean Air Act de 1970 a posé les bases de la régulation antipollution aux États-Unis.", [
    ("hint_law", "Clean Air Act"), ("hint_time_date", "1970"), ("hint_gpe", "États-Unis")])
_add("La Charte olympique définit les règles du Comité international olympique.", [
    ("hint_law", "Charte olympique"), ("hint_org_name", "Comité international olympique")])

# ═══════════════════════════════════════════════════════════════
#  hint_disease  (~55 phrases, 199 → ~250)
# ═══════════════════════════════════════════════════════════════

_add("Le syndrome d'Angelman provoque un retard de développement et des accès de rire.", [
    ("hint_disease", "syndrome d'Angelman")])
_add("La maladie de Kawasaki touche surtout les enfants de moins de cinq ans.", [
    ("hint_disease", "maladie de Kawasaki")])
_add("Le syndrome de Li-Fraumeni prédispose à de multiples cancers dès l'enfance.", [
    ("hint_disease", "syndrome de Li-Fraumeni")])
_add("La maladie de Stargardt est la dystrophie maculaire héréditaire la plus fréquente.", [
    ("hint_disease", "maladie de Stargardt")])
_add("Le syndrome de Bardet-Biedl associe obésité, rétinite pigmentaire et malformations rénales.", [
    ("hint_disease", "syndrome de Bardet-Biedl")])
_add("La maladie de Dercum se manifeste par des lipomes douloureux sous la peau.", [
    ("hint_disease", "maladie de Dercum")])
_add("Le syndrome de Noonan est une maladie génétique mimant certains traits du syndrome de Turner.", [
    ("hint_disease", "syndrome de Noonan"), ("hint_disease", "syndrome de Turner")])
_add("La maladie de Refsum est causée par une accumulation d'acide phytanique.", [
    ("hint_disease", "maladie de Refsum")])
_add("Le syndrome de Goodpasture attaque les poumons et les reins simultanément.", [
    ("hint_disease", "syndrome de Goodpasture")])
_add("La mastocytose est une prolifération anormale de mastocytes dans les tissus.", [
    ("hint_disease", "mastocytose")])
_add("Le syndrome de Budd-Chiari résulte d'une obstruction des veines hépatiques.", [
    ("hint_disease", "syndrome de Budd-Chiari")])
_add("La maladie de Norrie est une cécité héréditaire liée au chromosome X.", [
    ("hint_disease", "maladie de Norrie")])
_add("Le syndrome de Cockayne provoque un nanisme et un vieillissement prématuré.", [
    ("hint_disease", "syndrome de Cockayne")])
_add("La maladie de Farber est une lipogranulomatose rare du nourrisson.", [
    ("hint_disease", "maladie de Farber")])
_add("Le syndrome de Moebius est une paralysie congénitale des nerfs crâniens.", [
    ("hint_disease", "syndrome de Moebius")])
_add("La maladie de Vogt-Koyanagi-Harada provoque une uvéite bilatérale et une dépigmentation.", [
    ("hint_disease", "maladie de Vogt-Koyanagi-Harada")])
_add("Le cancer de la thyroïde est le cancer endocrinien le plus fréquent.", [
    ("hint_disease", "cancer de la thyroïde")])
_add("Le cancer du col de l'utérus est évitable grâce à la vaccination contre le papillomavirus.", [
    ("hint_disease", "cancer du col de l'utérus")])
_add("Le rétinoblastome est un cancer de l'œil qui touche principalement les enfants.", [
    ("hint_disease", "rétinoblastome")])
_add("Le neuroblastome est la tumeur solide extracrânienne la plus fréquente chez l'enfant.", [
    ("hint_disease", "neuroblastome")])
_add("Le sarcome d'Ewing touche les os des adolescents et des jeunes adultes.", [
    ("hint_disease", "sarcome d'Ewing")])
_add("L'ostéosarcome est le cancer primitif des os le plus courant.", [
    ("hint_disease", "ostéosarcome")])
_add("La maladie de Ménière affecte l'oreille interne et provoque des vertiges rotatoires.", [
    ("hint_disease", "maladie de Ménière")])
_add("Le syndrome de Wernicke-Korsakoff est lié à une carence en thiamine.", [
    ("hint_disease", "syndrome de Wernicke-Korsakoff")])
_add("La maladie de Berger est la néphropathie à IgA la plus répandue au monde.", [
    ("hint_disease", "maladie de Berger")])
_add("Le purpura thrombopénique idiopathique provoque des ecchymoses spontanées.", [
    ("hint_disease", "purpura thrombopénique idiopathique")])
_add("La maladie de Sever est une apophysite calcanéenne fréquente chez l'enfant sportif.", [
    ("hint_disease", "maladie de Sever")])
_add("Le syndrome de Lyell est une nécrolyse épidermique grave souvent médicamenteuse.", [
    ("hint_disease", "syndrome de Lyell")])

# ═══════════════════════════════════════════════════════════════
#  hint_rate  (~55 phrases, 198 → ~250)
# ═══════════════════════════════════════════════════════════════

_add("Le taux directeur de la Fed est resté à 5,25 % pendant six mois.", [
    ("hint_rate", "5,25 %"), ("hint_org_name", "Fed")])
_add("Le taux de croissance du Nigéria a atteint 3,3 % au dernier trimestre.", [
    ("hint_rate", "3,3 %"), ("hint_gpe", "Nigéria")])
_add("L'inflation sous-jacente reste à 2,8 % malgré le ralentissement économique.", [
    ("hint_rate", "2,8 %")])
_add("Le taux de vaccination a dépassé 85 % dans les pays du G7.", [
    ("hint_rate", "85 %")])
_add("Le rendement des obligations allemandes à dix ans est de 2,45 %.", [
    ("hint_rate", "2,45 %"), ("hint_norp", "allemandes")])
_add("Le PIB indien a progressé de 7,8 % en rythme annualisé.", [
    ("hint_rate", "7,8 %"), ("hint_norp", "indien")])
_add("Le taux de défaut sur les prêts immobiliers a grimpé à 1,2 % aux États-Unis.", [
    ("hint_rate", "1,2 %"), ("hint_gpe", "États-Unis")])
_add("La productivité du travail a reculé de 0,5 % au deuxième trimestre.", [
    ("hint_rate", "0,5 %")])
_add("Le taux de scolarisation primaire dépasse 98 % dans les pays de l'OCDE.", [
    ("hint_rate", "98 %"), ("hint_org_name", "OCDE")])
_add("Le CAC 40 a progressé de 1,8 % en une séance.", [
    ("hint_rate", "1,8 %"), ("hint_org_name", "CAC 40")])
_add("Le taux de change du yuan face au dollar a perdu 3,6 % depuis janvier.", [
    ("hint_rate", "3,6 %")])
_add("Le taux de fécondité en Corée du Sud est tombé à 0,72 enfant par femme.", [
    ("hint_rate", "0,72"), ("hint_gpe", "Corée du Sud")])
_add("Le spread entre les obligations italiennes et allemandes a atteint 180 points de base.", [
    ("hint_rate", "180 points de base"), ("hint_norp", "italiennes"), ("hint_norp", "allemandes")])
_add("L'indice des prix à la consommation a augmenté de 0,3 % en avril.", [
    ("hint_rate", "0,3 %")])
_add("Le taux d'épargne des ménages a bondi de 5 points pendant le confinement.", [
    ("hint_rate", "5 points")])
_add("Le taux d'absentéisme dans les entreprises a atteint 6,7 % en moyenne.", [
    ("hint_rate", "6,7 %")])
_add("Le Dow Jones a reculé de 2,1 % à la clôture.", [
    ("hint_rate", "2,1 %"), ("hint_org_name", "Dow Jones")])
_add("Le taux de pauvreté relative est de 14,6 % en France métropolitaine.", [
    ("hint_rate", "14,6 %"), ("hint_gpe", "France")])
_add("Le taux d'endettement des entreprises françaises dépasse 150 % du PIB.", [
    ("hint_rate", "150 %"), ("hint_norp", "françaises")])
_add("La croissance de la zone euro a plafonné à 0,9 % en 2024.", [
    ("hint_rate", "0,9 %"), ("hint_time_date", "2024")])
_add("Le taux de recyclage des emballages plastiques atteint 27 % en Europe.", [
    ("hint_rate", "27 %"), ("hint_gpe", "Europe")])
_add("Le prix de l'or a augmenté de 12 % depuis le début de l'année.", [
    ("hint_rate", "12 %")])
_add("Le taux d'intérêt moyen des crédits immobiliers est de 3,9 % sur vingt ans.", [
    ("hint_rate", "3,9 %")])
_add("Le taux de réussite en licence est de 33 % en trois ans dans les universités françaises.", [
    ("hint_rate", "33 %"), ("hint_norp", "françaises")])
_add("Le Nasdaq a bondi de 3,2 % après les résultats trimestriels des géants de la tech.", [
    ("hint_rate", "3,2 %"), ("hint_org_name", "Nasdaq")])

# ═══════════════════════════════════════════════════════════════
#  hint_tool  (~50 phrases, 203 → ~250)
# ═══════════════════════════════════════════════════════════════

_add("Le tensiomètre mesure la pression artérielle en milieu clinique et à domicile.", [
    ("hint_tool", "tensiomètre")])
_add("Le spiromètre évalue la fonction pulmonaire en mesurant les volumes d'air expiré.", [
    ("hint_tool", "spiromètre")])
_add("Le réfractomètre mesure l'indice de réfraction d'un liquide ou d'un matériau.", [
    ("hint_tool", "réfractomètre")])
_add("Le densimètre mesure la densité des liquides par le principe d'Archimède.", [
    ("hint_tool", "densimètre")])
_add("Le pyromètre permet de mesurer des températures très élevées sans contact.", [
    ("hint_tool", "pyromètre")])
_add("Le colorimètre analyse la composition chimique d'une solution par sa couleur.", [
    ("hint_tool", "colorimètre")])
_add("L'hygromètre mesure le taux d'humidité dans l'air ambiant.", [
    ("hint_tool", "hygromètre")])
_add("Le luxmètre mesure l'éclairement lumineux d'un environnement en lux.", [
    ("hint_tool", "luxmètre")])
_add("Le polarimètre mesure la rotation du plan de polarisation de la lumière.", [
    ("hint_tool", "polarimètre")])
_add("Le photomètre mesure l'intensité lumineuse reçue par un capteur.", [
    ("hint_tool", "photomètre")])
_add("Le turbidimètre évalue la turbidité de l'eau potable.", [
    ("hint_tool", "turbidimètre")])
_add("Le détecteur de métaux est utilisé en archéologie et en sécurité aéroportuaire.", [
    ("hint_tool", "détecteur de métaux")])
_add("La jauge de contrainte mesure les déformations mécaniques d'une structure.", [
    ("hint_tool", "jauge de contrainte")])
_add("Le théodolite mesure les angles horizontaux et verticaux en topographie.", [
    ("hint_tool", "théodolite")])
_add("Le tachéomètre combine les fonctions de théodolite et de télémètre laser.", [
    ("hint_tool", "tachéomètre"), ("hint_tool", "théodolite")])
_add("L'altimètre mesure l'altitude d'un aéronef par rapport au sol.", [
    ("hint_tool", "altimètre")])
_add("Le sextant a permis aux navigateurs de déterminer leur latitude en mer.", [
    ("hint_tool", "sextant")])
_add("Le gravimètre détecte les variations infimes du champ de gravité terrestre.", [
    ("hint_tool", "gravimètre")])
_add("Le magnétomètre mesure l'intensité et la direction du champ magnétique.", [
    ("hint_tool", "magnétomètre")])
_add("L'autoclave stérilise les instruments chirurgicaux par la vapeur sous pression.", [
    ("hint_tool", "autoclave")])
_add("Le microtome découpe des tranches de tissu biologique pour l'examen au microscope.", [
    ("hint_tool", "microtome")])
_add("La centrifugeuse sépare les composants d'un mélange par la force centrifuge.", [
    ("hint_tool", "centrifugeuse")])
_add("Le chromatographe sépare et identifie les composés chimiques d'un échantillon.", [
    ("hint_tool", "chromatographe")])
_add("L'endoscope permet d'explorer l'intérieur du corps humain sans chirurgie invasive.", [
    ("hint_tool", "endoscope")])
_add("Le coloscope est un type d'endoscope utilisé pour l'examen du côlon.", [
    ("hint_tool", "coloscope"), ("hint_tool", "endoscope")])

# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/boost_weak_r3.jsonl")
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()

    random.seed(args.seed)
    random.shuffle(SENTENCES)

    with open(args.output, "w", encoding="utf-8") as f:
        for i, sent in enumerate(SENTENCES):
            row = {"id": f"boost3_{i:04d}", "text": sent["text"], "spans": sent["spans"]}
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

