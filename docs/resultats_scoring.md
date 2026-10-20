# Résultats de scoring via l'API

Page générée par `python -m src.ml.score --docs` ([`src/ml/score.py`](../src/ml/score.py)) : 100 vols tirés au hasard dans l'entrepôt sont envoyés à `POST /predict`, puis chaque prédiction est confrontée au retard réellement observé.

| | |
|---|---|
| Date du scoring | 20/10/2026 23:01 UTC |
| Modèle servi par l'API | `20261020225807` |
| API appelée | `http://localhost:8000` |
| Échantillon | 100 vols opérés, tirés parmi les 3 derniers mois de l'entrepôt (vols du 02/10/2025 au 31/12/2025), graine 42 |
| Fichier complet | [`scoring/scoring_20261020225807_20261020T230159Z.csv`](scoring/scoring_20261020225807_20261020T230159Z.csv) |

Ces vols appartiennent à la période de test : le modèle ne les a pas vus à l'entraînement. Le retard réel n'est jamais envoyé à l'API.

## Résumé

| Indicateur | Valeur |
|---|---|
| Vols scorés / en erreur | 100 / 0 |
| Vols prédits en retard (≥ 15 min) | 24,0 % |
| Vols réellement en retard | 19,0 % |
| Exactitude | 73,0 % |
| Précision (vols signalés réellement en retard) | 33,3 % |
| Rappel (retards réels détectés) | 42,1 % |
| Probabilité moyenne, vols réellement en retard | 24,0 % |
| Probabilité moyenne, vols à l'heure | 15,9 % |
| Erreur moyenne sur les minutes de retard | 25,0 min |

## Matrice de confusion

| | Prédit « à l'heure » | Prédit « en retard » |
|---|---|---|
| **Réellement à l'heure** | 65 | 16 |
| **Réellement en retard** | 11 | 8 |

Sur 100 vols, ces indicateurs varient sensiblement d'un tirage à l'autre. Les résultats de référence, calculés sur les 103 583 vols de la période de test, sont dans le [rapport de modélisation](rapport_modelisation.md).

## Détail des vols

Trié par probabilité de retard décroissante. Météo à l'aéroport de départ à l'heure prévue (température en °C, précipitations en mm/h, vent en km/h, conditions observées).

| Vol | Départ → arrivée | Heure prévue | Temp. | Précip. | Vent | Conditions | Probabilité | Prédit en retard | Réellement en retard | Minutes prédites | Minutes réelles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WN332_20251228_ATL | ATL → BWI | 18:40 | 15,7 | 0,0 | 0 | Nuageux | 51 % | oui | oui | 32 | 87 |
| F94026_20251012_ATL | ATL → MCO | 20:13 | 17,7 | 0,0 | 0 | Clair | 42 % | oui | oui | 35 | 70 |
| F92552_20251030_ATL | ATL → PHL | 22:55 | 10,7 | 0,0 | 7 | Nuageux | 41 % | oui | oui | 30 | 113 |
| DL1495_20251226_ATL | ATL → PBI | 22:45 | 15,7 | 0,0 | 13 | Nuageux | 40 % | oui | oui | 28 | 105 |
| DL455_20251212_JFK | JFK → DTW | 19:48 | -1,0 | 0,0 | 17 | Beau | 39 % | oui | oui | 29 | 56 |
| DL1310_20251102_ATL | ATL → MSP | 23:05 | 6,7 | 0,0 | 6 | Brouillard | 31 % | oui | non | 20 | 0 |
| DL1714_20251227_ATL | ATL → GSO | 20:51 | 16,7 | 0,0 | 7 | Nuageux | 31 % | oui | non | 39 | 0 |
| UA1693_20251207_ATL | ATL → ORD | 10:25 | 1,7 | 0,0 | 0 | Brouillard | 30 % | oui | oui | 26 | 319 |
| DL350_20251214_ATL | ATL → BOS | 13:08 | 2,7 | 0,0 | 30 | Clair | 29 % | oui | non | 17 | 0 |
| B6515_20251019_JFK | JFK → SFO | 21:30 | 18,0 | 0,0 | 30 | Nuageux | 29 % | oui | non | 20 | 12 |
| WN2147_20251201_ATL | ATL → BNA | 19:15 | 8,7 | 0,0 | 15 | Nuageux | 28 % | oui | oui | 18 | 73 |
| DL823_20251204_ATL | ATL → SFO | 15:59 | 7,7 | 0,0 | 7 | Nuageux | 28 % | oui | non | 16 | 0 |
| DL941_20251214_ATL | ATL → LAS | 20:15 | -4,2 | 0,0 | 6 | Clair | 28 % | oui | non | 32 | 0 |
| DL1195_20251205_ATL | ATL → HOU | 12:15 | 7,7 | 0,0 | 7 | Nuageux | 27 % | oui | oui | 12 | 123 |
| OH5471_20251219_ATL | ATL → DCA | 11:02 | 8,7 | 0,0 | 26 | Clair | 26 % | oui | non | 29 | 0 |
| DL2342_20251231_JFK | JFK → ATL | 12:59 | 0,0 | 0,0 | 32 | Nuageux | 26 % | oui | non | 18 | 13 |
| WN455_20251011_ATL | ATL → PHX | 17:40 | 22,7 | 0,0 | 15 | Clair | 25 % | oui | non | 11 | 8 |
| B62129_20251011_JFK | JFK → RSW | 15:00 | 19,0 | 0,0 | 28 | Couvert | 25 % | oui | non | 11 | 12 |
| B61128_20251106_JFK | JFK → PVD | 22:45 | 6,0 | 0,0 | 11 | Beau | 25 % | oui | non | 11 | 0 |
| F93045_20251019_ATL | ATL → MSP | 13:56 | 22,7 | 0,1 | 13 | Pluie légère | 25 % | oui | non | 15 | 0 |
| OO3844_20251215_ATL | ATL → SHV | 15:17 | 6,7 | 0,0 | 6 | Clair | 25 % | oui | non | 30 | 0 |
| DL1131_20251026_ATL | ATL → IAH | 17:45 | 11,7 | 0,0 | 11 | Pluie | 24 % | oui | non | 16 | 7 |
| B6318_20251217_JFK | JFK → BOS | 09:35 | 1,0 | 0,0 | 19 | Nuageux | 24 % | oui | non | 13 | 0 |
| B6738_20251126_JFK | JFK → BQN | 20:00 | 14,0 | 0,0 | 9 | Nuageux | 23 % | oui | non | 9 | 0 |
| DL1169_20251019_ATL | ATL → PHL | 15:15 | 23,7 | 0,0 | 22 | Beau | 22 % | non | non | 9 | 0 |
| DL545_20251125_ATL | ATL → SFO | 21:05 | 17,7 | 0,0 | 0 | Pluie | 21 % | non | non | 15 | 1 |
| AA484_20251016_JFK | JFK → MIA | 18:13 | 13,0 | 0,0 | 20 | Beau | 21 % | non | oui | 14 | 61 |
| B61128_20251002_JFK | JFK → PVD | 22:00 | 14,0 | 0,0 | 7 | Beau | 20 % | non | non | 10 | 0 |
| F91871_20251009_JFK | JFK → ATL | 09:00 | 12,8 | 0,0 | 18 | Beau | 20 % | non | non | 12 | 12 |
| F92634_20251101_ATL | ATL → BWI | 12:58 | 16,7 | 0,0 | 11 | Clair | 20 % | non | non | 12 | 1 |
| DL779_20251125_ATL | ATL → ONT | 21:20 | 17,7 | 0,0 | 0 | Pluie | 20 % | non | oui | 7 | 45 |
| DL1871_20251029_ATL | ATL → CHS | 17:49 | 11,7 | 0,0 | 7 | Pluie | 20 % | non | non | 10 | 0 |
| DL2542_20251213_ATL | ATL → MSY | 11:20 | 15,7 | 0,0 | 9 | Clair | 20 % | non | non | 8 | 0 |
| DL636_20251212_JFK | JFK → HNL | 09:55 | -2,0 | 0,0 | 30 | Beau | 19 % | non | non | 13 | 0 |
| B653_20251102_JFK | JFK → PBI | 11:00 | 14,0 | 0,0 | 13 | Beau | 19 % | non | non | 14 | 0 |
| WN373_20251118_ATL | ATL → HOU | 18:15 | 14,7 | 0,0 | 0 | Clair | 18 % | non | non | 8 | 0 |
| B6515_20251020_JFK | JFK → SFO | 21:30 | 15,0 | 0,0 | 26 | Beau | 18 % | non | non | 11 | 0 |
| DL1056_20251225_ATL | ATL → ICT | 20:16 | 16,7 | 0,0 | 9 | Nuageux | 18 % | non | non | 8 | 0 |
| OO3530_20251109_ATL | ATL → XNA | 19:10 | 8,7 | 0,0 | 24 | Nuageux | 17 % | non | non | 18 | 0 |
| DL2654_20251109_ATL | ATL → MSP | 17:59 | 12,7 | 0,0 | 24 | Nuageux | 17 % | non | oui | 7 | 16 |
| OO3844_20251030_ATL | ATL → SHV | 15:40 | 12,7 | 0,0 | 20 | Nuageux | 17 % | non | non | 12 | 0 |
| DL2960_20251219_ATL | ATL → MSP | 08:25 | 6,7 | 0,0 | 11 | Clair | 17 % | non | non | 6 | 2 |
| DL1320_20251121_ATL | ATL → CLE | 22:53 | 17,7 | 0,0 | 7 | Nuageux | 17 % | non | non | 11 | 0 |
| WN203_20251003_ATL | ATL → MSY | 14:05 | 23,7 | 0,0 | 13 | Nuageux | 17 % | non | non | 9 | 0 |
| DL770_20251019_ATL | ATL → ONT | 17:50 | 22,7 | 0,0 | 39 | Clair | 17 % | non | non | 8 | 0 |
| AS474_20251020_ATL | ATL → SAN | 15:38 | 21,7 | 0,0 | 6 | Clair | 17 % | non | non | 9 | 0 |
| YX4546_20251114_JFK | JFK → DCA | 20:00 | 7,0 | 0,0 | 11 | Nuageux | 17 % | non | oui | 8 | 48 |
| DL832_20251129_ATL | ATL → DCA | 17:58 | 8,7 | 0,0 | 11 | Nuageux | 16 % | non | non | 8 | 13 |
| DL2286_20251228_JFK | JFK → ATL | 06:00 | -6,0 | 0,0 | 13 | Beau | 16 % | non | non | 23 | 0 |
| DL1214_20251222_ATL | ATL → RIC | 15:35 | 14,7 | 0,0 | 9 | Nuageux | 16 % | non | oui | 6 | 93 |
| DL1295_20251015_ATL | ATL → FLL | 22:59 | 16,7 | 0,0 | 0 | Clair | 15 % | non | non | 9 | 0 |
| DL2182_20251215_ATL | ATL → HSV | 13:52 | 4,7 | 0,0 | 0 | Clair | 15 % | non | non | 6 | 0 |
| AA2221_20251004_ATL | ATL → LAX | 17:30 | 23,7 | 0,0 | 15 | Nuageux | 15 % | non | non | 12 | 0 |
| DL1812_20251104_ATL | ATL → TPA | 16:27 | 20,7 | 0,0 | 9 | Clair | 14 % | non | non | 7 | 0 |
| DL2584_20251224_ATL | ATL → ALB | 19:59 | 14,7 | 0,0 | 0 | Nuageux | 14 % | non | non | 8 | 0 |
| DL1794_20251007_ATL | ATL → SBA | 18:20 | 25,7 | 0,0 | 6 | Nuageux | 14 % | non | non | 18 | 0 |
| DL1699_20251215_ATL | ATL → RDU | 08:34 | -6,3 | 0,0 | 0 | Clair | 14 % | non | non | 7 | 3 |
| DL1320_20251117_ATL | ATL → CLE | 22:53 | 5,7 | 0,0 | 0 | Clair | 14 % | non | non | 8 | 5 |
| DL2270_20251208_ATL | ATL → PHL | 09:34 | 6,7 | 0,0 | 9 | Nuageux | 14 % | non | oui | 7 | 46 |
| DL404_20251204_ATL | ATL → LGA | 06:30 | -1,3 | 0,0 | 0 | Brouillard | 14 % | non | non | 7 | 0 |
| UA1693_20251208_ATL | ATL → ORD | 10:25 | 6,7 | 0,0 | 11 | Nuageux | 14 % | non | non | 11 | 0 |
| DL1372_20251125_ATL | ATL → MCO | 13:58 | 18,7 | 0,0 | 19 | Nuageux | 13 % | non | oui | 5 | 54 |
| AA1075_20251122_JFK | JFK → CLT | 14:50 | 10,0 | 0,0 | 15 | Nuageux | 13 % | non | non | 13 | 0 |
| DL1426_20251020_ATL | ATL → RIC | 21:20 | 10,7 | 0,0 | 0 | Clair | 13 % | non | non | 6 | 0 |
| OO4083_20251130_ATL | ATL → BTR | 09:10 | 3,7 | 0,1 | 9 | Nuageux | 13 % | non | non | 10 | 0 |
| WN1723_20251015_ATL | ATL → DAL | 13:35 | 26,7 | 0,0 | 7 | Clair | 13 % | non | non | 6 | 8 |
| WN2649_20251208_ATL | ATL → HOU | 08:00 | 5,7 | 0,0 | 13 | Nuageux | 13 % | non | non | 7 | 1 |
| DL2213_20251123_ATL | ATL → LEX | 22:48 | 9,7 | 0,0 | 0 | Clair | 13 % | non | oui | 5 | 605 |
| YX4456_20251223_ATL | ATL → LGA | 10:12 | 11,7 | 0,0 | 0 | Nuageux | 12 % | non | non | 6 | 0 |
| DL2324_20251105_ATL | ATL → JAX | 17:59 | 18,7 | 0,0 | 0 | Clair | 12 % | non | non | 5 | 0 |
| B61297_20251106_JFK | JFK → DEN | 08:25 | 10,0 | 0,0 | 32 | Beau | 12 % | non | non | 6 | 0 |
| DL1946_20251117_ATL | ATL → SJU | 13:00 | 18,7 | 0,0 | 13 | Clair | 12 % | non | non | 5 | 4 |
| DL481_20251023_ATL | ATL → SEA | 13:21 | 21,7 | 0,0 | 9 | Nuageux | 11 % | non | non | 5 | 0 |
| YX4767_20251122_JFK | JFK → CMH | 15:59 | 11,0 | 0,0 | 13 | Nuageux | 11 % | non | non | 5 | 0 |
| DL472_20251014_ATL | ATL → SEA | 16:30 | 27,7 | 0,0 | 11 | Clair | 11 % | non | non | 5 | 0 |
| B6523_20251213_JFK | JFK → LAX | 06:59 | -2,0 | 0,0 | 11 | Nuageux | 11 % | non | non | 6 | 0 |
| DL2690_20251106_ATL | ATL → SYR | 15:27 | 23,7 | 0,0 | 0 | Clair | 11 % | non | oui | 3 | 45 |
| DL2224_20251022_ATL | ATL → MSY | 22:30 | 9,7 | 0,0 | 0 | Clair | 11 % | non | non | 6 | 0 |
| DL1082_20251222_ATL | ATL → BDL | 11:00 | 11,7 | 0,0 | 11 | Nuageux | 11 % | non | non | 5 | 0 |
| DL855_20251003_ATL | ATL → BOS | 09:50 | 17,7 | 0,0 | 19 | Clair | 10 % | non | non | 5 | 0 |
| DL1871_20251114_ATL | ATL → CHS | 17:59 | 16,7 | 0,0 | 0 | Clair | 10 % | non | non | 6 | 0 |
| DL1353_20251121_ATL | ATL → PIT | 14:42 | 22,7 | 0,0 | 13 | Nuageux | 10 % | non | non | 4 | 0 |
| DL2490_20251108_ATL | ATL → OKC | 09:12 | 18,7 | 0,0 | 0 | Nuageux | 10 % | non | oui | 4 | 85 |
| YX4359_20251002_JFK | JFK → ORF | 15:10 | 19,0 | 0,0 | 7 | Nuageux | 10 % | non | non | 4 | 0 |
| DL1892_20251111_ATL | ATL → ALB | 09:37 | 2,7 | 0,0 | 15 | Clair | 9 % | non | non | 6 | 0 |
| DL3135_20251124_ATL | ATL → BTR | 13:00 | 21,7 | 0,0 | 17 | Nuageux | 9 % | non | non | 5 | 0 |
| B6489_20251130_JFK | JFK → SAN | 08:10 | 8,0 | 0,0 | 35 | Couvert | 8 % | non | non | 6 | 0 |
| DL2270_20251129_ATL | ATL → PHL | 08:55 | 2,7 | 0,0 | 13 | Beau | 8 % | non | non | 4 | 0 |
| DL1182_20251106_ATL | ATL → COS | 11:05 | 21,7 | 0,0 | 7 | Clair | 8 % | non | oui | 4 | 21 |
| DL513_20251002_ATL | ATL → LGA | 08:30 | 17,7 | 0,0 | 17 | Clair | 8 % | non | non | 4 | 0 |
| DL3095_20251118_ATL | ATL → GSP | 15:36 | 22,7 | 0,0 | 17 | Clair | 8 % | non | non | 5 | 1 |
| DL1973_20251111_ATL | ATL → CLT | 07:35 | -2,3 | 0,0 | 9 | Clair | 8 % | non | non | 5 | 0 |
| WN449_20251114_ATL | ATL → STL | 08:05 | 8,7 | 0,0 | 0 | Clair | 8 % | non | non | 3 | 0 |
| DL1481_20251006_ATL | ATL → SDF | 10:32 | 21,7 | 0,0 | 15 | Nuageux | 7 % | non | non | 3 | 0 |
| DL3090_20251018_ATL | ATL → JAN | 09:40 | 17,7 | 0,0 | 6 | Beau | 7 % | non | non | 3 | 4 |
| DL927_20251118_ATL | ATL → LAS | 08:07 | 7,7 | 0,0 | 0 | Clair | 7 % | non | non | 3 | 0 |
| DL2390_20251127_JFK | JFK → MSP | 07:00 | 6,0 | 0,0 | 24 | Beau | 7 % | non | non | 4 | 0 |
| YX5598_20251125_JFK | JFK → RDU | 12:15 | 12,0 | 0,0 | 19 | Couvert | 7 % | non | non | 3 | 0 |
| DL3216_20251028_ATL | ATL → SAV | 13:29 | 11,7 | 0,0 | 17 | Nuageux | 7 % | non | non | 3 | 0 |
| AA76_20251007_JFK | JFK → SFO | 06:00 | 18,0 | 0,0 | 13 | Beau | 6 % | non | non | 3 | 0 |
