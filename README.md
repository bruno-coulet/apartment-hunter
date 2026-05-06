# apartment-hunter

Outil de prédiction de prix immobilier (Madrid) basé sur FastAPI, Streamlit, scikit‑learn et Docker.

---

## Vue d'ensemble

Le projet expose:
- une **API FastAPI** (Backend) pour la prédiction,
- une **UI Streamlit** (Frontend) pour saisir les caractéristiques et afficher le prix estimé.

**Modèle sélectionné** : **XGBoost** (meilleure performance)
- 10 variables d'entrée
- Entraîné sur le segment standard (prix ≤ 1.15M€)
- Prédiction du log-prix en entraînement, conversion en euros côté API
- **Performance** : MAE = 55.4 k€, RMSE = 82.2 k€, MAPE = 15.48% (sur test standard)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│         Streamlit UI (Port 8501)                    │
│  - Formulaire et affichage                          │
└──────────────┬──────────────────────────────────────┘
               │ HTTP
               ↓
┌─────────────────────────────────────────────────────┐
│            FastAPI (Port 8000)                      │
│  - GET /           (santé)                          │
│  - POST /predict  (prédiction)                      │
└──────────────┬──────────────────────────────────────┘
               │
               ↓
        Préprocesseur + Modèle (pickle)
```

---

## Structure du projet

```
apartment-hunter/
├── api.py
├── streamlit_app/
│   ├── app.py
│   └── style.css
├── 1_cleaning.ipynb
├── 2_analysis.ipynb
├── 3_model.ipynb
├── cleaning_utils.py
├── data_cleaned/
├── data_model/
├── models/
│   ├── xgboost_model.pkl         # Modèle XGBoost (meilleur)
│   ├── preprocessor.pkl          # Pipeline preprocessing sklearn
│   ├── model_config.json         # Config API : colonnes (10), use_log, segment
│   └── streamlit_config.json     # Config UI : ranges et valeurs catégories
├── raw_data/
├── pyproject.toml                 # gestion via uv
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
└── README.md
```

---

## Option 1 : Utiliser Docker Compose (recommandé)

1. Lancer l'application Docker Desktop.
2. Dans un terminal, exécuter la commande suivante :
```bash
docker compose up -d --build
```

3. (optionel) Vérifier si les conteneurs sont bien actifs :
```bash
docker compose ps
```

4. Accès à l'interface FastAPI.  
[http://localhost:8000/docs](http://localhost:8000/docs)
5. Accès à l'interface Streamlit.  
[http://localhost:8501](http://localhost:8501)



Commandes utiles :
```bash
# Relancer les services
docker compose up -d

# Redémarrer les services
docker compose restart api streamlit

# Voir les logs
docker compose logs -f api
docker compose logs -f streamlit

# Arrêter les services
docker compose down
```

---

## Option 2 : Démarrer les conteneurs manuellement

Cette option sert quand l'environnement Docker a déjà été préparé : les images sont déjà construites et les conteneurs ont déjà été créés une première fois.

Dans ce cas, pas besoin de relancer un build complet. Il suffit d'ouvrir Docker Desktop et de démarrer les conteneurs existants, ou les relancer depuis le terminal si besoin.

Il faut vérifier que les images nécessaires sont bien présentes en local, sinon il faudra revenir à l'option 1 pour reconstruire l'ensemble.

---

## API

### Santé
Cette commande appelle la route racine GET / de l'API. Elle ne lance pas de prédiction : elle sert juste à vérifier que le service FastAPI répond bien.

Si tout fonctionne, l'API renvoie un petit JSON avec son état, par exemple si le service tourne et si le modèle a bien été chargé.

```bash
curl http://localhost:8000/
```

Si cette commande répond, cela veut dire que l'API est accessible avant de tester /predict.

### Prédire un prix
Entrée attendue (10 features):
```json
{
  "sq_mt_built": 100.0,
  "n_rooms": 3,
  "n_bathrooms": 2,
  "neighborhood": 77,
  "has_lift": 1,
  "has_parking": 0,
  "has_pool": 0,
  "has_garden": 0,
  "has_storage_room": 0,
  "is_floor_under": 0
}
```

Exemple:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sq_mt_built": 100.0,
    "n_rooms": 3,
    "n_bathrooms": 2,
    "neighborhood": 77,
    "has_lift": 1,
    "has_parking": 0,
    "has_pool": 0,
    "has_garden": 0,
    "has_storage_room": 0,
    "is_floor_under": 0
  }'
```

Réponse:
```json
{
  "prediction": 579857.56,   // euros (déjà dé-log)
  "prediction_log": 13.2705, // informatif
  "status": "success"
}
```

Notes:
- `neighborhood` est transmis en **entier** (int) côté UI/JSON, mais l'API le convertit automatiquement en **string** pour le OneHotEncoder (cohérence avec l'entraînement).
- En cas d'erreur 422, vérifier que les 10 champs sont fournis avec les bons types.
- **Important** : Les valeurs de `neighborhood` doivent être comprises entre 1 et 135 (IDs de quartiers Madrid).

---

## UI Streamlit

L'UI consomme `models/streamlit_config.json` pour:
- la liste des colonnes d'entrée,
- les plages `ranges` pour les numériques,
- les valeurs catégorielles (`neighborhood`).

Affichage:
- `n_bathrooms` est un entier,
- le prix est formaté à la française (ex: `389.788,00 €`).

Lancer localement (hors Docker):
```bash
uv run streamlit run streamlit_app/app.py
```

---

## Modèle & Performance

Le notebook [3_model.ipynb](3_model.ipynb) entraîne et compare deux modèles :

### Modèle final : **XGBoost** ✅
- **Avantage** : Capture les interactions non-linéaires entre variables
- **Entraînement** : Segment standard (prix ≤ 1.15M€ → 95% du marché)
- **Performance (Test Standard)** :
  - **R²** : 0.9105 (91% de variance expliquée)
  - **MAE** : 55.4 k€ (erreur moyenne)
  - **RMSE** : 82.2 k€ (pénalise les grandes erreurs)
  - **MAPE** : 15.48% (erreur relative)

### Pourquoi XGBoost vs Ridge ?
| Métrique | Ridge | XGBoost | Gain |
|----------|-------|---------|------|
| MAE | 70.7 k€ | 55.4 k€ | **-21.6%** |
| RMSE | 133.8 k€ | 82.2 k€ | **-38.6%** |
| MAPE | 17.52% | 15.48% | **-11.7%** |

### Artefacts sauvegardés
- `xgboost_model.pkl` : Modèle entraîné
- `preprocessor.pkl` : Pipeline (StandardScaler + OneHotEncoder)
- `model_config.json` : Config API (colonnes, segment, threshold)
- `streamlit_config.json` : Config UI (ranges, catégories)

Après ré-entraînement et export du modèle, redémarrer les services :
```bash
docker compose restart api streamlit
```

**Note** : Le modèle n'accepte que le segment standard (≤1.15M€). Les biens de luxe retourneront une erreur ou une prédiction dégradée.

---

## Dépannage

- **`neighborhood` non pris en compte** : ✅ **[CORRIGÉ]** L'API convertit désormais `neighborhood` en string (cohérence avec l'entraînement). Si le problème persiste, vérifier que `xgboost_model.pkl` est bien chargé (ligne 19 de `api.py`).
- **422 sur /predict** : Vérifier les 10 champs, types, et que les valeurs sont dans les ranges de `streamlit_config.json`. Relancer `docker compose restart api`.
- **Valeurs inf/nan** : L'API retourne déjà les euros (conversion automatique de log). Ne pas appliquer `exp()` côté client.
- **Catégories inconnues** : `neighborhood` doit correspondre aux valeurs de `streamlit_config.json` (IDs 1-135). L'API convertit en chaîne pour le OneHotEncoder.
- **Segment luxe (>1.15M€)** : Le modèle n'a pas été entraîné sur ce segment. Résultats non fiables. Une v2 avec modèle luxe est envisagée.

---

## Licence

Projet de groupe - 2026
