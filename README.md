# apartment-hunter

Outil de prédiction de prix immobilier (Madrid) basé sur FastAPI, Streamlit, scikit‑learn et Docker.

---

## 📋 Vue d'ensemble

Le projet expose:
- une API FastAPI pour la prédiction,
- une UI Streamlit pour saisir les caractéristiques et afficher le prix estimé.

Le modèle actuel utilise 10 variables et prédit le log‑prix pendant l'entraînement, puis retourne le prix en euros côté API.

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

## 📁 Structure du projet

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
│   ├── ridge_model.pkl
│   ├── preprocessor.pkl
│   ├── model_config.json         # colonnes du modèle (10), use_log, etc.
│   └── streamlit_config.json     # colonnes UI, ranges et catégories
├── raw_data/
├── pyproject.toml                 # gestion via uv
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
└── README.md
```

---

## 🚀 Lancer avec Docker Compose (recommandé)

1. Lancer l'application Docker Desktop
2. Sur un terminal, lancer la commande :
```bash
docker compose up -d --build
```

Accès:
- Streamlit: http://localhost:8501
- API (docs): http://localhost:8000/docs

Commandes utiles:
```bash
# redemarrer les service
docker compose restart api streamlit
docker compose logs -f api
docker compose logs -f streamlit
docker compose down
```

---

## 🔧 API

### Santé
```bash
curl http://localhost:8000/
```

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
- `neighborhood` est transmis en entier côté UI; l'API le convertit en chaîne pour le OneHotEncoder.
- En cas d'erreur 422, vérifier que les 10 champs sont fournis avec les bons types.

---

## 🖥️ UI Streamlit

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

## 🧠 Modèle & artefacts

Le notebook [3_model.ipynb](3_model.ipynb) entraîne un pipeline scikit‑learn:
- Prétraitement: `SimpleImputer` + `StandardScaler` (numériques) et `OneHotEncoder` (catégorie `neighborhood`, drop='first'),
- Modèle: `Ridge` entraîné sur `log(buy_price)`.

Artefacts sauvegardés dans `models/`:
- `ridge_model.pkl`, `preprocessor.pkl`,
- `model_config.json` (colonnes du modèle, `use_log`),
- `streamlit_config.json` (colonnes UI, ranges, valeurs catégorielles).

Après ré‑export, redémarrer les services pour la prise en compte:
 
docker compose restart api streamlit
```

---

## 🛠️ Dépannage

- 422 sur /predict: vérifier les 10 champs et types; relancer `docker compose restart api`.
- Valeurs `inf`/`nan`: vérifier que l'UI n'applique pas `exp()` côté client; l'API renvoie déjà des euros.
- Catégories inconnues: `neighborhood` doit correspondre aux valeurs de `streamlit_config.json` (l'API convertit en chaîne pour le OneHotEncoder).

---

## 📝 Licence

Projet de groupe - 2026
