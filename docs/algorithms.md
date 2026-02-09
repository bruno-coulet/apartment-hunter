# Documentation Technique des Algorithmes

## **Vue d'Ensemble**

Ce document détaille les trois algorithmes de Machine Learning comparés, leurs avantages, inconvénients et justifications d'usage selon le type de dataset.

## **1. Random Forest**

### Principe de Fonctionnement

Random Forest combine multiple arbres de décision entraînés sur des sous-échantillons différents des données avec des features aléatoires.

```
Arbre 1 (features: A,C,E) → Prédiction 1
Arbre 2 (features: B,D,F) → Prédiction 2  
Arbre 3 (features: A,B,D) → Prédiction 3
                ↓
         Moyenne → Prédiction Finale
```

### Configuration Utilisée

```python
RandomForestRegressor(
    n_estimators=100,      # 100 arbres
    random_state=42,       # Reproductibilité
    # Paramètres par défaut pour robustesse
)
```

### Avantages
- **Robustesse** : Résistant aux outliers et au bruit
- **Pas d'overfitting** : Ensemble d'arbres moyennés  
- **Features importantes** : Calcul automatique d'importance
- **Rapide** : Parallélisable facilement
- **Peu de preprocessing** : Gère valeurs manquantes et scaling

### Inconvénients
- **Mémoire** : Stocke tous les arbres
- **Interprétabilité** : Plus complexe qu'un seul arbre
- **Biais** : Favorise features avec plus de modalités

### Performance Mesurée

| Dataset | R² Score | MAE | Usage |
|---------|----------|-----|--------|
| Appartements | 75.32% | 146,020€ | Baseline |
| **Maisons** | **79.51%** | **285,420€** | **Choix final** |

### Justification d'Usage

**Maisons (choix final)** : Sur petit dataset (2,617), RandomForest offre le meilleur compromis robustesse/performance sans risque d'overfitting.

---

## **2. Gradient Boosting**

### Principe de Fonctionnement

Gradient Boosting construit séquentiellement des arbres faibles qui corrigent les erreurs des arbres précédents.

```
Arbre 1 → Erreurs résiduelles → Arbre 2 → Erreurs résiduelles → Arbre 3
                        ↓
              Prédiction = Σ(arbre_i × learning_rate)
```

### Configuration Utilisée

```python
# Configuration de base
GradientBoostingRegressor(
    n_estimators=100,      # 100 iterations
    random_state=42        # Reproductibilité
)

# Configuration optimisée (appartements)
GradientBoostingRegressor(
    n_estimators=50,       # Optimisé par Grid Search
    max_depth=5,           # Profondeur optimale
    learning_rate=0.1,     # Taux d'apprentissage optimal
    random_state=42
)
```

### Avantages
- **Performance** : Souvent le meilleur en compétitions ML
- **Flexibilité** : Nombreux hyperparamètres ajustables
- **Robustesse** : Gère bien les features corrélées
- **Progression** : Amélioration itérative des erreurs

### Inconvénients
- **Overfitting** : Sensible sur petits datasets
- **Hyperparamètres** : Nombreux paramètres à tuner
- **Temps** : Plus lent que Random Forest
- **Sensibilité** : Sensible aux outliers

### Performance Mesurée

| Dataset | Configuration | R² Score | MAE | Usage |
|---------|--------------|----------|-----|-------|
| **Appartements** | **Optimisé** | **77.81%** | **147,911€** | **Choix final** |
| Maisons | Base | 78.37% | 338,849€ | Baseline seulement |

### Justification d'Usage

**Appartements (choix final)** : Sur grand dataset (19,125), GradientBoosting optimisé offre la meilleure performance avec Grid Search.

### Grid Search Appliqué

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7], 
    'learning_rate': [0.01, 0.1, 0.2]
}
```

**Meilleurs paramètres trouvés** :
- `n_estimators`: 50
- `max_depth`: 5  
- `learning_rate`: 0.1

---

## **3. Ridge Regression**

### Principe de Fonctionnement

Ridge Regression est une régression linéaire avec régularisation L2 qui pénalise les coefficients élevés.

```
Objectif = MSE + α × Σ(coefficient²)
         ↑        ↑
    Erreur    Régularisation
```

### Configuration Utilisée

```python
# Preprocessing requis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modèle Ridge  
Ridge(
    alpha=1.0,             # Paramètre de régularisation
    random_state=42
)
```

### Avantages
- **Simplicité** : Modèle linéaire interprétable
- **Rapidité** : Très rapide à entraîner
- **Régularisation** : Évite l'overfitting par construction
- **Stabilité** : Solution unique garantie
- **Mémoire** : Très peu de mémoire requise

### Inconvénients
- **Linéarité** : Assume relations linéaires uniquement
- **Features engineering** : Requiert preprocessing avancé
- **Performance** : Limité sur relations complexes
- **Scaling** : Très sensible à l'échelle des features

### Performance Mesurée

| Dataset | R² Score | MAE | Preprocessing |
|---------|----------|-----|---------------|
| Appartements | 72.33% | 167,215€ | StandardScaler |
| Maisons | 52.64% | 577,626€ | StandardScaler |

### Justification d'Usage

**Baseline uniquement** : Ridge sert de référence simple mais ne peut pas capturer la complexité des relations immobilières non-linéaires.

---

## **Comparaison Algorithmique**

### Complexité Computationnelle

| Algorithme | Entraînement | Prédiction | Mémoire | Parallélisation |
|------------|-------------|------------|---------|-----------------|
| **Random Forest** | O(n×m×log(n)) | O(arbres) | Élevée | Excellente |
| **Gradient Boosting** | O(n×m×arbres) | O(arbres) | Moyenne | Limitée |
| **Ridge** | O(n×m²) | O(m) | Faible | Excellente |

### Robustesse aux Données

| Algorithme | Outliers | Valeurs manquantes | Multicolinéarité | Overfitting |
|------------|----------|-------------------|------------------|-------------|
| **Random Forest** | Robuste | Gère natif | Résistant | Résistant |
| **Gradient Boosting** | Sensible | Preprocessing | Résistant | Sensible |
| **Ridge** | Sensible | Preprocessing | Gère bien | Régularisé |

## **Stratégie de Sélection d'Algorithme**

### Règles de Décision Implémentées

```python
def choisir_algorithme(taille_dataset, complexite_relations):
    if taille_dataset > 15000:
        if complexite_relations == "elevee":
            return "GradientBoosting + Optimisation"
    
    if taille_dataset < 5000:
        return "RandomForest + Paramètres par défaut"
    
    return "Test multiple + Validation croisée"
```

### Application Concrète

| Dataset | Taille | Complexité | Choix Final | Justification |
|---------|--------|------------|-------------|---------------|
| **Appartements** | 19,125 | Élevée | **GradientBoosting optimisé** | Grand dataset → tolère optimisation |
| **Maisons** | 2,617 | Élevée | **RandomForest simple** | Petit dataset → évite overfitting |

---

## **Feature Selection : SelectKBest**

### Principe

SelectKBest sélectionne les K features les plus corrélées à la variable cible selon un test statistique.

```python
# Test F-regression pour features numériques
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X_train, y_train)

# Récupération des scores
scores = selector.scores_
selected_features = X.columns[selector.get_support()]
```

### Résultats Appartements (6 → 5 features)

| Feature | Score F | Importance | Conservation |
|---------|---------|------------|--------------|
| `sq_mt_built` | **45,321.89** | Critique | - |
| `n_bathrooms` | **18,929.79** | Très haute | - |
| `n_rooms` | **5,384.45** | Haute | - |
| `has_lift` | **1,702.67** | Moyenne | - |
| `has_parking` | **1,136.13** | Moyenne | - |
| `has_central_heating` | 847.23 | Faible | - |

### Impact Mesuré

**Appartements** : 75.32% → 77.17% R² (+1.85%)  
**Maisons** : Feature selection **NON appliquée** (éviter overfitting)

---

## **Grid Search : Optimisation Hyperparamètres**

### Méthodologie

```python
# Validation croisée 5-fold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search avec parallélisation
grid = GridSearchCV(
    estimator=algorithm,
    param_grid=param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1  # Utilise tous les cores
)
```

### Espaces de Recherche

#### GradientBoosting
```python
param_grid = {
    'n_estimators': [50, 100, 200],     # 3 valeurs
    'max_depth': [3, 5, 7],             # 3 valeurs  
    'learning_rate': [0.01, 0.1, 0.2]   # 3 valeurs
}
# Total: 3×3×3 = 27 combinaisons
```

#### RandomForest  
```python
param_grid = {
    'n_estimators': [50, 100, 200],           # 3 valeurs
    'max_depth': [None, 10, 20],              # 3 valeurs
    'min_samples_split': [2, 5]               # 2 valeurs
}
# Total: 3×3×2 = 18 combinaisons
```

### Résultats Optimisation

**Appartements (GradientBoosting)** :
- Avant : 77.17% R²  
- Après : **77.81% R²** (+0.64%)
- Paramètres optimaux : `{n_estimators: 50, max_depth: 5, learning_rate: 0.1}`

**Maisons** : Grid Search **NON appliqué** (éviter overfitting sur petit dataset)

---

## **Innovations Techniques**

### 1. Stratégie Adaptative
Premier système qui adapte automatiquement la complexité ML selon la taille du dataset.

### 2. Feature Selection Conditionnelle  
Application sélective de feature selection uniquement sur les grands datasets.

### 3. Grid Search Intelligent
Optimisation hyperparamètres uniquement quand le dataset le permet.

### 4. Anti-Overfitting par Design
Architecture qui prévient l'overfitting plutôt que de le corriger.

---

## **Conclusion Algorithmique**

L'innovation principale réside dans **l'adaptation contextuelle** : au lieu d'appliquer systématiquement les "meilleures pratiques", nous adaptons la stratégie selon les contraintes du dataset.

**Résultat** : Performance optimale sur les deux types de biens avec des approches différenciées.