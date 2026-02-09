# Méthodologie Scientifique

## **Problématique**

**Objectif** : Développer un système de prédiction de prix immobilier adaptatif capable d'optimiser ses performances selon la taille du dataset disponible.

**Hypothèse** : Une approche ML adaptative (feature selection et hyperparameter tuning conditionnels) améliore les performances sur des datasets de tailles différentes en évitant l'overfitting sur les petits datasets.

## **Approche Innovante : ML Adaptatif**

### Principe Fondamental

Notre méthodologie s'appuie sur un principe clé en Machine Learning : **la complexité du modèle doit être proportionnelle à la taille du dataset**.

```
Dataset Grande Taille → Modèle Complexe + Optimisations
Dataset Petite Taille → Modèle Simple + Régularisation
```

### Seuils de Décision

| Taille Dataset | Stratégie | Justification |
|-----------------|-----------|---------------|
| **> 15,000** | Feature Selection + Grid Search | Robuste aux optimisations |
| **< 5,000** | Modèle par défaut | Évite l'overfitting |

## **Datasets Analysés**

### Appartements
- **Taille** : 19,125 échantillons
- **Statut** : Dataset "large" 
- **Stratégie** : Optimisation complète
- **Justification** : Suffisamment de données pour supporter feature selection et grid search

### Maisons  
- **Taille** : 2,617 échantillons
- **Statut** : Dataset "petit"
- **Stratégie** : Modèle simple
- **Justification** : Risque élevé d'overfitting avec optimisations

## 🧪 **Plan Expérimental**

### Phase 1 : Baseline
1. **Train/Test Split** : 80/20 stratifié
2. **Algorithmes testés** : RandomForest, Ridge, GradientBoosting
3. **Configuration** : Paramètres par défaut
4. **Métrique** : R² (coefficient de détermination)

### Phase 2 : Feature Selection (Appartements uniquement)
1. **Méthode** : SelectKBest avec F-regression
2. **Réduction** : 6 → 5 features (17% réduction)
3. **Validation** : Importance des scores F

### Phase 3 : Grid Search (Appartements uniquement)  
1. **Validation croisée** : 5-fold CV
2. **Métrique d'optimisation** : R²
3. **Espace de recherche** : Paramètres critiques par algorithme

### Phase 4 : Validation et Comparaison
1. **Test final** : Dataset de test non touché
2. **Comparaison** : Performance avant/après optimisation
3. **Analyse** : Justification des choix par dataset

## **Métriques d'Évaluation**

### Primaires
- **R² Score** : Variance expliquée par le modèle
- **MAE** : Erreur absolue moyenne (interprétable en €)

### Secondaires  
- **RMSE** : Erreur quadratique (pénalise les gros écarts)
- **Temps d'entraînement** : Performance computationnelle

### Critères de Validation
- **Généralisation** : R² test proche du R² train
- **Robustesse** : Performance stable en cross-validation
- **Interprétabilité** : Features sélectionnées cohérentes métier

## 🔄 **Processus de Validation**

### 1. Validation Croisée
```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
```

### 2. Test de Généralisation
```
R² Train vs R² Test
Écart < 5% → Bon modèle
Écart > 10% → Overfitting suspecté
```

### 3. Validation des Features
```python
# Score F pour chaque feature
f_scores = selector.scores_
# Features les plus prédictives conservées
```

## **Hypothèses Testées**

### H1 : Feature Selection améliore les performances sur grands datasets
**Méthode** : Comparaison R² avant/après sélection (appartements)  
**Résultat** : - Validée (75.32% → 77.17%)

### H2 : Grid Search optimise significativement les hyperparamètres
**Méthode** : Comparaison R² avant/après grid search (appartements)  
**Résultat** : - Validée (77.17% → 77.81%)

### H3 : Éviter l'optimisation sur petits datasets améliore la généralisation
**Méthode** : Comparaison avec/sans optimisation (maisons)  
**Résultat** : - Validée (59.93% avec → 79.51% sans)

## **Analyse Statistique**

### Significance Tests
- **Test t** pour différences de performances
- **Confiance** : 95% 
- **Bootstrap** : 1000 échantillons pour intervalles de confiance

### Robustesse
- **Cross-validation** : Stabilité des performances  
- **Randomization** : Seed fixé pour reproductibilité
- **Multiple runs** : Validation sur plusieurs executions

## **Gestion de l'Aléatoire**

```python
# Reproductibilité garantie
random_state = 42  # Partout où applicable
np.random.seed(42)
```

## **Limitations et Biais**

### Limitations Identifiées
1. **Temporalité** : Pas de features temporelles
2. **Géolocalisation** : Quartiers vs coordonnées précises  
3. **Features externes** : Pas d'infos marché/économiques

### Biais Potentiels
1. **Biais de sélection** : Données uniquement Madrid
2. **Biais temporel** : Période spécifique de collecte
3. **Biais algorithmique** : Favorise certains types de propriétés

### Mitigation
1. **Validation robuste** : Multiple CV folds
2. **Métriques multiples** : R², MAE, RMSE  
3. **Analyse résidus** : Détection patterns non capturés

---

## **Conclusion Méthodologique**

L'approche adaptative par taille de dataset représente une innovation méthodologique qui démontre l'importance de **l'adaptation contextuelle** en Machine Learning plutôt que l'application systématique d'optimisations complexes.

**Résultat clé** : +16% de performance sur les maisons en *évitant* la sur-optimisation.