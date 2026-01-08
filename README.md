# apartment-hunter


### Rôles des différents fichiers

**1_cleaning.ipynb**<br>
Importe les données brutes

fais un premier tri des variables inutiles (vide, une seule modalité, redondantes, ...)

Exporte dans data/ les données pré-nettoyées


**2_analysis.ipynb**<br>
Analyse les variables restante, les corrélations, selectionne les variables utiles


**3_model.ipynb**<br>
importe X_train et y_train pour faire la modélisation

Pour garder le fichier `requirements.txt` reflète toujours la réalité (par exemple si les collègues n'utilisent pas encore uv), on peut faut le régénérer avec la commande :

```shell
uv export --format requirements-txt --no-dev -o requirements.txt
```

### Créer l'image Docker
```shell
docker build -t apartment-api .


### Run l'image Docker
```shell
docker run -p 8000:8000 apartment-api
```
