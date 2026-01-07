# apartment-hunter


### Rôles des différents fichiers

**cleaning.ipynb**<br>
Importe les données brutes

fais un premier tri des variables inutiles (vide, une seule modalité, redondantes, ...)

Exporte dans data/ les données pré-nettoyées


**analysis.ipynb**<br>
Analyse les variables restante, les corrélations, selectionne les variables utiles



**fill_nan.ipynb**<br>
découpe le jeu de donnée en X, y, train et test

remplace les nan de X_train et y_train

Exporte X_train_imputed, y_train, X_test_imputed, y_test dans le dossier data_model



**model.ipynb**<br>
importe X_train et y_train pour faire la modélisation