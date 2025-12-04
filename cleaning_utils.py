# utils.py
import pandas as pd
import numpy as np

# ===============================
# Fonctions de nettoyage et exploration de données
# ===============================

# Colonnes vides
def empty_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes dont toutes les valeurs sont NaN"""
    return df.columns[df.isna().all()].tolist()

# Colonnes avec une seule valeur unique (incluant NaN)
def unique_value_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes avec une seule valeur unique"""
    return df.columns[df.nunique(dropna=False) <= 1].tolist()

# Colonnes de type string / object
def string_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes de type string ou object"""
    return df.select_dtypes(include=['object', 'string']).columns.tolist()

# Colonnes booléennes ou True/False/NaN
def boolean_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes booléennes ou contenant seulement True/False/NaN"""
    bool_cols = []
    for col in df.columns:
        s = df[col].dropna().unique()
        if set(s).issubset({True, False}):
            bool_cols.append(col)
    return bool_cols

# Colonnes numériques
def numeric_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes numériques (int, float)"""
    return df.select_dtypes(include=['number']).columns.tolist()

# Colonnes avec beaucoup de NaN
def high_na_columns(df: pd.DataFrame, threshold: float = 0.5) -> list:
    """Retourne les colonnes avec plus de `threshold` proportion de NaN"""
    return df.columns[df.isna().mean() > threshold].tolist()

# Colonnes catégorielles avec trop de modalités
def high_cardinality_columns(df: pd.DataFrame, max_modalities: int = 20) -> list:
    """
    Retourne les colonnes object/string qui ont plus de `max_modalities` valeurs uniques
    """
    cat_cols = string_columns(df)
    high_card_cols = [col for col in cat_cols if df[col].nunique(dropna=True) > max_modalities]
    return high_card_cols

# Colonnes avec valeurs nulles ou équivalentes
def missing_like_columns(df: pd.DataFrame) -> list:
    """
    Colonnes contenant des valeurs manquantes explicites : NaN, None, '', 'na', 'null'
    """
    missing_vals = {np.nan, None, '', 'na', 'NA', 'null', 'NULL'}
    cols = []
    for col in df.columns:
        if df[col].isin(missing_vals).any():
            cols.append(col)
    return cols

# Supprimer colonnes et garder trace
def drop_columns(df: pd.DataFrame, cols: list, dropped: list = None) -> pd.DataFrame:
    """
    Supprime les colonnes et optionnellement ajoute à la liste `dropped`
    """
    if dropped is not None:
        dropped.extend(cols)
    return df.drop(columns=cols)

# Conversion booléen -> UInt8
def convert_bool_to_uint8(df: pd.DataFrame, cols: list, keep_na: bool = True) -> pd.DataFrame:
    """
    Convertit True/False/NaN en UInt8
    keep_na=True : True->1, False->0, NaN->NaN
    keep_na=False: True->1, False->0, NaN->0
    """
    for col in cols:
        if keep_na:
            df[col] = df[col].astype('boolean').astype('UInt8')
        else:
            df[col] = df[col].astype('boolean').fillna(False).astype('UInt8')
    return df

# Conversion texte en minuscules pour colonnes existantes
def lower_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Convertit en minuscules uniquement les colonnes existantes et typées string/object
    """
    # Colonnes existantes dans le DataFrame
    existing_cols = [c for c in cols if c in df.columns]
    
    # Filtrer pour ne garder que les colonnes string/object
    string_cols = [c for c in existing_cols if pd.api.types.is_string_dtype(df[c])]
    
    for col in string_cols:
        df[col] = df[col].str.lower()
    
    return df


# Créer colonne catégorielle selon mots-clés
def add_type_column(df: pd.DataFrame, col_source: str, mapping: dict, col_dest: str = 'type') -> pd.DataFrame:
    """
    Crée une nouvelle colonne selon un mapping de mots-clés dans la colonne source
    mapping = {'piso': 'piso', 'casa': 'casa o chalet', ...}
    """
    df[col_dest] = None
    for key, value in mapping.items():
        mask = df[col_source].str.contains(key, na=False)
        df.loc[mask & df[col_dest].isna(), col_dest] = value
    return df

# Imputation simple pour colonnes numériques
def impute_numeric(df: pd.DataFrame, cols: list = None, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute les valeurs manquantes des colonnes numériques
    strategy: 'mean', 'median', 'zero'
    """
    if cols is None:
        cols = numeric_columns(df)
    for col in cols:
        if strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'zero':
            df[col] = df[col].fillna(0)
    return df

# Imputation simple pour colonnes catégorielles
def impute_categorical(df: pd.DataFrame, cols: list = None, fill_value: str = 'missing') -> pd.DataFrame:
    """Remplit les valeurs manquantes des colonnes object/string par fill_value"""
    if cols is None:
        cols = string_columns(df)
    for col in cols:
        df[col] = df[col].fillna(fill_value)
    return df

# ===============================
# Exemple rapide d'utilisation
# ===============================
if __name__ == "__main__":
    # df = pd.DataFrame({
    #     'a': [1,1,1,None],
    #     'b': [None,None,None,None],
    #     'c': [True, False, True, None],
    #     'd': ['Hello', 'World', None, 'Test'],
    #     'e': ['na', 'NA', '', None, 'valid']
    # })
    df = pd.read_csv("raw_data/houses_Madrid.csv", index_col=1)

    print("Empty cols:", empty_columns(df))
    print("Unique value cols:", unique_value_columns(df))
    print("Bool cols:", boolean_columns(df))
    print("String cols:", string_columns(df))
    print("High NA cols:", high_na_columns(df))
    print("High cardinality cols:", high_cardinality_columns(df, max_modalities=2))
    print("Missing-like cols:", missing_like_columns(df))
