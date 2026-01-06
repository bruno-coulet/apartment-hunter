# from pandas.api import types as ptypes


# ======== Séparer les colonnes booléennes des autres ========

# def is_boolean_like(series):
#     # dtype bool / nullable boolean
#     if ptypes.is_bool_dtype(series) or ptypes.is_bool_dtype(series.array):
#         return True

#     # integer/unsigned columns containing only 0/1 (ignorer les NaN)
#     if ptypes.is_integer_dtype(series) or ptypes.is_unsigned_integer_dtype(series):
#         vals = pd.unique(series.dropna())
#         if set(vals).issubset({0, 1}):
#             return True

#     # numeric columns that might be 0/1 stored as float
#     if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
#         vals = pd.unique(series.dropna())
#         if set(vals).issubset({0, 1}):
#             return True

#     # object / string columns containing only boolean-like strings
#     if ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series):
#         vals = set(series.dropna().astype(str).unique())
#         if vals.issubset({'0', '1', 'True', 'False', 'true', 'false'}):
#             return True

#     return False

# # détecter colonnes booléennes
# bool_cols = [col for col in df.columns if is_boolean_like(df[col])]
# num_cols = [col for col in df.columns if col not in bool_cols]

# # créer les deux dataframes (booléen et non booléen)
# df_bool = df[bool_cols].copy()
# df_num = df.drop(columns=bool_cols).copy()

# # résumé
# print(f"Total colonnes: {len(df.columns)}")
# print(f"Colonnes booléennes détectées ({len(bool_cols)}): {bool_cols}")
# print(f"df_bool.shape = {df_bool.shape}")
# print(f"df_quali_quanti.shape = {df_num.shape}")


# =================================================================

import pandas as pd
import numpy as np
from pandas.api import types as ptypes
from typing import Tuple, List

def is_boolean_like(series: pd.Series) -> bool:
    """Détecte si une série est booléenne ou 0/1 (même si stockée comme float/int/str)."""
    # dtype bool natif (inclut nullable boolean)
    try:
        if ptypes.is_bool_dtype(series.dtype) or ptypes.is_bool_dtype(series.array.dtype):
            return True
    except Exception:
        pass

    # valeurs non-null uniques converties en chaînes (minuscules) -> vérifier subset
    vals = series.dropna().unique()
    if vals.size == 0:
        return False

    str_vals = {str(v).strip().lower() for v in vals}
    allowed = {'0', '1', '0.0', '1.0', 'true', 'false'}
    if str_vals.issubset(allowed):
        return True

    return False

def split_by_dtype(
    df: pd.DataFrame,
    convert_bool: bool = True,
    bool_dtype: str = 'UInt8'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    """
    Sépare `df` en trois DataFrames :
      - df_bool : colonnes booléennes ou 0/1
      - df_num : colonnes numériques (excluant les booléennes détectées)
      - df_quali : colonnes restantes (qualitatives / catégorielles / texte)

    Paramètres :
      - convert_bool : si True, convertit les colonnes boolées détectées en dtype `bool_dtype`
      - bool_dtype : dtype pandas nullable à utiliser pour les bool (ex. 'UInt8' ou 'boolean')

    Retour :
      (df_bool, df_num, df_quali, bool_cols, num_col, quali_cols)
    """
    bool_cols = [col for col in df.columns if is_boolean_like(df[col])]
    num_col = [col for col in df.columns
                    if col not in bool_cols and ptypes.is_numeric_dtype(df[col].dtype)]
    quali_cols = [col for col in df.columns if col not in bool_cols + num_col]

    df_bool = df[bool_cols].copy()
    df_num = df[num_col].copy()
    df_quali = df[quali_cols].copy()

    if convert_bool and len(bool_cols) > 0:
        # Conversion sûre : on essaie d'abord par route numérique, sinon on mappe strings
        for col in bool_cols:
            s = df_bool[col]
            num = pd.to_numeric(s, errors='coerce')
            if num.dropna().isin([0, 1]).all():
                conv = num.round().astype(bool_dtype)
            else:
                mapdict = {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
                mapped = s.astype(str).str.strip().str.lower().map(mapdict)
                # Remplacer 'nan' mappés par <NA> pour permettre dtype nullable
                mapped = mapped.where(~mapped.isna(), pd.NA)
                conv = mapped.astype(bool_dtype)
            df_bool[col] = conv

    return df_bool, df_num, df_quali, bool_cols, num_col, quali_cols


# ======= utilisation =============

# from preprocess_utils import split_by_dtype

# df_bool, df_num, df_quali, bool_cols, num_col, quali_cols = split_by_dtype(df, convert_bool=True, bool_dtype='UInt8')

# print("bool:", bool_cols)
# print("numeric:", num_col)
# print("qualitative:", quali_cols)
# print(df_bool.shape, df_num.shape, df_quali.shape)


def detect_outliers_iqr(data, factor=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_z_score(data, threshold=3):
    z_scores = (data - data.mean()) / data.std()
    return np.abs(z_scores) > threshold