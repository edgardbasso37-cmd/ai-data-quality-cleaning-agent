#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Fonction pour automatiser sur streamlit _ _ _ _ _ _ _ _ _ _ __ _ _ _ _ _ _#

import pandas as pd
from groq import Groq
import os
import json
import io
from dotenv import load_dotenv
from tabulate import tabulate

# Nouveauté : Pydantic pour structurer la sortie
from pydantic import BaseModel, Field
from typing import List


def prepare_dataset_for_llm(df:pd.DataFrame,taille_echantillon: int =20):

    # Structure
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Échantillonnage pour éviter d'envoyer un gros jeu de données au llm
    sample_str = df.sample(
        min(taille_echantillon, len(df)),
        random_state=42
    ).to_markdown(index=False)

    # Valeurs manquantes
    missing_str = df.isna().sum().to_string()

    # Doublons
    duplicate_count = df.duplicated().sum()

    # Statistiques
    stats_str = df.describe().to_markdown()

    return {
        "info": info_str,
        "sample": sample_str,
        "missing": missing_str,
        "duplicates": duplicate_count,
        "statistics": stats_str
    }


# Définition de la structure attendue
class DataIssue(BaseModel):
    column: str = Field(..., description="Nom de la colonne concernée")
    issue_type: str = Field(..., description="Type de problème (ex: 'Format', 'Duplicata', 'Valeur manquante')")
    description: str = Field(..., description="Explication courte du problème")
    suggested_action: str = Field(..., description="Action Python suggérée (ex: 'pd.to_datetime', 'drop_duplicates')")

class DataAudit(BaseModel):
    issues: List[DataIssue]

class CodeOutput(BaseModel):
    python_script: str = Field(..., description="Le script Python complet et exécutable pour nettoyer les données, sans blocs markdown (```python).")


def prepare_dataset_for_llm(df:pd.DataFrame,taille_echantillon: int=20):

    # Structure
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Échantillonnage pour éviter d'envoyer un gros jeu de données au llm
    sample_str = df.sample(
        min(taille_echantillon, len(df)),
        random_state=42
    ).to_markdown(index=False)

    # Valeurs manquantes
    missing_str = df.isna().sum().to_string()

    # Doublons
    duplicate_count = df.duplicated().sum()

    # Statistiques
    stats_str = df.describe().to_markdown()

    return {
        "info": info_str,
        "sample": sample_str,
        "missing": missing_str,
        "duplicates": duplicate_count,
        "statistics": stats_str
    }

def analyze_dataframe(df: pd.DataFrame) -> List[dict]:
    """
    Analyse le dataframe et retourne une liste de problèmes structurés.
    """
    # Capture des infos techniques
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = df.head(50).to_markdown()

    # Prompt "Expert Impitoyable"
    df_llm=prepare_dataset_for_llm(df,20)
    prompt = f"""
Tu es un expert data analyst tres méticuleux 
Analyse ce dataset pour détecter les anomalies de qualité
(exemple: types, doublons, formats, valeurs manquantes) et d'autres que tu detecteras.

Retourne UNIQUEMENT un objet JSON.

Le JSON doit obligatoirement respecter exactement cette structure :

{{
    "issues": [
        {{
            "column": "nom de la colonne",
            "issue_type": "type du problème bien détaillé",
            "description": "description détaillée du problème"
            "suggested_action": "action Python suggérée"
        }}
    ]
}}

        S'il n'y a aucun problème, retourne :

        {{
            "issues": []
        }}

        ### STRUCTURE
        {df_llm["info"]}

        ### ÉCHANTILLON
        {df_llm["sample"]}

        ### VALEURS MANQUANTES
        {df_llm["missing"]}

        ### DOUBLONS
        {df_llm["duplicates"]}

        ### STATISTIQUES
        {df_llm["statistics"]}
        """

    try:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=GROQ_API_KEY)
        response_v2 = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
               response_format={
                "type": "json_object"
              }
             )
        
        response = response_v2.choices[0].message.content
                # Conversion Pydantic -> Dict pour Streamlit
                # Validation Pydantic
        audit_result = DataAudit.model_validate_json(response)
        
                # DEBUG LOG
                        
        if audit_result.issues:
                    print(
                        "DEBUG KEYS:",
                        audit_result.issues[0].model_dump().keys()
                    )
        return [
                issue.model_dump()
                for issue in audit_result.issues
                ]
    except Exception as e:
         
         print(f"Erreur Analyse LLM : {e}")
         return []
    

def generate_cleaning_code(df_sample: pd.DataFrame, selected_issues: List[dict]) -> str:
    """
    Génère le code Python pour appliquer les corrections sélectionnées.
    """
    if not selected_issues:
        return "print('Aucune action sélectionnée.')"

    cols_context = df_sample.dtypes.to_dict()

    prompt_f = f"""
        Tu es un développeur Python Senior expert en Pandas.

        Génère un script Python pour nettoyer un dataframe nommé 'df'
        en appliquant UNIQUEMENT ces actions validées :

        {json.dumps(selected_issues, indent=2, ensure_ascii=False)}

        CONTEXTE COLONNES :
        {cols_context}

        CONTRAINTES STRICTES :

        1. Le code doit agir directement sur la variable 'df'.
        2. Utilise des méthodes ROBUSTES :
        - pd.to_numeric(..., errors='coerce')
        - pd.to_datetime(..., errors='coerce')
        - regex pour nettoyer les strings si nécessaire.
        3. Gère les dates mixtes si mentionné.
        4. Retourne la réponse au format JSON.
        5. Le JSON doit obligatoirement avoir cette structure :

        {{
            "python_script": "code Python ici"
        }}

        6. La valeur de "python_script" doit contenir uniquement du code Python valide,
        sans markdown et sans ```python.
        7. LIBRARIES DISPONIBLES :
        pandas as pd, numpy as np, re.
        8. N'utilise PAS d'autres librairies.
        """

    try:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=GROQ_API_KEY)
        response_vf = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": prompt_f}],
                       response_format={
                        "type": "json_object"
                      }
        )
        response_vf = response_vf.choices[0].message.content



        # JSON -> dictionnaire Python
        result = json.loads(response_vf)

        # Récupération du script
        python_script = result.get("python_script")

        if not python_script:
            raise ValueError(
                "Le LLM n'a pas retourné la clé 'python_script'."
            )

        
        return python_script
    except Exception as e:
        raise RuntimeError(
            f"Erreur Génération Code : {e}"
        ) from e
