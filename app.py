import streamlit as st
import pandas as pd
import numpy as np
import os
import re  # Pour les regex dans le code généré
from dotenv import load_dotenv
import llm
import io

# --- Config ---
st.set_page_config(page_title="IA Data Cleaner", page_icon="🧹", layout="wide")

# Robust .env loading
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

#env_path = Path(__file__).parent / ".env" si app et env dans le meme dossier
load_dotenv(dotenv_path=env_path)

# --- CSS Custom ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Init pour charger les fichier en mémoire afin qu'il ne soient pas perdus à chaque execution---
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'issues' not in st.session_state:
    st.session_state['issues'] = None
if 'df_cleaned' not in st.session_state:
    st.session_state['df_cleaned'] = None
if 'cleaning_code' not in st.session_state:
    st.session_state['cleaning_code'] = None

# --- Sidebar ---
with st.sidebar:
    st.title("🔧 Config")
    
    # API Key est maintenant gérée uniquement par le backend (.env)
    api_key = os.getenv("GROQ_API_KEY")

    if api_key :
        st.success("la clé a bien chargée")
    # DEBUG : Si pas de clé, on affiche des infos utiles
    if not api_key:
        st.error("⚠️ Clé non trouvée.")
        st.caption(f"Chemin cherché : `{env_path}`")
        st.caption(f"Fichier existe ? : {env_path.exists()}")
        
    #st.divider()
    
    # on importe une base de donnée
    uploaded_file = st.file_uploader("Charger un CSV", type=["csv"])
    #on charge en memoire notre bd avec le nom df
    if uploaded_file and st.session_state['df'] is None:
        try:
            st.session_state['df'] = pd.read_csv(uploaded_file)
            st.success("Fichier chargé !")
        except Exception as e:
            st.error(f"Erreur lecture : {e}")

    st.info("Ce tool utilise un LLM pour analyser et nettoyer vos données.")

# --- Main Page ---
st.title("🧹 Modern Data Cleaning Agent")
st.markdown("### De l'analyse automatique à l'exécution Python")

if not api_key:
    st.error("⚠️ Clé API non trouvée ! Vérifiez votre fichier `.env`.")
    st.stop()

if st.session_state['df'] is None:
    st.info("👈 Chargez un fichier CSV pour commencer.")
    st.stop()

# 1. Aperçu des Données
st.subheader("1. Aperçu des Données Brutes")
st.dataframe(st.session_state['df'].head())
col1, _ ,col2 = st.columns(3)
col1.metric("Nombre de lignes", st.session_state['df'].shape[0])
col2.metric("Nombre de Colonnes", st.session_state['df'].shape[1])

st.divider()

# 2. Analyse Agentique
st.subheader("2. Audit & Validation (Human-in-the-loop)")

if st.button("🔍 Lancer l'Analyse IA", type="primary"):
    with st.spinner("L'IA examine votre fichier..."):
        issues = llm.analyze_dataframe(st.session_state['df'])
        st.session_state['issues'] = issues

# Affichage du Formulaire de Validation
selected_issues = []
if st.session_state['issues']:
    st.write("L'IA a détecté les problèmes suivants. Décochez ceux que vous voulez ignorer.")
    
    for i, issue in enumerate(st.session_state['issues']):
        # On crée une checkbox par issue
        is_checked = st.checkbox(
            f"**{issue.get('column', 'Inconnu')}** : {issue.get('description', 'Pas de description')} ({issue.get('suggested_action', 'Pas d action')})",
            value=True,
            key=f"issue_{i}"
        )
        if is_checked:
            selected_issues.append(issue)
            
    st.caption(f"{len(selected_issues)} actions sélectionnées sur {len(st.session_state['issues'])}.")
    
    st.divider()

    # 3. Génération & Exécution
    
    st.subheader("3. Nettoyage Automatique")
    
    if st.button("✨ Générer le Code & Nettoyer", type="primary"):
        with st.spinner("Génération du script Python en cours..."):
            # A. Génération Code
            code = llm.generate_cleaning_code(st.session_state['df'], selected_issues)
            st.session_state['cleaning_code'] = code
            
            # B. Exécution Sécurisée
            try:
                local_scope = {'df': st.session_state['df'].copy(), 'pd': pd, 'np': np, 're': re}
                exec(code, {}, local_scope)
                st.session_state['df_cleaned'] = local_scope['df']
                st.balloons()
            except Exception as e:
                st.error(f"Erreur lors de l'exécution du script : {e}")

# Résultats
if st.session_state['df_cleaned'] is not None:
    st.markdown("---")
    st.subheader("✅ Résultat Final")
    
    # Affichage Code utilisé
    with st.expander("Voir le code Python généré"):
        st.code(st.session_state['cleaning_code'], language='python')
    
    # Comparaison
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Avant")
        st.dataframe(st.session_state['df'].head())
    with col_b:
        st.write("Après")
        st.dataframe(st.session_state['df_cleaned'].head())
        
    st.success(f"Taille finale : {st.session_state['df_cleaned'].shape}")
    
    # Export
    csv = st.session_state['df_cleaned'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger le CSV Propre",
        data=csv,
        file_name="clean_data.csv",
        mime="text/csv",
    )