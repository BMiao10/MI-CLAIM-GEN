import os
import glob
import json
import regex as re
from collections import defaultdict

import pandas as pd
import streamlit as st
import huggingface_hub
from huggingface_hub import ModelCard

# Settings
OUTDIR="./data"
st.set_page_config(layout="wide")

@st.cache_resource
def save_model_headers(tag,
    limit=5):
    """
    Retrieve all model info by tag
    """
    # list all models
    models = huggingface_hub.list_models(full=True,
        cardData=True,
        sort='downloads',
        direction=-1,
        fetch_config=False,
        limit=limit,
        tags=tag)

    # Get model card headers
    all_model_headers = []
    model_id=0
    for model_card in models:
        mc_headers = get_model_card_headers(model_card)
        all_model_headers.append(mc_headers)
        model_id+=1

        # Periodically save
        if model_id%250==0:
            with open(f'{tag}_{model_id-250}_{model_id}.json', 'w') as f:
                json.dump(all_model_headers, f)
                all_model_headers = []

    with open(f'{tag}.json', 'w') as f:
        json.dump(all_model_headers, f)

def _get_model_card_headers(model_card_text:str):
    """Get headers from model cards"""
    return re.findall(r"(?<=\n)#[#]+ [\S ]*", model_card_text)

def get_model_card_headers(model_example):
    """Get model card headers for a set of models"""
    model_card = ModelCard.load(model_example.modelId,
                                ignore_metadata_errors=True)
    return {model_example.modelId: _get_model_card_headers(model_card.content)}

def _clean_single_header(curr_header):
    curr_header = re.sub(r'\W+ ', ' ', curr_header)
    curr_header = curr_header.replace("[optional]", "").strip().lower()
    curr_header = curr_header.replace("model ", "")

    return curr_header

def _get_model_header_proportions(headers):
    # Get coverage by all model cards
    headers_dict = defaultdict(lambda: 0)
    for mc_info in headers:
        clean_headers = [_clean_single_header(s) for s in list(mc_info.values())[0]]
        for header in clean_headers:
            headers_dict[header] = headers_dict[header] + 1

    return headers_dict

def _clean_model_headers(headers, top_k=10):
    """Clean up headers and return metrics on model card coverage"""

    # Get proportion of models with each header (limit to top_k)
    # TODO: cache this
    proportions = _get_model_header_proportions(headers)
    proportions_sorted = sorted(proportions.items(), key=lambda k_v: k_v[1], reverse=True)
    top_k_values = proportions_sorted[:top_k]
    total_models = len(headers)
    total_models_with_cards = len([list(h.values())[0] for h in headers if len(list(h.values())[0])>0])

    # Get current model headers
    model_names = [list(name.keys())[0] for name in headers]
    model_card = st.sidebar.selectbox("Show model card coverage", model_names)

    curr_model_headers = [s for s in headers if model_card==list(s.keys())[0]][0][model_card]
    curr_model_headers = [_clean_single_header(s) for s in curr_model_headers]

    # This model is missing XX (included in XX% of other model cards), XX (XX%), XX (XX%) etc
    # For each model, get headers
    missing = []
    for header_name, prop in top_k_values:
        if header_name not in curr_model_headers:
            missing.append(f"**{header_name.capitalize()}** (used in {prop*100/total_models_with_cards:.1f}% of other cards)")

    # Create tabs
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### This model card contains: ")
        for i in curr_model_headers:
            st.markdown("- " + i.capitalize())
    with col2:
        st.write("### This model is missing these common headers: ")
        for i in missing:
            st.markdown("- " + i)


def load_model_headers(input_tags):
    """Get display for model headers"""

    # Load headers
    all_headers = []
    for input_tag in input_tags:
        for header_file in glob.glob(f"{OUTDIR}/{input_tag}*.json"): 
            with open(header_file) as fname:
                subset_file = json.load(fname)
                all_headers.extend(subset_file)

    # Clean and return display dataframe
    _clean_model_headers(all_headers)

def create_dashboard():
    """
    Handles user input 
    """
    user_input = st.sidebar.multiselect("Search Huggingface models", options=["medical", "biomedical", "clinical"])

    for input_tag in user_input:
        if not os.path.exists(os.path.join(OUTDIR,f"{input_tag}.json")):
            save_model_headers(tag=input_tag, limit=None)
        
    load_model_headers(input_tags = user_input)
        
create_dashboard()




