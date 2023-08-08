import streamlit as st
import pandas as pd
import logging

from preprocess_docs import preprocess_docs, vector_stores, generate_embeddings
from custom_pipeline import create_pipeline
from utils import SingletonToken
from update_table import update_dataframe

import time

if __name__ == '__main__':

    def preprocess():
        doc_dir = r"C:\Users\johna\anaconda3\envs\longfunctioncall_env\long_functioncall\knowledge_base" # replace with your own directory
        
        with st.spinner("Preprocessing docs..."):
            time.sleep(5)
            docs = preprocess_docs(doc_dir) 
        
        with st.spinner("Creating document store..."):
            time.sleep(2)
            document_store = vector_stores(docs)
        
        with st.spinner("Generating Embeddings..."):
            time.sleep(3)
            retriever = generate_embeddings(document_store)

        with st.spinner("initialising dataframe.."):
            # Define the columns
            columns = ['Product_reference_num', 'fund_name', 'investment_objective', 'investment_strategy', 'investment_policy', 'ESG']
            
            # Create an empty DataFrame with the specified columns
            df = pd.DataFrame(columns=columns)
                
        return docs, document_store, retriever, df
        
    if "docs" not in st.session_state:
        (
            st.session_state["docs"],
            st.session_state["document_store"],
            st.session_state["retriever"],
            st.session_state["df"]
        ) = preprocess()

st.markdown(
    """
    #### Prototype Built by [Data-Centric Solutions](https://www.data-centric-solutions.com/)
    """,
    unsafe_allow_html=True
)

# Side panel for OpenAI token input
st.sidebar.title("Configuration")
API_KEY = st.sidebar.text_input("Enter OpenAI Key", type="password")

# Initialize an empty placeholder
placeholder = st.empty()

if API_KEY:
    SingletonToken.set_token(API_KEY)
    API_KEY = SingletonToken.get_token()

    with st.spinner("Building pipeline..."):
        st.session_state["pipeline"] = create_pipeline(st.session_state["retriever"], API_KEY)

    # If OpenAI key and data_url are set, enable the chat interface
    st.title("Extract your data")
    query_user = placeholder.text_input("Which fund should I analyse?...")
    
    if st.button("Submit"):
        with st.spinner('Extracting Data...'):
            res = st.session_state["pipeline"].run(query_user)
            Product_reference_num = res['Product_reference_num']
            investment_objective = res['investment_objective']
            investment_strategy = res['investment_strategy']
            investment_policy = res['investment_policy']
            ESG = res['ESG']
            fund_name = res['fund_name']

            st.session_state["df"] = update_dataframe(st.session_state["df"], Product_reference_num, investment_objective, investment_strategy, investment_policy, ESG, fund_name)
            
            st.write(st.session_state["df"])
else:
    # If OpenAI key and data_url are not set, show a message
    placeholder.markdown(
        """
        **Please enter your OpenAI key and data URL in the sidebar.**
        
        Follow this [link](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/) to get your OpenAI API key.
        """,
        unsafe_allow_html=True,
    )