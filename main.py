import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import streamlit_extras
from Homepage import Home
from preprocessing import Preprocessing
from modelling import Clustering
from prediction import Prediction
from test import Plotting


def main():
    # setting the sidebar
    with st.sidebar:
        raw_data = st.file_uploader("Upload csv file")
        st.image("file.png", width=150)

    # Link to html code
    with open("style.html", "r") as html_file:
        html_content = html_file.read()
        st.sidebar.markdown(html_content, unsafe_allow_html=True)
    # Sign-out button
    if st.sidebar.button("Sign Out"):
        st.session_state.login_success = False
        st.session_state.animation = False
        st.session_state.tries = 4

    # Options menu
    options = option_menu(
        menu_title=None,
        options=["Homepage", "Preprocessing", "Clusterings", "Visualizations", "Predictions"],
        icons=["house", "gear", "collection-fill", "gear", "arrow-repeat"],
        orientation="horizontal"
    )
    # Global variables

    if "preprocessed_data" not in st.session_state:
        st.session_state.preprocessed_data = None
    # Pages --> Instantiations
    if raw_data:
        df = pd.read_csv(raw_data)
        if options == "Homepage":
            home = Home(df)
            home.datasetOverview()

        if options == "Preprocessing":
            preprocessing = Preprocessing(df)
            scale, label, data = preprocessing.text()
            st.session_state.scaler = scale
            st.session_state.labeller = label
            st.session_state.preprocessed_data = data
        if options == "Clusterings":
            model = Clustering(st.session_state.preprocessed_data)
            model.text()
        if options == "Visualizations":
            visual = Plotting()
            visual.pairplot()
        if options == "Predictions":
            model_instance = Prediction()
            model_instance.cluster_predict()
    else:
        st.warning("No file uploaded")
