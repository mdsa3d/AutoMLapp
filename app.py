# import streamlit library
import streamlit as st
import pandas as pd
import os
# import profiling
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# ml stuff
from pycaret.classification import setup, compare_models, pull, save_model

# create a side bar
with st.sidebar:
    st.image("https://img.icons8.com/external-soft-fill-juicy-fish/100/null/external-machine-voice-technology-soft-fill-soft-fill-juicy-fish.png")
    st.title("AutoML")
    # control the navigation suing radio
    choice = st.radio("Navigation", [
                                    "Upload",
                                    "Profiling",
                                    "ML",
                                    "Download"])
    # some information to display about the app
    st.info("This application allows yu to build an automated ML pipeline using Streamlit, Pandas profiling and Pycaret !")

source_file_dir = "temp\sourcedata.csv" 
if os.path.exists(source_file_dir):
    df = pd.read_csv(source_file_dir)
if choice == "Upload":
    st.title("Upload your data for modelling")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv(source_file_dir, index=None) # save data
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine Learning Modelling")
    # select box for the user to select the target
    target = st.selectbox("Select your target", df.columns)
    if st.button("Train Model"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the MOdel", f, "trained_model.pkl")