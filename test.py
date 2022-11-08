import streamlit as st
from pycaret.classification import load_model

pipeline = load_model("trained_model")
pipeline