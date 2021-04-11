import streamlit as st
import numpy as np 
import pandas as pd 


def main():
    st.title("Visualize Data")
    
    st.write("### Select which countries(s) you would like to learn more about!")        
    
    if st.button("Submit"):
        st.balloons()
  