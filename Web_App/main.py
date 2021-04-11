import streamlit as st

import app1, app2
import streamlit as st


Pages = {"Predict Scores": app1, "Visualize Data": app2}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(Pages.keys()))

Pages[selection].main()



