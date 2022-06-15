import streamlit as st
from firebase import  firebase
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import pandas as pd
from scipy.fft import fftshift,fft, fftfreq, rfft, rfftfreq, dct, idct, dst, idst
import os
import glob
import json
from PIL import Image
import csv
import pymysql
import boto3
import json


#Streamlit GUI starts from here
st.set_page_config(
	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="collapsed",  # Can be "auto", "expanded", "collapsed"
	page_title=None,  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)

a=st.sidebar.radio('Navigation',['Farm Information','Farmer Data'])
# df = pd.read_csv("Trebirth.csv")


st.header("Welcome to Trebirth Tech Development")
#  form = st.form(key='my_form',clear_on_submit=True)
#  F_name= form.text_input(label='Enter Farmer Name')
#  F_health= form.text_input(label='Enter Farm Health')
#  Number= form.number_input(label='Enter No. of trees scanned')
#  Remark = form.text_area(label='Remark')
#  submit_button = form.form_submit_button(label='Submit
 
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     dataframe = pd.read_csv(uploaded_file)
     Np_array = np.squeeze(np.array(dataframe.iloc[:,[1]]))
     st.write(Np_array)

generate_graph_button = st.button("Generate Graphs")

if generate_graph_button:
	st.write("Graphs Generated!")
	filtered_array = Apply_Filter(Np_array)
	Plot_Graph(filtered_array)
	#st.write(Np_array)
	Calculate_FFT(Np_array)
	Calculate_DCT(Np_array)
	Calculate_DST(Np_array)
	Calculate_STFT2(Np_array)
	Calculate_Phase_Spectrum(Np_array)
