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


# Digital Filter starts from here
def Data_Preprocess(x):
 sig = [np.array(x)]
 # print("Sig is ",sig)
 return sig

def Apply_Filter(sig):
    sos = signal.butter(1, [0.1,10], 'band', fs=100, output='sos')
    filtered = signal.sosfilt(sos, sig)
    # print ("Filtered data is ",filtered)
    return filtered.squeeze()


def Plot_Graph(filtered):
   t = np.linspace(0, 30,3000, False)
   t = t[:filtered.size]
   fig = px.line(x=t, y=filtered, labels={'x':'Time', 'y':'Amplitude'},title='Time Series', width = 1000, height = 600)
   fig.update_traces(line_color='#2AA6FB', line_width=1.5)
   st.plotly_chart(fig, use_container_width=False, sharing="streamlit")

   
def Calculate_DCT(sig_data):
   
   N=1500
   t = np.linspace(0, 15,1500, False)
   y = dct(sig_data[:1500], norm='ortho')
   window = np.zeros(N)
   window[:20] = 1
   yf = idct(y*window, norm='ortho')   
   fig = px.line(x=t, y= yf, labels={'x':'Time', 'y':'DCT'},title='Discrete Cosine Transform', width = 1000, height = 600, markers=True)
   fig.update_traces(line_width=1.5)
   st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
   
def Calculate_DST(sig_data):
   
   N=1500
   t = np.linspace(0, 15,1500, False)
   y = dst(sig_data[:1500], norm='ortho')
   window = np.zeros(N)
   window[:20] = 1
   yf = idst(y*window, norm='ortho')   
   fig = px.line(x=t, y= yf, labels={'x':'Time', 'y':'DST'},title='Discrete Sine Transform', width = 1000, height = 600, markers=True)
   fig.update_traces(line_width=1.5)
   st.plotly_chart(fig, use_container_width=False, sharing="streamlit")   
   

def Calculate_STFT2(sig_data):
   
     fs = 100
     f, t, Zxx = signal.stft(sig_data, fs)
     #fig = go.Figure(data=[go.Mesh3d(x=t, y=f, z=np.real(Zxx), color='red', opacity=0.50)])
     trace = [go.Heatmap(
      x= t,
      y= f,
      z= np.abs(Zxx),
      name = 'STFT',     
      colorscale = 'Hot',
	     
      )]
     layout = go.Layout(
     title = 'STFT',
     yaxis = dict(title = 'Frequency'), # x-axis label
     xaxis = dict(title = 'Time'), # y-axis label

     )
     fig = go.Figure(data=trace, layout=layout)
     #fig.update_traces(line_width=1.5)
     st.plotly_chart(fig, use_container_width=False, sharing="streamlit")

def Calculate_Phase_Spectrum(sig_data):
   
     fs = 100
     f, t, Sxx = signal.spectrogram(sig_data, fs)
     #fig = go.Figure(data=[go.Mesh3d(x=t, y=f, z=np.real(Zxx), color='red', opacity=0.50)])
     trace = [go.Heatmap(
     x= t,
     y= f,
     z= np.abs(Sxx),
     name = 'Spectrogram',     
     colorscale = 'Hot',
	     
      )]
     layout = go.Layout(
     title = 'Spectrogram',
     yaxis = dict(title = 'Frequency'), # x-axis label
     xaxis = dict(title = 'Time'), # y-axis label

     )
     fig = go.Figure(data=trace, layout=layout)
     #fig.update_traces(line_width=1.5)
     st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
  
  
def Calculate_FFT(sig_data):
   N = 1500
   yf = rfft(sig_data[:1500])
   xf = rfftfreq(N, 0.01)
   #yf = yf[:60000]
   fig = px.line(x=xf, y=np.abs(yf), labels={'x':'Frequency(Hz)', 'y':'Amplitude'},title='Fourier Transform', width = 1000, height = 600, markers=True)
   fig.update_traces(line_width=1.5)
   fig.update_layout(yaxis_range=[0,120000])
   fig.update_layout(xaxis_range=[0,20])
   st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
   
   



#Streamlit GUI starts from here
st.set_page_config(
	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="collapsed",  # Can be "auto", "expanded", "collapsed"
	page_title=None,  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)

a=st.sidebar.radio('Navigation',['Farm Information','Farmer Data'])
# df = pd.read_csv("Trebirth.csv")

if a == "Farm Information":
 st.header("Welcome to Trebirth Tech Development")

uploaded_files = st.file_uploader("Choose a file",accept_multiple_files=True)
if uploaded_files is not None:
	st.write((len(uploaded_files))+"Files uploaded")
	#for i in range(len(uploaded_file)):
	
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
 # col1, col2= st.columns(2)
 #
 # with col1:
 #     st.header("Filtered Data")
 #     st.line_chart(Filtered_data)

 # with col2:
 #     st.header(" Accelerometer")
 #     st.line_chart(Filtered_data)




