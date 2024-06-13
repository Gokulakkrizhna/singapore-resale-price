from Datacoll_dataclean import a,b,c,v1,v2,v3,v4,v5,v6,v7,v8,v9,V1,V2,V3,V4,V5,V6,V7,V8,V9
import streamlit as st
import pandas as pd
import numpy as np
import pickle


def streamlit():
    st.markdown('<h1 style="text-align: center; color: red;">Singapore Resale Flat Prices</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        town = st.selectbox("Select the Town",v1)
        t1 = v1.index(town)
        town = V1[t1]
        flat_ty = st.selectbox("Select the Flat type",v2)
        f1 = v2.index(flat_ty)
        flat_ty = V2[f1]
        block = st.selectbox("Select the block",v3)
        b1 = v3.index(block)
        block = V3[b1]
        street = st.selectbox("Select the Street name",v4)
        S1 = v4.index(street)
        street = V4[S1]
        storey = st.selectbox("Select the Storey range",v5)
        S2 = v5.index(storey)
        storey = V5[S2]
    with col2:
        floor = st.number_input("Enter the floor area in sqm {0} and {1}".format(b[0],c[0]), value=b[0], placeholder="Type a number...")
        flat_ml = st.selectbox("Select the flat model",v6)
        f2 = v6.index(flat_ml)
        flat_ml = V6[f2]
        lease = st.selectbox("Select the lease commence date",v7)
        l1 = v7.index(lease)
        lease = V7[l1]
        trans_yr = st.selectbox("Select the trans year",v8)
        t1 = v8.index(trans_yr)
        trans_yr = V8[t1]
        yrs_rm = st.selectbox("Select the years remaining",v9)
        y1 = v9.index(yrs_rm)
        yrs_rm = V9[y1]
    
    if st.button("Predict Resale Price"):
            with open('ar.pkl', 'rb') as file:
                model = pickle.load(file)
            
            y_pred = model.predict([[town,flat_ty,block,street,storey,floor,flat_ml,lease,trans_yr,yrs_rm]])
            pred_val = 10**(y_pred)

            st.subheader(":red[Predicted Resale price:] {0}".format(pred_val[0]))


streamlit()