from Datacoll_dataclean import a,b,c,v1,v2,v3,v4,v5,v6,v7,v8,v9,V1,V2,V3,V4,V5,V6,V7,V8,V9,df1
from statistical_eda_analysis import s1,s2,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import zipfile
import os
    

def streamlit():
    tab1, tab2, tab3, tab4= st.tabs(["Home", "Statistical Insights","EDA Insights","Visual Insights"])
    with tab1:
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
            floor = st.number_input("Enter the floor area(sqm) between {0} and {1}".format(b[0],c[0]), value=b[0], placeholder="Type a number...")
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
    
    with tab2:
        st.header(":red[Correlation using Heatmap]")

        fig = px.imshow(s1, text_auto=True, aspect="auto",color_continuous_scale="reds")
        fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Do you like to see statistical data"):
            st.dataframe(s2)

    with tab3:
        #Analysis1
        st.header(":red[Analysis of Town over Resale Price]")
        if e1 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection between different towns and their resale prices. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection between different towns and their resale prices. It is a strong evidence.           
            """)

        #Analysis2
        st.header(":red[Analysis of flat and resale price]")
        if e2 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection between flats and their resale price. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection between flats and their resale price. It is a strong evidence.           
            """)
        
        #Analysis3
        st.header(":red[Analysis for block and resale price]")
        if e3 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection between different blocks and their resale prices. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection between different blocks and their resale prices. It is a strong evidence.           
            """)

        #Analysis4
        st.header(":red[Analysis for Street Over price]")
        if e4 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection between different street and their resale prices. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection between different street and their resale prices. It is a strong evidence.           
            """)

        #Analysis5
        st.header(":red[Analysis for resale price over storey range]")
        if e5 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection between different storey range and their resale prices. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection between different storey range and their resale prices. It is a strong evidence.           
            """)
        
        #Analysis6
        st.header(":red[Analysis for Floor(sqm) over resale price]")
        if e6 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection between different floor(sqm) and their resale prices. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection between different floor(sqm) and their resale prices. It is a strong evidence.           
            """)

        #Analysis7
        st.header(":red[Analysis for flat model over resale price]")
        if e7 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection between different flat model and their resale prices. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection between different flat model and their resale prices. It is a strong evidence.           
            """)

        #Analysis8
        st.header(":red[Analysis for lease commence date over resale price]")
        if e8 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection between lease commence dates and their resale prices. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection between lease commence dates and their resale prices. It is a strong evidence.           
            """)

        #Analysis9
        st.header(":red[Analysis of Transaction year over resale price]")
        if e9 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection over each transaction year and their resale prices. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection over each transaction year and their resale prices. It is a strong evidence.           
            """)
            
        #Analysis10
        st.header(":red[Analysis of years remaining over resale price]")
        if e10 < 0.05:
            st.markdown("""
            - P-Value < 0.05
            - Reject H0
            - Dependent
            - From this we can conclude that there is a significant connection over number of years remaining in the lease period and their resale prices. It is a strong evidence.           
            """)
        else:
            st.markdown("""
            - P-Value > 0.05
            - Fail to Reject H0
            - Independent
            - From this we can conclude that there is no connection over number of years remaining in the lease period and their resale prices. It is a strong evidence.           
            """)

    with tab4:
        st.subheader(":red[Exploring Resale Price Trends Over Transaction years]")
        df2 = df1.groupby(["trans_year"])[["resale_price"]].sum().reset_index()
        fig = px.line(df2, x="trans_year", y="resale_price")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(":red[Exploring Resale Price Trends Over remaining Lease years]")
        df2 = df1.groupby(["years_remaining"])[["resale_price"]].sum().reset_index()
        fig = px.scatter(df2, x="years_remaining", y="resale_price")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(":red[Exploring Resale Price Trends Over lease commence Years]")
        df2 = df1.groupby(["lease_commence_date"])[["resale_price"]].sum().reset_index()
        fig = px.bar(df2, x="lease_commence_date", y="resale_price")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(":red[Exploring Resale Price Trends Over town]")
        df2 = df1.groupby(["town"])[["resale_price"]].sum().reset_index()
        fig = px.pie(df2, values='resale_price', names='town')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(":red[Exploring Resale Price Trends Over town and flats type]")
        df2 = df1.groupby(["town", "flat_type"])[["resale_price"]].sum().reset_index()
        fig = px.bar(df2, x="town", y="resale_price", color="flat_type")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(":red[Exploring Resale Price Trends Over storey range]")
        df2 = df1.groupby(["storey_range"])[["resale_price"]].sum().reset_index()
        fig = px.pie(df2, values='resale_price', names='storey_range',
                    hole=0.4, labels={'resale_price': 'Total Resale Price', 'storey_range': 'Storey Range'})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(":red[Exploring Resale Price Trends Over different flat model]")
        fig = px.box(df1, x="flat_model", y="resale_price",
             labels={"resale_price": "Resale Price", "flat_model": "Flat Model"})
        st.plotly_chart(fig, use_container_width=True)


streamlit()
