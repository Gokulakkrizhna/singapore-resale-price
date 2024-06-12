import pandas as pd
import category_encoders as ce
import numpy as np
import warnings


def data_collection():
    df = pd.read_csv("data.csv")
    return df


def outlier_handling(df):
    upper_limit = df['floor_area_sqm'].mean() + 3*df['floor_area_sqm'].std()
    lower_limit = df['floor_area_sqm'].mean() - 3*df['floor_area_sqm'].std()
    df['floor_area_sqm'] = np.where(df['floor_area_sqm']>upper_limit,upper_limit,df['floor_area_sqm'])
    df['floor_area_sqm'] = np.where(df['floor_area_sqm']<lower_limit,lower_limit,df['floor_area_sqm'])

    upper_limit = df['resale_price'].mean() + 3*df['resale_price'].std()
    lower_limit = df['resale_price'].mean() - 3*df['resale_price'].std()
    df['resale_price'] = np.where(df['resale_price']>upper_limit,upper_limit,df['resale_price'])
    df['resale_price'] = np.where(df['resale_price']<lower_limit,lower_limit,df['resale_price'])

    return df


def data_cleaning(df):
    warnings.filterwarnings("ignore", category=FutureWarning, module="category_encoders")

    df.drop('Unnamed: 0', axis = 1, inplace = True)

    df["trans_year"] = pd.to_datetime(df['month']).dt.year

    df['years_remaining'] = 99 - (df['trans_year'] - df['lease_commence_date'])
    df['years_remaining'] = df['years_remaining'].apply(lambda x: min(x, 99))

    df.drop(columns=["month"],inplace=True)

    mappings = {'MULTI-GENERATION': 'MULTI GENERATION'}
    df['flat_type'] = df['flat_type'].replace(mappings)

    df['flat_model'] = df['flat_model'].str.lower()
    mappings = {'improved': 'Improved',
                'new generation': 'New Generation',
                'model a': 'Model A',
                'standard': 'Standard',
                'simplified': 'Simplified',
                'model a-maisonette': 'Model A-Maisonette',
                'apartment': 'Apartment',
                'maisonette': 'Maisonette',
                'terrace': 'Terrace',
                '2-room': '2 Room',
                '2 room': '2 Room',
                'improved-maisonette': 'Improved Maisonette',
                'multi generation': 'Multi Generation',
                'premium apartment': 'Premium Apartment',
                'adjoined flat': 'Adjoined Flat',
                'premium maisonette': 'Premium Maisonette',
                'model a2': 'Model A2',
                'type s1': 'Type S1',
                'type s2': 'Type S2',
                'dbss': 'Dbss',
                'premium apartment loft': 'Premium Apartment Loft',
                '3gen': '3-Gen'}
    df['flat_model'] = df['flat_model'].replace(mappings)

    df = outlier_handling(df)

    df_min = []
    for i in ['floor_area_sqm', 'resale_price']:
        df_min.append(min(df[i]))

    df_max = []
    for i in ['floor_area_sqm', 'resale_price']:
        df_max.append(max(df[i]))

    df_mean = []
    for i in ['floor_area_sqm', 'resale_price']:
        df_mean.append(int(df[i].mean())) 

    df_median = []
    for i in ['floor_area_sqm', 'resale_price']:
        df_median.append(df[i].median()) 

    df_mode = []
    for i in ['floor_area_sqm', 'resale_price']:
        df_mode.append(df[i].mode()[0]) 
    
    df_stddev = []
    for i in ['floor_area_sqm', 'resale_price']:
        df_stddev.append(df[i].std())

    encoder = ce.OrdinalEncoder(cols=[ 'town', 'flat_type','block','street_name','storey_range','flat_model','lease_commence_date','trans_year','years_remaining'])
    df = encoder.fit_transform(df)

    df["resale_price"] = np.log10(df["resale_price"])

    return df,df_min,df_max,df_mean,df_median,df_mode,df_stddev


a = data_collection()
a,b,c,d,e,f,g = data_cleaning(a)