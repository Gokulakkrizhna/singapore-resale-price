import pandas as pd
import category_encoders as ce
import numpy as np
import warnings


def data_collection():
    df1 = pd.read_csv('large_file_part1.csv')
    df2 = pd.read_csv('large_file_part2.csv')
    df3 = pd.read_csv('2.csv')
    df4 = pd.read_csv('3.csv')
    df5 = pd.read_csv('4.csv')
    df6 = pd.read_csv('5.csv')
    df =  pd.concat([df1,df2,df3,df4,df5,df6],ignore_index=True)

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


def data_extract(df):
    value_counts = df["town"].value_counts()
    values_list1 = value_counts.index.tolist()

    value_counts = df["flat_type"].value_counts()
    values_list2 = value_counts.index.tolist()


    value_counts = df["block"].value_counts()
    values_list3 = value_counts.index.tolist()

    value_counts = df["street_name"].value_counts()
    values_list4 = value_counts.index.tolist()

    value_counts = df["storey_range"].value_counts()
    values_list5 = value_counts.index.tolist()

    value_counts = df["flat_model"].value_counts()
    values_list6 = value_counts.index.tolist()

    value_counts = df["lease_commence_date"].value_counts()
    values_list7 = value_counts.index.tolist()

    value_counts = df["trans_year"].value_counts()
    values_list8 = value_counts.index.tolist()

    value_counts = df["years_remaining"].value_counts()
    values_list9 = value_counts.index.tolist()

    return values_list1,values_list2,values_list3,values_list4,values_list5,values_list6,values_list7,values_list8,values_list9


def data_cleaning(df):
    warnings.filterwarnings("ignore", category=FutureWarning, module="category_encoders")

    df["trans_year"] = pd.to_datetime(df['month']).dt.year

    df['years_remaining'] = 99 - (df['trans_year'] - df['lease_commence_date'])
    df['years_remaining'] = df['years_remaining'].apply(lambda x: min(x, 99))

    df.drop(columns=["month","remaining_lease"],inplace=True)

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
    for i in ['floor_area_sqm', 'resale_price','lease_commence_date','trans_year','years_remaining']:
        df_min.append(min(df[i]))

    df_max = []
    for i in ['floor_area_sqm', 'resale_price','lease_commence_date','trans_year','years_remaining']:
        df_max.append(max(df[i]))

    df_mean = []
    for i in ['floor_area_sqm', 'resale_price','lease_commence_date','trans_year','years_remaining']:
        df_mean.append(int(df[i].mean())) 

    df_median = []
    for i in ['floor_area_sqm', 'resale_price','lease_commence_date','trans_year','years_remaining']:
        df_median.append(df[i].median()) 

    df_mode = []
    for i in ['floor_area_sqm', 'resale_price','lease_commence_date','trans_year','years_remaining']:
        df_mode.append(df[i].mode()[0]) 
    
    df_stddev = []
    for i in ['floor_area_sqm', 'resale_price','lease_commence_date','trans_year','years_remaining']:
        df_stddev.append(df[i].std())
    
    v1,v2,v3,v4,v5,v6,v7,v8,v9 = data_extract(df)

    df1 = df

    encoder = ce.OrdinalEncoder(cols=[ 'town', 'flat_type','block','street_name','storey_range','flat_model','lease_commence_date','trans_year','years_remaining'])
    df = encoder.fit_transform(df)

    V1,V2,V3,V4,V5,V6,V7,V8,V9 = data_extract(df)

    df["resale_price"] = np.log10(df["resale_price"])

    return df,df_min,df_max,df_mean,df_median,df_mode,df_stddev,v1,v2,v3,v4,v5,v6,v7,v8,v9,V1,V2,V3,V4,V5,V6,V7,V8,V9,df1


a = data_collection()
a,b,c,d,e,f,g,v1,v2,v3,v4,v5,v6,v7,v8,v9,V1,V2,V3,V4,V5,V6,V7,V8,V9,df1 = data_cleaning(a)
