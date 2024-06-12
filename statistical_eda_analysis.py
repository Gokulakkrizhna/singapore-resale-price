from Datacoll_dataclean import a,b,c,d,e,f,g
import pandas as pd
from scipy.stats import spearmanr
import scipy.stats as stats


def statistical_analysis(df,b,c,d,e,f,g):
    correlation_matrix = df.corr()

    a_dict = {"Category":['floor_area_sqm', 'resale_price'],
              "Min":b,"Max":c,"Mean":d,"Median":e,"Mode":f,"Standard Deviation":g}

    df1 = pd.DataFrame(a_dict)

    return correlation_matrix,df1


def eda_analysis(df):
    price_by_town = {}
    for town_category in df['town'].unique():
        key = town_category
        price_by_town[key] = df[(df['town'] == town_category)]['resale_price']
    statistic,p_value = stats.kruskal(*price_by_town.values())

    price_by_flat_type = {}
    for flat_type_category in df['flat_type'].unique():
        key = flat_type_category
        price_by_flat_type[key] = df[(df['flat_type'] == flat_type_category)]['resale_price']
    statistic,p_value1 = stats.kruskal(*price_by_flat_type.values())

    price_by_block = {}
    for block_category in df['block'].unique():
        key = block_category
        price_by_block[key] = df[(df['block'] == block_category)]['resale_price']
    statistic,p_value2 = stats.kruskal(*price_by_block.values())

    price_by_street_name = {}
    for street_category in df['street_name'].unique():
        key = street_category
        price_by_street_name[key] = df[(df['street_name'] == street_category)]['resale_price']
    statistic,p_value3 = stats.kruskal(*price_by_street_name.values())

    price_by_storey_range = {}
    for storey_category in df['storey_range'].unique():
        key = storey_category
        price_by_storey_range[key] = df[(df['storey_range'] == storey_category)]['resale_price']
    statistic,p_value4 = stats.kruskal(*price_by_storey_range.values())

    correlation, p_value5 = spearmanr(df['floor_area_sqm'], df['resale_price'])

    price_by_flat_model = {}
    for flat_model_category in df['flat_model'].unique():
        key = flat_model_category
        price_by_flat_model[key] = df[(df['flat_model'] == flat_model_category)]['resale_price']
    statistic,p_value6 = stats.kruskal(*price_by_flat_model.values())

    price_by_lease_commence_date = {}
    for lease_commence_date_category in df['lease_commence_date'].unique():
        key = lease_commence_date_category
        price_by_lease_commence_date[key] = df[(df['lease_commence_date'] == lease_commence_date_category)]['resale_price']
    statistic,p_value7 = stats.kruskal(*price_by_lease_commence_date.values())

    price_by_trans_year = {}
    for trans_year_category in df['trans_year'].unique():
        key = trans_year_category
        price_by_trans_year[key] = df[(df['trans_year'] == trans_year_category)]['resale_price']
    statistic,p_value8 = stats.kruskal(*price_by_trans_year.values())

    price_by_years_remaining = {}
    for years_remaining_category in df['years_remaining'].unique():
        key = years_remaining_category
        price_by_years_remaining[key] = df[(df['years_remaining'] == years_remaining_category)]['resale_price']
    statistic,p_value9 = stats.kruskal(*price_by_years_remaining.values())

    return p_value,p_value1,p_value2,p_value3,p_value4,p_value5,p_value6,p_value7,p_value8,p_value9

s1,s2 = statistical_analysis(a,b,c,d,e,f,g)
e1,e2,e3,e4,e5,e6,e7,e8,e9,e10 = eda_analysis(a)