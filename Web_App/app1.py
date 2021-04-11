import streamlit as st
import numpy as np 
import pandas as pd 
import evalml
import woodwork as ww
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from featuretools.selection import remove_low_information_features, remove_highly_null_features, remove_single_value_features, remove_highly_correlated_features


def main():

    st.title("DataHacks 2021")
    
    st.write("## Predict any country's pillar and prosperity scores!")
    
    st.write("#### Using our machine learning models, predict any country's prosperity score")
    
    countries_df = pd.read_csv('https://raw.githubusercontent.com/Andrewl7127/UCSD-DataHacks-2021/main/Data/merged.csv')
    countries_df = list(countries_df['country'].unique())    
    
    names = ['busi', 'econ', 'educ', 'envi', 'gove', 'heal', 'pers', 'safe', 'soci']

    def pillar(name = 'busi', countries = ['Chad']):
    
        url = 'https://raw.githubusercontent.com/Andrewl7127/UCSD-DataHacks-2021/main/Data/'
        df = pd.read_csv(url+name+'_train.csv')
        df = df.drop(['Unnamed: 0'], axis = 1)
    
        for i in df.columns:
            if i.find('year') > -1:
                df = df.drop([i], axis = 1)
    
        y = df[name]
    
        df = df.drop(['rank_'+name, name], axis = 1)
    
        df = remove_low_information_features(df)
    
        df = remove_highly_null_features(df)
    
        df = remove_single_value_features(df)
    
        df = remove_highly_correlated_features(df)
    
        X = df
    
        problem_type = 'regression'
        objective =  'auto'
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify = X['country'])
    
    
        automl = evalml.automl.AutoMLSearch(X_train, y_train, problem_type=problem_type, objective = objective)
        
   
        best_pipeline = automl.load(name+'_best_pipeline')
    
        df = pd.read_csv(url+name+'_test.csv')
        df = df.drop(['Unnamed: 0'], axis = 1)
    
        for i in df.columns:
            if i.find('year') > -1:
                df = df.drop([i], axis = 1)
    
        df = remove_low_information_features(df)
    
        df = remove_highly_null_features(df)
    
        df = remove_single_value_features(df)
    
        df = remove_highly_correlated_features(df)
    
        predictions = best_pipeline.predict(df)
    
        predictions = predictions.to_series()
    
        result = pd.DataFrame()
    
        result[name] = predictions
    
        df = pd.read_csv(url+name+'_test.csv')
        temp = df[['country', 'year']]
    
        result = pd.merge(left = temp, right = result, how="left", on=[temp.index, result.index])
        result = result.drop(['key_0', 'key_1'], axis = 1)
    
        result['rank_'+name] = result.groupby("year")[name].rank("dense", ascending=False)
        result['rank_'+name] = result['rank_'+name].astype('int')
    
        result = result[result['country'].isin(countries)]
        
        return result
    
    for name in names:
        print(pillar(name))
    
    country_sel = st.multiselect("Select which countries you want to learn more about!", countries_df)
    name = st.selectbox('Score', names)
    
    if st.button("Submit"):
        st.balloons()
        st.write(pillar(name, list(country_sel)))
  
   
                 
    
