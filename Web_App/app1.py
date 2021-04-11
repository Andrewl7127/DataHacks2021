import streamlit as st
import pandas as pd 
import evalml
import woodwork as ww
from featuretools.selection import remove_low_information_features, remove_highly_null_features, remove_single_value_features, remove_highly_correlated_features


def main():

    st.title("DataHacks 2021")
    
    st.write("## Predict any country's pillar and prosperity scores!")
    
    st.write("#### Using our machine learning models, predict any country's prosperity score")
    
    countries_df = pd.read_csv('https://raw.githubusercontent.com/Andrewl7127/UCSD-DataHacks-2021/main/Data/merged.csv')
    countries_df = list(countries_df['country'].unique())    
    
    names = ['busi', 'econ', 'educ', 'envi', 'gove', 'heal', 'pers', 'safe', 'soci', 'prosperity']

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
        
        automl = evalml.automl.AutoMLSearch(problem_type=problem_type, objective = objective)
        
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
    
        result = pd.DataFrame()
    
        result[name] = predictions
    
        df = pd.read_csv(url+name+'_test.csv')
        temp = df[['country', 'year']]
    
        result = pd.merge(left = temp, right = result, how="left", on=[temp.index, result.index])
        result = result.drop(['key_0', 'key_1'], axis = 1)
    
        result['rank_'+name] = result.groupby("year")[name].rank("dense", ascending=False)
        result['rank_'+name] = result['rank_'+name].astype('int')
    
        result = result[result['country'].isin(countries)]
        metric = pd.read_csv('https://raw.githubusercontent.com/Andrewl7127/UCSD-DataHacks-2021/main/Metrics/'+name+'_metrics.csv')
        
        return result, metric
    
    def prosperity(countries = ['Chad', 'Togo', 'Zimbabwe', 'Ivory Coast', 'Georgia']):
    
        url = 'https://raw.githubusercontent.com/Andrewl7127/UCSD-DataHacks-2021/main/Data/'
        df = pd.read_csv(url+'merged.csv')
        df = df.drop(['Unnamed: 0'], axis = 1)
    
        metrics = ['educ', 'soci', 'heal', 'pers', 'busi', 'econ', 'safe', 'gove', 'envi']
        ranks = ['rank_' + metric for metric in metrics]
        drop = metrics + ranks + ['year', 'prosperity_score']
    
        y = df['prosperity_score']

        df = df.drop(drop, axis = 1)

        df = remove_low_information_features(df)
    
        df = remove_highly_null_features(df)
    
        df = remove_single_value_features(df)
    
        df = remove_highly_correlated_features(df)
    
        X = df
        
        problem_type = 'regression'
        objective =  'auto'
    
        
        automl = evalml.automl.AutoMLSearch(problem_type=problem_type, objective = objective)
        
        #automl.search(X,y)
        #best_pipeline = automl.best_pipeline
        #best_pipeline.fit(X,y)
        #best_pipeline.save('prosperity_best_pipeline')
        
        best_pipeline = automl.load('prosperity_best_pipeline')
        
        test = pd.read_csv(url+'test.csv', index_col = 0)
        
        drop = ['year']
        df = test.copy()
        df = df.drop(drop, axis = 1)
    
        df = remove_low_information_features(df)
    
        df = remove_highly_null_features(df)
    
        df = remove_single_value_features(df)
    
        df = remove_highly_correlated_features(df)
    
        X = df
        
        predictions = best_pipeline.predict(X)
        
        result = pd.DataFrame()
    
        result['prosperity'] = predictions
        
        df = pd.read_csv(url + 'test.csv')
        temp = df[['country', 'year']]
    
        result = pd.merge(left = temp, right = result, how="left", on=[temp.index, result.index])
        result = result.drop(['key_0', 'key_1'], axis = 1)
    
        result['rank_prosperity'] = result.groupby("year")["prosperity"].rank("dense", ascending=False)
        result['rank_prosperity'] = result['rank_prosperity'].astype('int')
    
        result = result[result['country'].isin(countries)]
        
        metric = pd.read_csv('https://raw.githubusercontent.com/Andrewl7127/UCSD-DataHacks-2021/main/Metrics/prosperity_metrics.csv')
        
        return result, metric
    
    
    country_sel = st.multiselect("Select which countries you want to learn more about!", countries_df)
    name = st.selectbox('Score', names)
    
    if st.button("Submit"):
        st.balloons()
        if len(list(country_sel)) < 1:
            country_sel = ['Chad', 'Togo', 'Zimbabwe', 'Ivory Coast', 'Georgia']
        
        if name == 'prosperity':
            r, m = prosperity(list(country_sel))
        else:
            r, m = pillar(name, list(country_sel))
        
        st.write(r)
        st.write(m)
  
   
                 
    
