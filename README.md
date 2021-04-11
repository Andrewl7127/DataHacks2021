# DataHacks 2021: Intermediate Track - Economics

## By Andrew Liu, Adhvaith Vijay, Shail Mirpuri, Youngseo Do

## Key Links
[GitHub Repository](https://github.com/Andrewl7127/UCSD-DataHacks-2021)

[Streamlit Web Application ](https://datahacks2.herokuapp.com/)

[Visualization Dashboard ](http://datahacks-2021.herokuapp.com/)

## a. Introduction to Data

The data consists of 18 datasets, one train and one test for each pillar. The 9 pillar scores combine to create the prosperity score of a country through a simple average. These datasets are ranked and ordered by year and country. The training sets include the years 2007 to 2014 while the testing sets contain information from 2015 to 2016. Each training set has 1192 entries and each testing set has 298 entries. Both training and testing sets have 10+ informative features. All 18 datasets have no null values for any columns. Each of the nine datasets contain the rank_PILLAR_NAME which shows the rank of each country in that pillar, as well as the score index indicated by the column PILLAR_NAME. We later merged all of the datasets into one big dataset in order to derive insights and visualizations from the overall prosperity scores.

## b. Data Cleaning

One of the first tasks that we did was explore the data. We noticed that for all of the training and testing datasets, there were no missing values. However, there were a couple columns that we deemed unnecessary. First, we removed all columns containing the word “year.” We did this because our ultimate goal was to predict data on the 2015 and 2016 years, which are future years that are unseen in the training set. We then utilized an automated feature engineering library called FeatureTools to remove low information, highly null, single value, or highly correlated features. These all have their own thresholds and definitions in the library. For example, a highly null feature is defined as a feature with 95% or higher null values by default. For the most part, the library only removed single value columns such as start and ahead, which often only had one unique value.

## c. Data Visualization

In order to best represent our five visualizations, we created a separate heroku app to allow for different levels of interactivity from the user ([Visualization Dashboard](http://datahacks-2021.herokuapp.com/)).

### Visualization #1 (Top 5 Countries By Prosperity Score)

In order to determine the top 5 nations with the most growth in prosperity overall, we looked at calculating percent change. Since we wanted an overall trend, we investigated the percent change between 2014 (the most recent year in our labeled data) and 2007 (the earliest year in our labeled data). After data-preprocessing through Pandas, we found that the countries that made the most overall progress were Chad, Togo, Zimbabwe, Ivory Coast, and Georgia. In order to represent this data against time, we also looked at the percent change (since 2007) in each of the top 5 countries for every year in 2007-2014. This gives us a general sense of the overall growth that has occurred in the 7-year span between 2007 and 2014.

### Visualization #2 (Correlation Matrix Heatmap)

In our second visualization, we aimed to recognize how different variables are correlated to one another - chiefly the correlation coefficient between the percent change of different pillars and the percent change in prosperity score. To achieve this, we created 9 new columns (based on the 9 pillar scores) and looked at the percent change in each pillar between 2007 and 2014. The reason for this is to look at what percent change in pillars is most most correlated with percent change in prosperity. We found that percent change in gove (Governance) is most correlated with percent change in prosperity. As such, gove is likely to have the most significant impact on predicting country prosperity in the future. We explore this relationship in greater detail in part d.

### Visualization #3 (3D Scatterplot: Pillars vs. Prosperity)

From the correlation matrix heatmap above, we recognized that two pillars that are most correlated with change in prosperity are gove (Governance) and busi (Business Environment). Using interactive visualization libraries, we created a 3D scatterplot to investigate what trends exist across the gove and busi pillars and prosperity score across the top 5 countries with the most overall growth. The 3D Scatterplot can be rotated and if the ‘Play’ button is clicked a simple animation displays a 180 degree view of our figure. Interestingly, across the top 5 nations with the most overall prosperity growth, there appears to be a direct relationship between gove, busi, and prosperity score across each axis.

### Visualization #4 (Prosperity % Change (2007-2014))

In order to validate our claims made concerning the top 5 countries we looked towards mapping percent change in prosperity score across all nations. This led us to create a global choropleth, with a focus on the percent change since 2007. Right from the start many nations in Africa distinguish themselves with Chad and Zimbabwe benign frontrunners. As time goes on, there is a clear trend with many African nations - namely in Central/Western Africa (i.e. Toga, Ivory Coast) - slowly becoming more and more prosperous relative to 2007. 

### Visualization #5 (User-Generated Pillar/Prosperity Scores)

Next, we wanted to give users the opportunity to develop visualizations that interest them. We looked at all the raw pillar and prosperity scores in the year 2014 and allow users to select from a dropdown to visualize metrics of interests. The default map (Prosperity Score), shows that much of the Western World, including Europe and Australia, seems to have higher overall prosperity compared to the rest of the world. Asia and South America follow suit and on average possess a higher prosperity score. Interestingly, Japan, South Korea, and Malaysia are among the only Asian nations that have a prosperity score comparable to the Western World. The Southern tip of Africa fares far better than the rest of the continent - likely attributable to its general stability and affinity towards outside investment in recent years.

## d. Machine Learning Methods (Pillars)

Our machine learning workflow revolved around the Evalml library by Alteryx. Evalml is an AutoML library that builds, optimizes, and evaluates machine learning pipelines. It incorporates other automation libraries such as FeatureTools and Compose to create end-to-end supervised machine learning solutions. For our pillar models, after cleaning and preprocessing the labeled data (2007-2014), we separated the target variable from the feature variables, removing scores and ranks from the training set, and split the labeled data into 75% for training and 25% for testing, stratifying by country to ensure that we had a proportional representation of countries between our testing and training sets. We then optimized for R2 in a regression pipeline, which ultimately resulted in XGBoost Regressor w/ Imputer + One Hot Encoder as our best model for almost all of the pillars. Afterward, we observed the performance (R2, MSE, RMSE, MAE) of the model on the holdout set, the 25% that we split off earlier, and feature importance before using the model to predict the pillar scores and ranks for the unlabeled data (2015-2016). We repeated this process for each pillar. 

For the overall prosperity model, we followed the same process, the only difference being that we needed to merge the pillar data into one big dataset with all the features beforehand and that we predicted for overall prosperity ranks and scores as opposed to pillar ranks and scores. The best pipeline for the overall prosperity score was also XGBoost Regressor w/ Imputer + One Hot Encoder. To dive deeper into the most influential categories for each pillar, we referred to the feature importance returned to us by our models.

We noticed that a majority of the most influential categories were numeric, which is most likely due to the numeric features in these datasets holding a more granular value. The top 5 most influential categories for each pillar can be found here: [Top 5 Most Influential Categories](https://github.com/Andrewl7127/UCSD-DataHacks-2021/tree/main/Feature-Importance). These categories were instrumental in predicting the scores and ranks for the top 5 countries for 2015-2016 after we performed the same cleaning and data pre-processing steps on the unseen 2015-2016 data as described above. The predicted pillar scores and ranks for the top 5 countries for 2015-2016 can be found here: [Pillar Predictions](https://github.com/Andrewl7127/UCSD-DataHacks-2021/blob/main/Predictions/merged_predictions.csv). The evaluation metrics also turned out very well when testing our model on our holdout set, with most models returning a R2 over .98 and an RMSE of around 1. The evaluation metrics each model performed on the holdout set can be found here: [Pillar Metrics](https://github.com/Andrewl7127/UCSD-DataHacks-2021/blob/main/Metrics/merged_metrics.csv).

## f. Machine Learning Methods (Prosperity)

In order to investigate the pillars that were most influential in predicting a country's overall prosperity score, we used the feature importance of each best pipeline in our auto ML model. For each of the top 5 countries, we created an individual model to predict the prosperity score for that specific country. For example, for Chad, we created a model that solely predicted the prosperity scores for Chad using the features in our merged dataframe. Through the use of AutoML and due to the nature of decision-tree-based ensemble machine learning algorithms, our pipeline ignored or removed features that were redundant or unhelpful in predicting the prosperity score for the given country. After this, we grouped the features by the pillar they came from, aggregating by the sum of the feature importances. This allowed us to see which pillars tended to be the most significant for each country. The most influential pillars in predicting overall prosperity for each of our top 5 countries (Chad, Togo, Zimbabwe, Ivory Coast, and Georgia) can be found here: [Most Influential Pillars](https://github.com/Andrewl7127/UCSD-DataHacks-2021/tree/main/Feature-Importance-Country).


We decided to take this one step further by analyzing which features as a whole determine the overall prosperity for all countries. After grouping by the different data-frames each feature came from, we found that the Government related features tended to have the greatest impact on the prosperity score for any country. This suggests the importance and responsibility of the government in driving their country’s overall success. 

We also wanted to use our model to predict the prosperity scores and ranks for the top 5 countries. Using the AutoML pipeline described above, we split the labeled data (2007-2014) in a 75/25 ratio, stratified by country, and were able to perform feature selection and feature engineering on the labeled data (2007-2014) as well as model tuning and optimization to explain 99% of the total variation in the prosperity by the sub-category features when testing on the holdout set. We did not include the pillar scores or ranks in our model since this would be target leakage as when predicting future years we would not have access to these scores. Furthermore, since the prosperity score is a simple average of the pillar scores, by including these pillar scores as features in our model we would not be generating any useful insight. 

Using our model, we predicted the prosperity for our top 5 countries for 2015-2016 through the unlabeled dataset and ranked them appropriately. The predicted prosperity scores and ranks for the top 5 countries for 2015-2016 can be found here: Prosperity Predictions. The evaluation metrics also turned out quite well with a R2 score of .99+ and an RMSE and MSE under 1 when testing on the holdout set. The evaluation metrics the model performed on the holdout set can be found here: Prosperity Metrics.

We also wanted to allow users to take advantage of our ML models and investigate any country they may be interested in. To enable this, we created a web application that allows any user to input their countries and metrics (pillars of prosperity score) of interest and see the respective predictive scores as well as the model’s performance metrics on the holdout set so that they gauge the level of uncertainty within these predictions. Check out the web application, and have a go at using it: [Streamlit Web Application ](https://datahacks2.herokuapp.com/). Alternatively, here is a link to a demonstration of the web application and its uses: [Web App Demo](https://youtu.be/AszT5vd-_7Y). 


## g. Conclusion 

Overall, through data cleaning, pre-processing, and model building we have taken the 9 different pillars to generate informative insights about the prosperity of a country. Our visualizations have shown us the substantial growth in overall prosperity achieved by Chad, Togo, Zimbabwe, Ivory Coast, and Georgia between 2007 and 2014. We’ve dived deeper into these countries and developed interpretable machine learning models that tell us the pillars that provide the most insight into each country’s prosperity. One insightful finding we’ve uncovered is the large impact of the government on its respective country’s prosperity. We're proud that our model is accurate, interpretable, and insightful. Additionally, we’ve made these models and visuals accessible to anyone by creating interactive web applications for each. As a whole, by applying data science we’ve uncovered some fascinating insights that will undoubtedly allow us to predict any country's future prosperity scores with a high degree of accuracy.      



