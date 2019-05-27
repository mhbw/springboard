# Building Effective Fantasy Hockey Models: A Case Study

### Introduction

Fantasy sports is an exploding industry, with fantasy football alone estimated to be worth $7 billion in 2017. Models have existed in other sports to predict overall player performance since at least 2003, the most notable example being Baseball Prospectus’ PECOTA, but thus far NHL lacks a similar standard. The game is possibly the most difficult to predict of the major sports, and those who have developed such models have quickly been hired away by NHL teams, taking their models with them. 

To that end, this project will attempt to create a competitive replicable model for hobbyists in a standard ESPN annual fantasy hockey league, who currently buy preseason guides or fantasy subscriptions. 

For this case study I took the ESPN fantasy league that I play in as a guide, and built a number of models using various tools from python's scikit-learn packages, the objective being to build a model that will perform at least as well as a human in the current league.

### Table of Contents

1. [Data Acquisition and Wrangling](#Data_Wrangling)
  - [Cleaning](#Data_Cleaning)
  - [Feature Creation](#Feature_Creation)

2. [Data Exploration](#Data_Exploration)
  - [Global View](#Global_View)
  - [Correlations](#cor)
    
3. [Model Building](#Model_Building)

4. [Application: Comparing Model Performance to 2018 Draft](#Application)

5. [Conclusions and Future Work](#Extensions)
  - [Extensions](#Future_Work)
  - [Conclusions](#Conclusions)


### 1. Data Acquisition and Wrangling <a class="anchor" id="Data_Wrangling"></a>

   All data was collected from Natural Stat Trick an invaluable resource curated by Micah McCurdy, who compiles all kinds of data from the Nation Hockey League. I'm going to outline some of the process here, but you can read a full rundown of my process in this [notebook](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Data%20Wrangling:%20Notebook%201.ipynb).

   The only preprocessing I did was to edit some of the header rows in the csv, and added a year column, in order to have that preserved.

   I created the following as my naming conventions for the downloaded files:

**Year**: refers to the year the season completed PK: means a penalty kill unit PP: means powerplay S: means 5-on-5 or standard play 

**Counts**: are an individual occurrence Rates: are something over 60 minutes of time, the length of a standard game

#### Data Cleaning <a class="anchor" id="Data_Cleaning"></a>

   Given the specialty nature of this data set there are few na values; it wasn't as tidy as a Kaggle exercise but it was close. The notable exception is Draft information, as a number of players were undrafted. After consideration, I decided to make this a number higher than possible, 1000, which would make it stand out and be important for a linear regression, where as if I make it say, 0, have those players sitting by those drafted first (marked as 1 in the overall draft). A normal draft class has about 370 players, so I wanted this number to be higher, showing that they were ranked lower than the players who were drafted. This was mostly for naught, as in the end it appears to have little overall importance and I dropped the feature all together from some of the models. Draft Year I made into 1900, and team is now 'Undrafted'.

   I filled in other values using a set logic based on if the value was a count or a rate. If the value is a count or a percentage of a count, I entered it as zero. If it's a percentage of a rate, then I filled to median. Example of each:

1) **IPP is Individual Point Percentage**, the percentage of goals for that player's team while that player is on the ice that the player earned a point on. For this I filled the NA to 0, because logically if you have no points in this category your count would be zero.

2) **OffZoneStartPer is Offensive Zone Start Percentage**. this is a rate, and thus, simply hasn't been calculated since the player hasn't been given the opportunity to have such a stat created.

   In essence the logic here is "if you have a NA due to not getting any points, you are marked zero, if you have an NA due to not being on the ice for that metric, you are returned to the mean". I've heard some arguments that you might leave these as na since many of the sklearn models have the ability to deal with NA values and this unfairly 'bulks up' the mean or median, but again this was a fairly limited number of values in any given column (less than 1%) so mostly academic. 

   Finally, I found some pointless or redundant columns ('power play position', etc) which I dropped wholesale. 


#### Feature Creation <a class="anchor" id="Feature_Creation"></a>

   I created a number of features for these models, all of which were transformations intended to simplify or otherwise interpret existing columns with the exception of Age and Total Time on Ice. Age was a calculation based on birthdate, and is important because players performance tends to fall on what's called an 'age curve', where they peak in their late 20s and then trail off. Total Time On Ice matters because it's an important feature both in terms of other variables and as we'll see later is a key indicator of player performance.
   
   I also created the target variable 'Fantasy Points' which is the target for all the models and follows the scoring of the fantasy league, and some features around points and minutes per game, as well as points per 60 minutes. I used points per 60 since many in the hockey analytics community have argued that this is more valuable since it shows the rate at which a player performs and smooths out the curves, and I did per game scores thinking that would also show some more clarity; for example showing those that consistently have multi-point games instead of a flash in the pan with a few solid outings.
    

### 2. Data Exploration<a class="anchor" id="Data_Exploration"></a>

#### Global View <a class="anchor" id="Global_View"></a>

   Globally there are some trends and notacable curves that we can see at the outset. Most notably the points distribution had a bimodal distribution of points with a very long tail to the right. 
   
![scoredistro](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/scoredistro.png)

   I theorized that this was due to teams bringing up young players for trial basis to give them some experience or a try out but keep them under the 20 game cap that would qualify the player for their rookie season (hockey insider baseball: after a certain number of qualified seasons players have to be payed a higher amount, this type of movement lets a team use a player but not have this count as a qualified season and therefore pay the player less). For point of comparison this is the distribution if you were to only look at players above 20 games. 
   
 ![scoredistro20](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/scoredistro20pg.png) 
   
   As you can see it is now more unimodal, although it retains the long tail. I also did a quick boxplot of age, height, and weight. These are fairly normally distributed and not particularly remarkable.
   
 ![demos](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/demos.png) 
 
 #### Correlation <a class="anchor" id="cor"></a>
 
   I was very interested in what values correlated to fantasy points, as I thought those would be good for feature tuning, so i created a correlation matrix. 
   
    corr = player_data.corr(method='pearson')
    sb.heatmap(corr)
    sorted_fantasy_value = corr.iloc[:,-237].sort_values(ascending = False)
    sorted_fantasy_value.iloc[0:15,]

| Feature      | Correlation           |
| ------------- |:-------------:|
|Total Minutes Played      | 0.918|<br>
|TOI_SIndR             |     0.918|<br>
|iCF                    |    0.911|<br>
|iFF                     |   0.909|<br>
|Shots                    |  0.902|<br>
|TOI                 |       0.901|<br>
|Total_Assists        |      0.898|<br>
|GP                     |    0.869|<br>
|Fantasy Points Per Game |   0.834|<br>
|TOI_PPIndC               |  0.823|<br>
|PPTOI                     | 0.823|<br>
|iSCF                       |0.817|<br>
|Rebounds_Created     |      0.81|<br>
|Takeaways             |     0.808|<br>

At the bottom of the heap was

    sorted_fantasy_value.iloc[230:,]
| Feature      | Correlation           |
| ------------- |:-------------:|
|FA_60                   | -0.103|<br>
|CA_60                    |-0.113|<br>
|LDCA_60                  |-0.153|<br>
|PPOnTFStarts_60          |-0.187|<br>
|Draft_Round              |-0.207|<br>
|Round_Pick               |-0.225|<br>
|Overall_Draft_Position   |-0.232|<br>
|OnTFStarts_60            |-0.605|<br>

   There are two particularly interesting observations here; Corsi (iCF), Fenwick (iFF), and time on ice (metrics with TOI) all correlate better than values that I'd have guessed would be way more predictive, such as assists or goals, which are actually included in the metric. Corsi and Fenwick are called 'possession metrics', indicating the time a player has the puck, but are better understood as the amount of times a player is **shooting** the puck.  Also interesting is the negative correlation from Draft Position, which I'd have assumed would be similar in value to TOI. 
   
   Someplace in between is the values against players (GA_60 = Goals against the player over 60 min, SA_60 shots against the player's team, etc) also have a negative correlation. While I wouldn't have expected a massive link, negative is a bit surprising. My thesis on this (which will go unexplored here because it would be a much different project) is that players work harder if they're doing poorly defensively or playing from behind frequently.
   
   Finally Draft Position seems very low on correlation which fits with the first draft here, but is perhaps even lower than expected. Charted here are the drafted vs. undrafted players (drafted being the group marked '1'), as well as the correlation. As you can see there are wide bands on each, and 'drafted' actually has a very slight negative correlation to fantasy points.
   
 ![drafted2](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/drafted_undrafted.png) 
 
    sorted_fantasy_value.loc['drafted']
 -0.008
   
   I'd expected the draft metrics to actually have a fairly high correlation but that seems not to be the case. Here is the charting of overall draft position compared to points per season, with the undrafted players removed. As you can see, the regression line in the middle slopes downward very gently, and over all this is a loose grouping. 
   
   
 ![drafted3](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/draft_position.png)    
   
   I also broke out scores by position, expecting that this would have a strong mapping of scores. while there was a distinct grouping, I noticed two trends. 
   
   1) The data has a heavy rightward skew
   2) there was something of a bimodal distribution of the fantasy values
   
I chalked that up to two different effects from longevity and number of games played. If you play at a high value for your position you play for more years and more games, hence the tail. The distribution is the effect of the earlier 20 game rule. Here we see the positions without control for number of games played.

![score_per](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/score_distro1.png) 

![score_per](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/score_distro2.png) 

   
   In all would have liked to see more values that were perhaps causal instead of simply correlated. I did a deeper dive, splitting out data based on points per game and found essentially the same thing. 
   
   What's interesting to me here is that the values are approximately the same still (Corsi and Fenwick specifically sit at the same ranking), but the strength of correlation has dropped off, as has the value of scoring chances, rebounds and takeaways. Witness the top values side by side with respective correlation

| Feature      | Total Points           | Points Per Game  | Points Per Game Rank |
| ------------- |:-------------:| -----:| -----:|
| TOI_SIndR     |  0.919 | 0.69 | 6th |
| iCF      | 0.911     |   0.715 | 2nd |
| iFF | 0.909     |    0.714 | 3rd |
| TOI    |  0.901 | 0.662 | 9th |
| GP     | 0.869    |   0.579 | 20th |
| TOI_PPIndC  | 0.823     |    0.719 | 1st |
| iSCF      | 0.823 | 0.65 | 11th |
| Rebounds_Created      | 0.81     |   0.636 | 15th |
| Takeaways | 0.808     |    0.643 | 14th |

Here are the top ten correlated values directly for Points Per Game:

| Feature       | Correlation to Points Per Game         | 
| ------------- |:-------------:| 
TOI_PPIndC    |             0.719
iCF            |            0.715
iFF            |            0.714
TOI_GP         |            0.702
Total Minutes Played   |    0.69
TOI_SIndR             |     0.69
PPTOI_GP              |     0.683
iSCF_PPIndC            |    0.681
TOI                    |    0.662
iFF_PPIndC            |     0.657


   Interestingly, these values are now 6 out of 10 based around minutes played, where as with over all points it was 4 of 10. This was essentially the inverse of what I'd expected, since I thought that the Points Per Game feature would reduce the dependence on time.  

   What we see here from our quick analysis breaks down into three main points, the first two of which are not particularly surprising. Those primary points are that once you get past the set of players who will be cut, there is a remarkably normal distribution of points, and second that there is little benefit to creating additional features around points per game.

   The more interesting segment is how correlated time is to a players performance as we can see in the mapping of correlations above.  This is certainly a prime example of how correlation is not causation, but time on ice stands as the most important metric we see above, and will certainly play a role in our models in the next segment. For a more full and detailed version of these steps, as well as the code, you can see the [second notebook here](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Data%20Exploration%20and%20Story%20Telling:%20Notebook%202.ipynb).
   

### 3. Model Building <a class="anchor" id="Model_Building"></a>

   I went through three classes of models, starting with the classic linear regression, then more advanced ensemble models such as  Random Forests and Gradient Boosted Regressors. I also employed grid search to tune my models as I went. Again, this segment is mostly top lines and essentials, while code and detailed elements are in the [third notebook](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Model%20Building:%20Notebook%203.ipynb)

   I made an 80/20 split of training and testing data and began with the linear regression.

    model_data.set_index(keys=['Player','Team','Year','Position'],inplace=True)
    Target = model_data.iloc[:,0].values  
    features = model_data.iloc[:, 1:].values 
    X_train, X_test, y_train, y_test = train_test_split(features, Target, test_size=0.2, random_state=0)  
    
    ## linear regression
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()  
    lin_reg.fit(X_train, y_train) 
    
    ## looking at metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
    print('Regressor Score:',LinearRegression.score(lin_reg,X_test,  y_test))

Mean Absolute Error: 12.529<br>
Mean Squared Error: 280.046<br>
Root Mean Squared Error: 16.735<br>
Regressor Score: 0.954<br>
   I considered that an acceptable starting point, but not great. after all, if your mean total points are almost exactly 100, that makes for a deviation of 12.5%. From there I switched to ensemble based methods, starting with a Random Forest Regressor.
   
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
    regressor.fit(X_train, y_train) 

    y_pred = regressor.predict(X_test)  
    from sklearn import metrics

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
    print('Regressor Score:',RandomForestRegressor.score(regressor,X_test,  y_test))
    
   That did better, but not that much better and actually dropped off at MSE.

| Metric      | Linear Regression          | Random Forest  | 
| ------------- |:-------------:| -----:| 
| Mean Absolute Error:    |  12.5 | **11.75** | 
| Mean Squared Error:    |  **280.05** | 295.75 | 
| Root Mean Squared Error:     |  16.73 | **17.2** | 
| Regressor Score:    |  0.95 | 0.95 | 

   This is a fun glance at the important features for the Random Forest: 


 ![features1](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/features1.png) 
 
   Looking at features scores it's interesting that time again holds a lot of import. What I dislike here is that's not really something that we can then use as predictive value since that's a coaching decision, not a player skill. From here I turned to grid search, to see if I could improve the trees slightly.
 
         RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)


    param_grid = {"max_depth": [10, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              'n_estimators':[20,50,100],
              "bootstrap": [True, False]}

    # run grid search
    grid_search = GridSearchCV(regressor, param_grid=param_grid, cv=5)
    start = time.gmtime()
    grid_search.fit(X_train,  y_train)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time.mktime(time.gmtime()) - time.mktime(start), len(grid_search.cv_results_['params'])))
    grid_search.best_params_
    
    
  GridSearchCV took 365.00 seconds for 108 candidate parameter settings.
Out[10]:
{'bootstrap': False,<br>
 'max_depth': None,<br>
 'max_features': 10,<br>
 'min_samples_split': 2,<br>
 'n_estimators': 100}<br>
 
     best_grid = grid_search.best_estimator_
    y_pred = best_grid.predict(X_test) 

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
    print('Regressor Score:',RandomForestRegressor.score(best_grid,X_test,  y_test))

Mean Absolute Error: 11.198<br>
Mean Squared Error: 264.446<br>
Root Mean Squared Error: 16.262<br>
Regressor Score: 0.957<br>

   The grid search did improve some but not a ton across the board, other than in MSE, which makes sense as that's the metric it's scoring on.


| Metric      | Linear Regression          | Random Forest  | Grid Search  | 
| ------------- |:-------------:| -----:| -----:| 
| Mean Absolute Error:    |  12.5 | 11.75 | **11.2** | 
| Mean Squared Error:    |  280.05 | 295.75 | **264.45** | 
| Root Mean Squared Error:     |  16.73 | 17.2 | **16.26** | 
| Regressor Score:    |  0.95 | 0.95 | **0.96** | 

   I next attempted a tree model based on just the primary features, but that ended in disaster, which I won't bother recording here as all the errors increased, but I think it should be noted. I trimmed to the top 20 features instead of some +200 features thinking that would refine the model by reducing noise, but instead it increased the errors in every metric, which stands as perhaps a lesson in overfitting and definitely one in being too clever by half. 
   
   I then switched to gradient boosted regressors, and went through a number of iterations there, each of which had improvements.
   
    from sklearn import ensemble
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls', 'warm_start':'True'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf_pred = clf.fit(X_train, y_train).predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, clf_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, clf_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred))) 
    print('Regressor Score:',LinearRegression.score(clf,X_test,  clf_pred))
   
Mean Absolute Error: 10.127<br>
Mean Squared Error: 209.873<br>
Root Mean Squared Error: 14.488<br>
Regressor Score: 1.0<br>

   Here the features shifted very slightly while the errors decreased. This is the first model to outperform the cross-validated Random Forest Regressor, and perhaps as interesting the first to move to Fenwick up to the top three. Time factors sill dominate but not as much.  Also perhaps the first to have to have significant improvements across the board as far as errors.


| Metric      | Linear Regression          | Random Forest  | Grid Search  | Gradient Boost  |
| ------------- |:-------------:| -----:| -----:| -----:| 
| Mean Absolute Error:    |  12.5 | 11.75 | 11.2 |  **10.13** |
| Mean Squared Error:    |  280.05 | 295.75 | 264.45 |  **209.94** | 
| Root Mean Squared Error:     |  16.73 | 17.2 | 16.26 | **14.49** | 
| Regressor Score:    |  0.95 | 0.95 | 0.96 | **1.0** |

   From there I ran a couple of different grid searches, finally getting down to a error rate that was lower than 10 MAE. The later grids were even better, with the final looking like this:
   
   
    clfparam_grid2 = {"max_depth": [ None,3,  5],
              "max_features": [None, 10, 25],
              "min_samples_split": [2, 3, 10],
              'n_estimators':[500, 600] }


    clfgrid_search2 = GridSearchCV(clf, param_grid=clfparam_grid2, cv=5)
    start = time.gmtime()
    clf_grid2 = clfgrid_search2.fit(X_train,  y_train).predict(X_test)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time.mktime(time.gmtime()) - time.mktime(start), len(clfgrid_search2.cv_results_['params'])))
    clfgrid_search2.best_params_

 GridSearchCV took 34545.00 seconds for 54 candidate parameter settings. 'max_depth': 5, 'max_features': 25, 'min_samples_split': 2, 'n_estimators': 600}
   
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, clf_grid2))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, clf_grid2))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, clf_grid2))) 
    print('Regressor Score:',LinearRegression.score(clf,X_test,  clf_grid2))

Mean Absolute Error: 9.245<br>
Mean Squared Error: 177.463<br>
Root Mean Squared Error: 13.322<br>
Regressor Score: 0.997<br>
    
That makes this slightly better than the others are far as MAE but dramatically better in MSE and RMSE and if you round the regressor score it's a 1

| Metric      | Linear Regression         | Grid Search  | Gradient Boost  | Gradient Grid 1  | Gradient Grid 2 |
| ------------- |-------------:| -----:| -----:| -----:| -----:| 
| Mean Absolute Error:    |  12.53 | 11.2 | 10.13 | 10.26 | **9.24** |
| Mean Squared Error:    |  280.05 | 264.45 | 209.94 |  220.11 | **177.46** |
| Root Mean Squared Error:     |  16.73 | 16.26 |  14.49 | 14.84 | **13.32** |
| Regressor Score:    |  0.95 | 0.96 |  **1** | 0.99 |0.997|

   I used a package called [SHAP](https://github.com/slundberg/shap) to create a graphic of SHAP values, which assign a value to each feature that show how they effect the output of the model in a simplified way. These values represent 'a feature's responsibility for a change in the model output'. So for example, Time on ice can change a players point total by as much as 40 total points, -15 points in some cases, and as high as +25 for some players. Over all this is gives a tangable snapshot of how features impact the model's output.

 ![Shap](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/SHAP_value.png) 
   

   I think arguably you could run through deeper regressions with more estimators and get a closer and closer score, but for our purposes here there's no real need. Also, you could make a decent argument that the standard Gradient Boost works just as well for the effort as going through those iterations of grid searches. 

   That said, I'm happy here with the version that we found with the second grid search. Now we'll make this somewhat forward looking and see how teams in the actual fantasy league would have done with this model.


### 4. Application: Comparing Model Performance to 2018 Draft<a class="anchor" id="Application"></a>

   From here I wanted to give this project some real world context, so I thought it would be informative to see how accurately the model would have predicted this years' performance and how the current teams draft would have ranked by that scoring. To that end I imported the ESPN data from the league, and then projected the 2019 season performance.  The model actually lined up quite nicely here, although it was in fact slightly conservative.
   
 ![scores2019](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/model_perf2019.png) 

   That's a reasonably straight through line, and overall looks about as expected. So how would that track for our league? well pretty much the league would have been static. Only two teams would have seen a significant shift in position, which would have been Paul North Kariya and Datsyuk's Dekes. If the model were accurate and the teams had the same draft they both would have risen two positions in the standings. In fact, if you look at it side by side this way it appears even more static:

| **Team**     | **Fantasy Points**       | **Modeled_Fantasy_Points**  | **Movement**  |
| ------------- |-------------:| -----:| -----:| 
| **Gritty Gritty Bang Bang**    |  1 | 1 | 0 | 
| **Shat Deuces**    |  2 | 2 | 0 |  
| **Amesville Aliens**    |  3 | 4 |  -1 |
| **Beet Beet Gourde Gourde**    |  4 | 5 |  -1 |
| **Datsyuk's Dekes**    |  5 | 3 | +2 | 
| **Every Day I'm Byfuglien**    | 6 | 8 | -2 |  
| **Team Game Blouses**    |  7 | 6 |  +1 |
| **Tronhelm Black Bears**    |  8 | 9 |  -1 |
| **Paul North Kariya**    |  9 | 7 | +2 | 
| **Davinci's McDavid**   |  10 | 10 | 0 |  
| **London Bacon Blades**   |  11 | 11 |  0 |
| **Takoma Park Holtbeasts**   |  12 | 12 |  0 |

   Only 7 teams shifted, about half the league, and only 3 by more than one position. This isn't a huge surprise, but does speak to some accuracy since the margin in some of these was as slim as 4 points in the actual and 10 in the modeled, or less than 1/1000 of the top score. 

   Let's look at some players who did well, namely who we were most off on and who we were closest with.
   
 Closest margin:
 
  ![close](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/close.png) 

 Furthest:
 
 ![far](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/far.png) 


   What this shows me here is that the model over all does very well, especially at the mid point, and does poorly with the outliers. Those folks at the top are mostly the top of the league as well, in almost that exact order: Kucherov, Stamkos, Draisaitl, Ovechkin. As Micah McCurdy said once in a talk, 'we can look at the outliers and laugh at them, because that is not where the math is'. 
   
   The Standout was Mika Zibanejad, who ended up at 11th overall but was drafted at number 198 by the Takoma Park Holtbeasts. 

   Also, fun hockey trivia; many of those top players (Stamkos, Kucherov, Point) are on the best team in the league, who got swept in the first round of the playoffs that barely made it in. This data was all pulled two weeks prior, so I'd like to believe that these players simply regressed to where the model expected them. 
   
   Personally this was my favorite part of the project, and there are many many more details on the league and performance in the [league notbook](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Model%20Building:%20Notebook%203.ipynb), if you'd like to see more on specific rounds and players.
   
### 5. Conclusions and Future Work <a class="anchor" id="Extensions"></a>


#### Extensions<a class="anchor" id="Future_Work"></a>

   I think this model would benefit from a few things in future iterations, namely a reduction in time features, which caused those to rise to the top, and secondly a normalization and scaling of the features. The noise in the features could be reduced by normaliziation which would reduce the volatility.
   
   It would be valuable as well to add more data sources that might have metrics beyond the ones from Natural Stat Trick, to see if there might be more causal data out there, instead of relying on data generated based on coaching decisions. Additionally, sklearn's pipelines could refine this process.
   
   Finally it might make sense to use outside resources such as an Amazon AWS S3 instance to expand computational power and to see if further gains could be made from deeper trees or grid search. Recall that there were still improvements being made with deeper trees; while that gain was minimal in exchange for the time and resources on a local machine, it would be minimal cost to explore much deeper depth on a virtual host.
    

#### Conclusions <a class="anchor" id="Conclusions"></a>

   This type of model could be a valuable product as an introductory primer, and could be marketed to new players.  For example London Bacon Blades was managed by a new NHL fan who had only watched one season before this. The manager drafted two of the top over all performers but still dropped off ended at 11th in basically every metric. Had they had such a product, they would have perhaps been better able to make experienced moves like Takoma Park Holtbeasts; a former player, who as we noted above picked up a top overall player who he found at pick number 198.
     
   Most importantly, this capstone shows that while it is [most difficult to predict the outcome of a hockey game](https://www.vox.com/videos/2017/6/5/15740632/luck-skill-sports) it is possible to model player performance.  Not only did it get the league results mostly in line to their actual standings it did well at predicting the players, despite being somewhat conservative.  
   
   Both of these findings show that there is real value in the model and the product, and machine learning can lend tangible insights in a difficult field. 