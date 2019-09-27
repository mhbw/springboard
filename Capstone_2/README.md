# Capstone 2: Applied Machine Learning For Daily Fantasy Hockey

### Introduction

By even the most conservative estimates daily fantasy sports is a multi-billion dollar industry, the largest volume of which is moved for fantasy football but the sites at this point cover everything down to e-sports and MMA.
 	
For this capstone project the intent was to cover if machine learning models could be trained to be sufficiently competitive for daily fantasy contests in hockey, a sport that is widely considered the most difficult to predict due to the level of variance, interaction among players, and general ‘puck luck’. 

### Table of Contents

1. [Data Acquisition and Wrangling](#Data_Wrangling)
  - [Collection](#Data_Collection)

2. [Data Exploration](#Data_Exploration)
  - [Point Distributions](#point_distro)
  - [Applied Market Forecasting Theories: Moving Average Convergence ](#cor)
    
3. [Model Building](#Model_Building)

4. [Application: FantasyCruncher](#Application)

5. [Conclusions](#Conclusions)
  - [Extensions](#Future_Work)
  - [Acknowledgments](#Acknowledgments)


### 1. Data Acquisition and Wrangling <a class="anchor" id="Data_Wrangling"></a>

   The data scraped from the NHL’s unofficial API, and then processed using scripts developed by the team at [EvolvingHockey](https://evolving-hockey.com), an invaluable resource for daily results. 

#### Data Collection <a class="anchor" id="Data_Collection"></a>

   For this project most of the data was in a raw format from the API and required compiling on a player-per-game instance. That was assisted by the Evolving Hockey scripts, but also required a PostgreSQL instance be stood up to house the data and easily access it, as well as create rank ordering functions.  Additionally, significant was done to break the api scraping down into chunks to reduce the load on the local machine. 


### 2. Data Exploration<a class="anchor" id="Data_Exploration"></a>

The Target Metric, Fantasy Points, are not evenly distributed as they are across a season. Below are points on a per-season basis (left) and on a per-game basis (right). This significantly complicates forecasting. 


![scoredistro](https://github.com/mhbw/springboard/blob/master/Capstone%201/Springboard%20Capstone%20Raw%20Data%20Sets/Images/scoredistro.png)

 
![dailyscoredistro](Capstone_2/images/scoredistro.png) 
   
  Time on ice (left) remains the constant for forecasting points, as  predictive and almost as correlated as core possession metric, Fenwick (right). 

   
 ![pbs](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/pointsbyshots.pneg) 
 
 ![sbs](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/pointsbytime.pneg) 
 
 
 #### Applied Market Forecasting Theories: Moving Average Convergence   <a class="anchor" id="cor"></a>
 
 	The 'Hot Hand Theory' might be the original piece of sports analytics work, and certainly the most discussed and dissected. Here we propose a slightly different take, which is to apply financial methods of finding trends to identify values. While there is significant value in a players past overall performance, but being able to spot immediate trends creates a near term arbitrage opportunity where players are undervalued but on an upswing.  
 
 	The theory in this case is that since time on ice and time on the power play are the biggest indicators of success, rivaled only by possession metrics, if a model could detect an upward trend it could find undervalued players in advance and forecast positive events.  
 	
 	While global movements are relatively rare, line shifts are frequent, sometimes happening in-game, and identifying when a player is seeing increased deployment over their previous shifts is highly valuable. A signal metric was developed to identify those events. 


	Here is a random sample of players with the positive (black arrow) and negative (red arrow) signal markers overlayed on top of the moving averages (blue and dark blue lines). Note how the black arrow frequently proceeds a distinct upward trend in the moving averages. 
	
 ![dh](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/DHMP_intial1.pneg) 
 
 ![mai](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/MA_intial1.pneg) 
 
That said the model can be ‘psyched’ out, as seem here, by highly uneven deployment which could be caused by a number of factors. 

 ![zk](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/ZPMP_intial1.pneg) 
   
Most promising, when used in an OLS model both signals had very positive linear outcomes, adding nearly a half point, or a quarter standard deviation improvement for a positive signal. 

 ![ff](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/olsff.pneg) 
 
  ![toi](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/olstoi.pneg) 

   

### 3. Model Building <a class="anchor" id="Model_Building"></a>

   Initially three models were employed; Stochastic Gradient Decent, Random Forests, and Gradient Boosted Regressors. These failed in spectacular fashion. Below is a chart of the predicted vs actual values with the best performing model of that group.
   
![tree](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/xgboost_tree.pneg) 

 The problem here is two-fold: one, upon performance of a Variance Inflation Factor (VIF) analysis it was confirmed that the features were highly correlated amongst themselves and thus multicollinear, the models have a hard time sifting out worthwhile metrics when the target variables are so skewed and thus the results end as an amorphous blob with most values being projected as the mean with little ability to create separation. This is another charting of the results which shows this in clearer fashion.

![tree](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/blob.pneg) 

In order to try and deal with these issues a new model was employed using Poisson Regression Model which is better suited to the purpose given the timebound nature of the contests with a discrete probability of each event occurring. While this did not yield a perfect distribution it was significantly better at predicting the range of values that might occur. 

![tree](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/nonblob.pneg) 




### 4. Application: FantasyCruncher <a class="anchor" id="Application"></a>

	In order to further validate the success of this program the model was build it was tested using FantasyCruncher, a tool which is considered an essential lineup optimizer by most daily fantasy sports professionals. This site does the difficult leg work of maximizing points per lineup, a computationally difficult problem, while allowing you to easily import your own models and calculate line performance based on those. 

	Five days were chosen at random, three of which had data in FantasyCruncher, 2/23/2016, 5/11/2016, and 10/27/2018. Then the model was imported and then compared to a generic projection provided by FantasyCruncher.

 In naïve contests without player input, the model did not perform significantly differently than the off-the-shelf projections, and in fact was de-prioritized by the platform. Below is a comparison of the naïve lineups for the model (upper left) and FantasyCruncher (lower right): 
 ![nm](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/naive1.pneg) 

![fc1](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/naive2.pneg) 

 
 	 That said, the story is wildly different when there is minimal participation by a human. In these contests I told FantasyCruncher to limit the goalies to two options, chosen based on being the starting goalie for the team favored to win by the Las Vegas Casinos. I set the same assignment for the FantasyCruncher model but the scores were wildly different in this case. The Non-Naïve model is on the left, FantasyCruncher on the right.

	In this case both set of contests would have cashed but the model would take a much higher prize while the in-house version would have scraped by.

 ![nm1](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/non_naive1.pneg) 

![fc2](https://github.com/mhbw/springboard/blob/master/Capstone%202/notebooks_datasets/images/nonnaive2.pneg) 

 


### 5. Conclusions <a class="anchor" id="Conclusions"></a>

While the model isn’t predictive enough to be a stand alone product, minimal human coaching can make it viable. 

Significant challenges remain in finding more predictive features that aren’t multicollinear or otherwise reproductions of basic stats. Appropriate model selections for the type of distribution are important as well.

The moving averages have some promise, and perhaps an even simpler model based on those and core features might perform just as well. 



#### Extensions<a class="anchor" id="Future_Work"></a>

 Further apply Las Vegas odds: Perhaps use those as a feature to inform models alongside the core metrics.
 
Consider strength of opponent metrics; marry the trends of the opposing teams to the basic stats. Perhaps facing tougher teams would significantly impact players.

 Attempt an even simpler model with more stripped out features. Attempts were made based on  position but some a revision might be in order with a core set of numbers.


#### Acknowledgments <a class="anchor" id="Acknowledgments"></a>

   I’d like to thank both my mentor for this course, Alex Rutherford, and my mentor through out my data science career, Michael Burton, for helping along the way to guide my progress, point out areas of improvement, and generally being great sounding boards. This project would be much less successful without them. 