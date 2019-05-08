# Building Effective Fantasy Hockey Models: A Case Study

### Introduction

Fantasy sports is an exploding industry, with fantasy football alone estimated to be worth $7 billion in 2017. Models have existed in other sports to predict overall player performance since at least 2003, the most notable example being Baseball Prospectus’ PECOTA, but thus far NHL lacks a similar standard. The game is possibly the most difficult to predict of the major sports, and those who have developed such models have quickly been hired away by NHL teams, taking their models with them. 

To that end, this project will attempt to create a competitive replicable model for hobbyists in a standard ESPN annual fantasy hockey league, who currently buy preseason guides or fantasy subscriptions. 

For this case study I took the ESPN fantasy league that I play in as a guide, and built a number of models using various tools from python's scikit-learn packages, the objective being to build a model that will perform at least as well as a human in the current league.

### Table of Contents

1. [Data Acquisition and Wrangling](#Data_Wrangling)
    i.  [Cleaning](#Data_Cleaning)
    ii. [Feature Creation](#Feature_Creation)


### 1. Data Acquisition and Wrangling <a class="anchor" id="Data_Wrangling"></a>

   * All data was collected from Natural Stat Trick an invaluable resource currated by Micah McCurdy, who compiles all kinds of data from the Nation Hockey League.

   * The only preprocessing I did was to edit some of the header rows in the csv, and added a year column, in order to have that preserved.

    I created the following as my naming convetions for the downloaded files:

**Year**: refers to the year the season completed PK: means a penalty kill unit PP: means powerplay S: means 5-on-5 or standard play 

**Counts**: are an individual occurance Rates: are something over 60 minutes of time, the length of a standard game

#### Data Cleaning <a class="anchor" id="Data_Cleaning"></a>

    Given the specialty nature of this data set there are few na values; it wasn't as tidy as a kaggle excerise but it was close. The notable exception is Draft information, as a number of players were undrafted. After consideration, I decided to make this a number higher than possible, 1000, which would make it stand out and be important for a linear regression, where as if I make it say, 0, have those players sitting by those drafted first (marked as 1 in the over all draft). A normal draft class has about 370 players, so I wanted this number to be higher, showing that they were ranked lower than the players who were drafted. This was mostly for naught, as in the end it appears to have little over all importance and I dropped the feature all together from some of the models. Draft Year I made into 1900, and team is now 'Undrafted'.

    I filled in other values using a set logic based on if the value was a count or a rate. If the value is a count or a percentage of a count, I entered it as zero. If it's a percentage of a rate, then I filled to median. Example of each:

1) **IPP is Individual Point Percentage**, the percentage of goals for that player's team while that player is on the ice that the player earned a point on. For this I filled the NA to 0, becuase logically if you have no points in this category your count would be zero.

2) **OffZoneStartPer is Offensive Zone Start Percentage**. this is a rate, and thus, simply hasn't been calculated since the player hasn't been given the opportunity to have such a stat created.

     In essence the logic here is "if you have a NA due to not getting any points, you are marked zero, if you have an NA due to not being on the ice for that metric, you are returned to the mean". I've heard some arguments that you might leave these as na since many of the sklearn models have the ability to deal with NA values and this unfairly 'bulks up' the mean or median, but again this was a fairly limited number of values in any given column (less than 1%) so mostly academic. 

     There is another alternative which is I found some pointless columns (power play position, etc) which I dropped wholesale.


#### Feature Creation <a class="anchor" id="Feature_Creation"></a>

    I created a number of features for these models, each with 
    
    Age
    
    
