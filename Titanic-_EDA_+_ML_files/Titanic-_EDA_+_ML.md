Titanic
================
Rhitamjeet Saharia
6 December 2017

Loading the data
================

``` r
titanic_train = read.csv("D:/Machine Learning/titanic-train.csv",na.strings = '')
titanic_test = read.csv("D:/Machine Learning/titanic-test.csv",na.strings = '')

Survived = titanic_train$Survived
titanic_full = bind_rows(titanic_train[-2],titanic_test)
```

Data Sanity
===========

``` r
colSums(is.na(titanic_full))/nrow(titanic_full)*100
```

    ## PassengerId      Pclass        Name         Sex         Age       SibSp 
    ##  0.00000000  0.00000000  0.00000000  0.00000000 20.09167303  0.00000000 
    ##       Parch      Ticket        Fare       Cabin    Embarked 
    ##  0.00000000  0.00000000  0.07639419 77.46371276  0.15278839

``` r
colSums(is.na(titanic_full))
```

    ## PassengerId      Pclass        Name         Sex         Age       SibSp 
    ##           0           0           0           0         263           0 
    ##       Parch      Ticket        Fare       Cabin    Embarked 
    ##           0           0           1        1014           2

-   It is seen that there are a lot of missing values in Age column(20%) and Cabin(77%).
-   1 value of Fare is missing along with 2 missing values of Embaked.

Exploratory Data Analysis
=========================

Univariate Analysis
-------------------

-   Pclass

``` r
pclass = titanic_train %>% group_by(Pclass,Survived) %>% summarise(Count = n())

ggplot(pclass) + geom_bar(aes(x = Pclass, y = Count, fill = as.factor(Survived)), stat = 'identity') + labs(title = "Pclass wise Survivors") 
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-3-1.png) - No of Passengers in 1st and 2nd Class were roughly around 200 each. But the number of passengers in 3rd class were almost double i.e 500. - Also , it can be seen that the proportion of survival in 1st class and 2nd class is more than in 3rd class.

-   Sex

``` r
sex = titanic_train %>% group_by(Sex,Survived) %>% summarise(Count = n())
ggplot(sex) + geom_bar(aes(x = Sex, y = Count, fill = as.factor(Survived)), stat = 'identity')+ labs(title = "Sex v/s Survivors")
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-4-1.png) - We see that females had a much higher survival rate than males.

-   Age

``` r
ggplot(titanic_train) + geom_histogram(aes(x = Age, fill = as.factor(Survived))) + facet_wrap(~Sex)
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
ggplot(titanic_train) + geom_boxplot(aes(y = Age, x = Sex, fill = Sex), alpha = 0.5)
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-5-2.png)

``` r
ggplot(titanic_train) + geom_boxplot(aes(y = Age, x = Sex, fill = Sex), alpha = 0.5) + facet_wrap(~Pclass) + labs(title = "Pclass wise Age plots") 
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-5-3.png) - The Age Distribution of Males and Females follow an almost similar pattern overall. But the survival rates for females is much more than males.

-   However, we will dig deeper in order to find more patterns in Age as we have to impute missing values.
-   We see that, median age of 3rd class passengers was less. This might be because of more children travelling with them( we will have to find out!)
-   This leads us to our first Feature extraction step. We will mutate a column indicating Adult or Child. - And then try and find if there is some interesting info from it.

-   Feature Extraction -Age

``` r
titanic_full$Age_category = ifelse(titanic_full$Age<14,"Child","Adult")
titanic_train$Age_category = ifelse(titanic_train$Age<14,"Child","Adult")
age_category = titanic_train %>% group_by(Age_category,Survived,Pclass) %>% summarise(Count = n()) %>% filter(!is.na(Age_category))

ggplot(age_category) + geom_bar(aes(x = Pclass, y = Count, fill = Age_category),stat = 'identity') + labs(title = "No. of children per Pclass")
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-6-1.png)

-   So we see that there were indeed more children in 3rd class. While imputing value of age we will keep it in mind.

-   Next, we use apriori algorithm to find some rules regarind survival. We will focus on Pclass, Age\_category and Sex.

Association Rule Mining - Aproiri
=================================

``` r
titanic_apriori = titanic_train[,c(2,3,5,13)]
for(i in 1:4)
{
  titanic_apriori[,i] = as.factor(titanic_apriori[,i])
}
titanic_rules = apriori(titanic_apriori,parameter = list(minlen = 3, supp = 0.002, conf = 0.8),appearance = list(default = "none",rhs = c("Survived=1","Survived=0"),
                                  lhs = c("Pclass=1","Pclass=2","Pclass=3"    ,"Age_category=Child","Age_category=Adult","Sex=male","Sex=female")))
```

    ## Apriori
    ## 
    ## Parameter specification:
    ##  confidence minval smax arem  aval originalSupport maxtime support minlen
    ##         0.8    0.1    1 none FALSE            TRUE       5   0.002      3
    ##  maxlen target   ext
    ##      10  rules FALSE
    ## 
    ## Algorithmic control:
    ##  filter tree heap memopt load sort verbose
    ##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
    ## 
    ## Absolute minimum support count: 1 
    ## 
    ## set item appearances ...[9 item(s)] done [0.00s].
    ## set transactions ...[9 item(s), 891 transaction(s)] done [0.00s].
    ## sorting and recoding items ... [9 item(s)] done [0.00s].
    ## creating transaction tree ... done [0.00s].
    ## checking subsets of size 1 2 3 4 done [0.00s].
    ## writing ... [13 rule(s)] done [0.00s].
    ## creating S4 object  ... done [0.00s].

``` r
rules.sorted = sort(titanic_rules, by = 'confidence')


#removing redundant rules
subset.matrix = is.subset(rules.sorted, rules.sorted, sparse = F)
subset.matrix[lower.tri(subset.matrix, diag= T)] = NA

redundant = colSums(subset.matrix,na.rm = T) >=1

which(redundant)
```

    ## {Survived=1,Pclass=2,Sex=female,Age_category=Child} 
    ##                                                   2 
    ##   {Survived=1,Pclass=2,Sex=male,Age_category=Child} 
    ##                                                   3 
    ## {Survived=1,Pclass=2,Sex=female,Age_category=Adult} 
    ##                                                   9

``` r
#removing redundant rules

rules.pruned = rules.sorted[!redundant]

inspect(rules.pruned)
```

    ##      lhs                     rhs              support confidence     lift count
    ## [1]  {Pclass=2,                                                                
    ##       Age_category=Child} => {Survived=1} 0.020202020  1.0000000 2.605263    18
    ## [2]  {Pclass=1,                                                                
    ##       Sex=male,                                                                
    ##       Age_category=Child} => {Survived=1} 0.003367003  1.0000000 2.605263     3
    ## [3]  {Pclass=1,                                                                
    ##       Sex=female,                                                              
    ##       Age_category=Adult} => {Survived=1} 0.092031425  0.9761905 2.543233    82
    ## [4]  {Pclass=1,                                                                
    ##       Sex=female}         => {Survived=1} 0.102132435  0.9680851 2.522116    91
    ## [5]  {Pclass=2,                                                                
    ##       Sex=male,                                                                
    ##       Age_category=Adult} => {Survived=0} 0.094276094  0.9333333 1.514754    84
    ## [6]  {Pclass=2,                                                                
    ##       Sex=female}         => {Survived=1} 0.078563412  0.9210526 2.399584    70
    ## [7]  {Pclass=3,                                                                
    ##       Sex=male,                                                                
    ##       Age_category=Adult} => {Survived=0} 0.223344557  0.8728070 1.416523   199
    ## [8]  {Pclass=3,                                                                
    ##       Sex=male}           => {Survived=0} 0.336700337  0.8645533 1.403128   300
    ## [9]  {Pclass=2,                                                                
    ##       Sex=male}           => {Survived=0} 0.102132435  0.8425926 1.367486    91
    ## [10] {Sex=male,                                                                
    ##       Age_category=Adult} => {Survived=0} 0.386083053  0.8269231 1.342055   344

-   We see some interesting association rules that the apriori algorithm mines for us!. Let's examine some of them.
-   The top 2 rules tell us that *children* from Pclass 1 & 2 had a *100% survival rate!*. The same does not hold true for Pclass 3 children!
-   Rules 3, 4, 6 tell us that *females* from Pclass 1 & 2 had very high chance of survival*(over 90%)*
-   We also see that adult males from Pclass 2 & 3 had very high chance of non-survival(80-90%).

-   Survival of Children & Females by Pclass

``` r
children = age_category %>% filter(Age_category=='Child')
p1 = ggplot(children) + geom_bar(aes(x = Pclass, y = Count, fill = as.factor(Survived)),stat = 'identity') + labs(title = "Survival of Children across Pclass")

sex_pclass_female = titanic_train %>% group_by(Sex, Pclass,Survived) %>% summarise(Count = n()) %>% filter(Sex =="female")
p2 = ggplot(sex_pclass_female) + geom_bar(aes(x = Pclass, y = Count, fill = as.factor(Survived)),stat = 'identity')  + labs(title = "Survival of Females across Pclass")

ggarrange(p1,p2)
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-8-1.png) - So, like we saw, not all children had the same chance of survival. Only half the children of Pclass 3 survived. A sad fact! - Same is the case with females from Pclass 3. - So both females & children had a lower survival rate in Pclass. We have a column of Parents+Children & Siblings+Spouse. So maybe, families had lower chances of survival then single males?? We will find out!

Feature Engineering - Family Size
=================================

``` r
titanic_full$Family_size = titanic_full$SibSp + titanic_full$Parch + 1
titanic_train$Family_size = titanic_train$SibSp + titanic_train$Parch + 1

ggplot(titanic_train) + geom_histogram(aes(x = Family_size, fill = as.factor(Survived))) + labs(title = "Family Size v/s Survival")
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-9-1.png)

-   Fare

``` r
ggplot(titanic_train) + geom_histogram(aes(x = Fare,fill = as.factor(Survived))) + labs(title = "Does Higher fare ensure Survival?")
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-10-1.png) - With higher fares, the chances of survival increase!

-   Cabin

``` r
cabin = gsub(pattern = '[^[:alpha:]]', replacement = '', titanic_full$Cabin)
cabin = substr(cabin,1,1)
titanic_full$cabin_deck = cabin
titanic_full$cabin_deck = as.factor(titanic_full$cabin_deck)

cabin1 = gsub(pattern = '[^[:alpha:]]', replacement = '', titanic_train$Cabin)
cabin1 = substr(cabin1,1,1)
titanic_train$cabin_deck = cabin1
titanic_train$cabin_deck = as.factor(titanic_train$cabin_deck)

ggplot(titanic_train) + geom_bar(aes(x = cabin_deck, fill = as.factor(Pclass),stat = "count")) + labs(title = "Distribution of Pclass passengers across decks")
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-11-1.png)

-   Aha!We see that deck A , B, C are completely occupied by Pclass 1.
-   D,E have Pclass 1 & 2. deck F has Pclass 2 passengers.
-   Also we see that, in our data the cabin decks of mostly 1st and 2nd Pclass passengers are captured. And very few Pclass 3rd passengers share the same decks with Pclass 1 & Pclass 2. So, we can assume that Pclass 3 passengers mostly had a seperate deck category.
-   We now check the fares corresponding to the decks to get a more clear picture.

-   deck\_fare

``` r
p3 = ggplot(titanic_train) + geom_boxplot(aes(x = cabin_deck, y = Fare, fill = cabin_deck), alpha = 0.5) + labs(title = "cabin_deck wise Fare")
p4 = ggplot(titanic_train) + geom_boxplot(aes(x = as.factor(titanic_train$Pclass), y = Fare, fill = as.factor(Pclass)), alpha = 0.5) + labs(title = "Pclass wise Fare")

ggarrange(p3,p4)
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-12-1.png)

-   Deck wise Survival

``` r
ggplot(titanic_train) + geom_bar(aes(x = cabin_deck, fill = as.factor(Survived)), stat = 'count') + labs(title = "cabin_deck wise Survival")
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-13-1.png)

-   we tried to see if the cabin\_decks had any kind of bearing on survival. From here it is hard to tell.
-   Maybe after imputing the cabin deck values, we can have a better picture.
-   we are quite ready to impute the deck values as we have already plotted the class wise decks occupied and also the average fares of each deck. We have sufficient grounds to impute cabin decks now.It depends on Pclass & Fare primarily. Though family size may also determine in which deck families are residing.....possible!(A hypothesis)

``` r
titanic_full$Family_group = cut(titanic_full$Family_size,breaks = c(1,2,5,12), include.lowest = T, right = F, labels = c('single','small family','large family'))

titanic_train$Family_group = cut(titanic_train$Family_size,breaks = c(1,2,5,12), include.lowest = T, right = F, labels = c('single','small family','large family'))

ggplot(titanic_train) + geom_bar(aes(fill = Family_group,x = cabin_deck), stat = 'count')
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-14-1.png) - The singles and small families are quite evenly distributed. - We don't see clearly where the large families are? lets find out.

``` r
table(titanic_full[titanic_full$Family_group=='large family',]$Pclass)
```

    ## 
    ##  1  2  3 
    ## 11  2 69

``` r
titanic_full[titanic_full$Family_group=='large family' & titanic_full$Pclass==1,]$cabin_deck
```

    ##  [1] C C B C C B B C B C B
    ## Levels: A B C D E F G T

-   Most of the large families are from Pclass 3. Since they anyway donot belong to deck A,B,C,D we will not bother about them.
-   We try to find out about Large families of Pclass 1.
-   We see that all the Pclass 1 large families are residing in decks B or C. And they have multiple rooms.
-   So our hypothesis that families are assigned a particular deck irrespective of Pclass is wrong. Our earlier hypothesis that decks are decided primarily by Pclass holds more truth.

Data Imputation
===============

-   Model Selection
-   We will try various algorithms.Random Forest, GBM & Decision tree.
-   First we impute the Fare column where values are missing.

``` r
titanic_full[is.na(titanic_full$Fare),]$Fare=median(titanic_full[titanic_full$Embarked =='S' & titanic_full$Pclass == 3 & titanic_full$Family_size == 1,]$Fare, na.rm = T)
```

-   We impute the fare by looking at the chracteristics of the passenger. Fare is likely to be decided by port of Embarkment, Pclass and Family Size.
-   We see the Fare for above characteristics and impute the median fare.

-   Imputation of Age
-   First we look if age determines survival or not.

``` r
ggplot(titanic_train) + geom_histogram(aes(x = Age, fill = as.factor(Survived )),binwidth = 3)
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-17-1.png) - We see it does! For children and elderly, survival rates are more. - For children of Ages upto 14 and elders above 52 years of Age, surival rate is more. - So we will modify the Age\_grp category to include Age\_group categories.

``` r
titanic_full$Age_category = cut(titanic_full$Age,breaks = seq(from = 0, to = 85,by = 5), include.lowest = F, right = T)

titanic_train$Age_category = cut(titanic_train$Age,breaks = seq(from = 0, to = 85,by = 5), include.lowest = F, right = T)

age_test = titanic_full[is.na(titanic_full$Age),c(2,4,5,6,7,9)]
age_train = titanic_full[!is.na(titanic_full$Age),c(2,4,5,6,7,9)]

#Random Forest
age_rf = randomForest(x = age_train[-3],
                      y = age_train$Age,
                      ntree = 500)
age_impute = predict(age_rf, newdata = age_test[-3])

age_test$Age = age_impute
age_test$PassengerId = rownames(age_test)

titanic_full[is.na(titanic_full$Age),]$Age = age_test$Age

titanic_full$Age_category = cut(titanic_full$Age,breaks = seq(from = 0, to = 85,by = 5), include.lowest = F, right = T)
```

-   We will not impute Embarked column as it should have no bearing on Survival(hypothesis)

-   Now, we impute Cabin.

``` r
cabin_train = titanic_full[!is.na(titanic_full$cabin_deck),c(2,4,5,9,15,14)]

cabin_test = titanic_full[is.na(titanic_full$cabin_deck),c(2,4,5,9,15,14)]

cabin_rf = randomForest(x = cabin_train[-6],
                        y = cabin_train$cabin_deck,
                        ntree = 500)
cabin_impute = predict(cabin_rf, newdata = cabin_test[-6])
cabin_test$cabin_deck = cabin_impute

titanic_full[is.na(titanic_full$cabin_deck),]$cabin_deck = cabin_test$cabin_deck
```

Feature Selection
=================

-   Name, Title

``` r
titanic_full$Title = gsub('(.*,)|(\\..*)','',titanic_full$Name)

table(titanic_full$Title)
```

    ## 
    ##          Capt           Col           Don          Dona            Dr 
    ##             1             4             1             1             8 
    ##      Jonkheer          Lady         Major        Master          Miss 
    ##             1             1             2            61           260 
    ##          Mlle           Mme            Mr           Mrs            Ms 
    ##             2             1           757           197             2 
    ##           Rev           Sir  the Countess 
    ##             8             1             1

``` r
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

titanic_full$Title[titanic_full$Title == 'Mlle']        = 'Miss' 
titanic_full$Title[titanic_full$Title == 'Ms']          = 'Miss'
titanic_full$Title[titanic_full$Title == 'Mme']         = 'Mrs' 
titanic_full$Title[titanic_full$Title %in% rare_title]  = 'Rare Title'
titanic_full$Title = as.factor(titanic_full$Title)
```

-   We are ready now to fit the data into Machine Learnig model!

-   Selecting features

``` r
final_train = titanic_full[1:891,c(2,4,5,6,7,9,12,13,14,15,16)]
final_test = titanic_full[892:1309,c(2,4,5,6,7,9,12,13,14,15,16)]
final_train$Survived = as.factor(Survived)

#Random Forest

final_rf = randomForest(x = final_train[-12], y = final_train$Survived,
                        ntree = 1000)

pred = predict(final_rf, newdata = final_test)

# GBM
final_gbm = gbm.fit(x = final_train[-12],
                    y = final_train$Survived,
                    distribution = 'gaussian',
                    n.trees = 1000,
                    interaction.depth = 3,
                    shrinkage = 0.01,
                    nTrain = 0.8*nrow(final_train))
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.2358          0.2285     0.0100    0.0020
    ##      2        0.2338          0.2263     0.0100    0.0020
    ##      3        0.2318          0.2242     0.0100    0.0019
    ##      4        0.2299          0.2222     0.0100    0.0017
    ##      5        0.2280          0.2201     0.0100    0.0019
    ##      6        0.2263          0.2186     0.0100    0.0016
    ##      7        0.2247          0.2170     0.0100    0.0015
    ##      8        0.2228          0.2152     0.0100    0.0017
    ##      9        0.2212          0.2134     0.0100    0.0018
    ##     10        0.2196          0.2121     0.0100    0.0015
    ##     20        0.2043          0.1964     0.0100    0.0013
    ##     40        0.1805          0.1719     0.0100    0.0009
    ##     60        0.1645          0.1560     0.0100    0.0007
    ##     80        0.1532          0.1446     0.0100    0.0004
    ##    100        0.1450          0.1366     0.0100    0.0003
    ##    120        0.1391          0.1311     0.0100    0.0001
    ##    140        0.1348          0.1272     0.0100    0.0001
    ##    160        0.1313          0.1243     0.0100    0.0000
    ##    180        0.1287          0.1222     0.0100   -0.0000
    ##    200        0.1266          0.1205     0.0100    0.0000
    ##    220        0.1250          0.1195     0.0100    0.0000
    ##    240        0.1234          0.1180     0.0100    0.0000
    ##    260        0.1221          0.1169     0.0100    0.0000
    ##    280        0.1211          0.1169     0.0100    0.0000
    ##    300        0.1202          0.1162     0.0100   -0.0000
    ##    320        0.1193          0.1158     0.0100   -0.0000
    ##    340        0.1183          0.1151     0.0100    0.0000
    ##    360        0.1175          0.1147     0.0100   -0.0000
    ##    380        0.1169          0.1138     0.0100   -0.0000
    ##    400        0.1161          0.1133     0.0100   -0.0000
    ##    420        0.1153          0.1130     0.0100   -0.0000
    ##    440        0.1147          0.1130     0.0100   -0.0000
    ##    460        0.1138          0.1125     0.0100   -0.0000
    ##    480        0.1133          0.1125     0.0100   -0.0001
    ##    500        0.1127          0.1124     0.0100   -0.0001
    ##    520        0.1121          0.1120     0.0100   -0.0000
    ##    540        0.1116          0.1119     0.0100   -0.0000
    ##    560        0.1111          0.1120     0.0100   -0.0001
    ##    580        0.1105          0.1118     0.0100   -0.0000
    ##    600        0.1100          0.1117     0.0100   -0.0000
    ##    620        0.1095          0.1117     0.0100   -0.0001
    ##    640        0.1090          0.1119     0.0100   -0.0000
    ##    660        0.1086          0.1122     0.0100   -0.0000
    ##    680        0.1081          0.1120     0.0100   -0.0001
    ##    700        0.1077          0.1121     0.0100   -0.0001
    ##    720        0.1072          0.1125     0.0100   -0.0001
    ##    740        0.1068          0.1124     0.0100   -0.0001
    ##    760        0.1065          0.1124     0.0100   -0.0001
    ##    780        0.1061          0.1120     0.0100   -0.0000
    ##    800        0.1059          0.1119     0.0100   -0.0000
    ##    820        0.1055          0.1120     0.0100   -0.0001
    ##    840        0.1051          0.1118     0.0100   -0.0001
    ##    860        0.1048          0.1115     0.0100   -0.0001
    ##    880        0.1044          0.1116     0.0100   -0.0000
    ##    900        0.1040          0.1119     0.0100   -0.0000
    ##    920        0.1037          0.1122     0.0100   -0.0001
    ##    940        0.1034          0.1123     0.0100   -0.0001
    ##    960        0.1031          0.1122     0.0100   -0.0000
    ##    980        0.1027          0.1124     0.0100   -0.0001
    ##   1000        0.1024          0.1124     0.0100   -0.0001

``` r
gbm.perf(final_gbm)
```

    ## Using test method...

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-21-1.png)

    ## [1] 869

``` r
summary(final_gbm)
```

![](Titanic-_EDA_+_ML_files/figure-markdown_github/unnamed-chunk-21-2.png)

    ##                       var    rel.inf
    ## Title               Title 41.1000909
    ## Age_category Age_category 16.1649099
    ## Fare                 Fare 13.7963218
    ## Pclass             Pclass  9.5549277
    ## cabin_deck     cabin_deck  7.9148110
    ## Age                   Age  3.2783127
    ## Family_group Family_group  3.1633744
    ## Family_size   Family_size  3.0870701
    ## SibSp               SibSp  1.0655546
    ## Sex                   Sex  0.6723247
    ## Parch               Parch  0.2023022

``` r
pred1 = predict(final_gbm, newdata = final_test, n.trees = 836)
pred1 = ifelse(pred1<1.5,0,1)

###decision tree
final_tree = tree(data = final_train,
                  formula = Survived~.)
pred2 = predict(final_tree, newdata = final_test, type = 'class')

##NaiveBayes
final_NaiveBayes = naiveBayes(x = final_train[-c(12)],
                y = final_train$Survived)
pred3 = predict(final_NaiveBayes, newdata = final_test)
##ensemble learning

final = as.data.frame(cbind(as.numeric(as.character(pred)),as.numeric(as.character(pred2)),as.numeric(as.character(pred3))))

final_yes = apply(final,1,function(x) x[which.max(table(x))] )
final_yes
```

    ##   [1] 0 1 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 1 0 1 0 1 0 0 0 0 0 1 1 0
    ##  [36] 0 1 1 0 1 0 1 0 1 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 1
    ##  [71] 1 0 1 0 1 0 0 1 0 1 1 0 0 0 0 0 1 1 1 1 1 0 1 0 0 0 1 0 1 0 1 0 0 0 1
    ## [106] 0 0 0 0 0 0 1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0
    ## [141] 0 1 0 0 1 0 0 0 0 0 1 0 0 1 0 0 1 1 1 1 1 1 1 0 0 1 0 0 1 1 0 0 0 0 0
    ## [176] 1 1 0 1 1 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 1 0 1 1 1 0 1 0 0 1 0 1 0
    ## [211] 0 0 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 1
    ## [246] 0 1 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 1 0 0 0 1 1 0 1 0 0 0 0
    ## [281] 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 0 1 1
    ## [316] 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 1
    ## [351] 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 1 0 0 1 0 0 1 1 1 0 0 0 0 0 1 1 1
    ## [386] 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 0 1 0 0 0 0 0 1 1 1 1 1 0 0 0 0 1

-   We use different models - Random Forest, Decision Tree,GBM and NaiveBayes. We take the voting approach to create an ensemble model prediction.
-   We finally get a accuracy of 78.99%.
