###############################################
###############################################
###############################################
### TITLE: trumpVolatility ####################
### AUTHOR: Miguel Sebastian de la Mata #######
### DATE: October 3rd, 2020 ###################
###############################################
###############################################
###############################################









# The purpose of this project is to gauge what affect, if any,
# Trump's tweets may have on market volatility
# we will use tweet data to predict if volatility goes up







##############################
##############################
###### Import Libraries ######
##############################
##############################



# libraries for data cleaning and manipulation frameworks
import pandas as pd
import numpy as np

# libraries for parallel processing and memory management
import dask.dataframe as dd
import dask.array as da
import multiprocessing
import joblib
from dask import visualize
from dask.distributed import Client, LocalCluster

# data visualization
# the matplotlib import stuff looks dumb, but matplotlib uses tkinter on the backend
# which doesn't play nice with parallel jobs, so we'll change it to Agg
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as mtk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
import graphviz

# Natural Language processing and related data cleaning
import contractions
from nltk import TweetTokenizer
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# data web-scraping for VIX
import yfinance as yf

# other data cleaning
from datetime import timezone
from pandas.tseries.offsets import DateOffset
import sparse
from scipy.sparse import csr_matrix

# model building
from sklearn.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler as daskStandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from dask_ml.wrappers import ParallelPostFit
from sklearn.metrics import accuracy_score
from dask_ml.model_selection import GridSearchCV
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from dask_ml.wrappers import Incremental
from dask_ml.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier




#####################
#####################
## OUTPUT FILEPATH ##
#####################
#####################


outputFilepath = r"C:\Users\sebid\OneDrive\Desktop\trumpVolatilityImages"
dpiSettings = 300
plt.figure(figsize=([10, 12]))




##################
##################
# Data Gathering #
##################
##################




# json files sourced from https://github.com/bpb27/trump_tweet_data_archive
df2015 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2015.json")
df2016 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2016.json")
df2017 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2017.json")
df2018 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2018.json")

# gather our VIX data using Yahoo Finance API
tickerRetrieve = yf.Ticker('^VIX')
VIXdf = pd.DataFrame(tickerRetrieve.history(period='max')['Close'])
VIXdf.rename(columns={'Close' : 'VIX_Daily_Close'}, inplace=True)


# localize the timezone to the eastern US
VIXdf = VIXdf.tz_localize('US/Eastern')




####################
# Data Cleaning ####
####################




# let's start with cleaning our VIX data frame since it is smaller
# we want to determine if the Volatility is higher than it
# was the day before.
VIXdf['volatilityUp'] = VIXdf['VIX_Daily_Close'] - VIXdf['VIX_Daily_Close'].shift()
VIXdf['volatilityUp'] = np.where(VIXdf['volatilityUp'] >= 0, 1, 0)
VIXdf.drop(columns=['VIX_Daily_Close'], inplace=True)


# we're just going to do all of these in a for loop to save time
# first we will make a list of our dfs
dfList = [df2015,
          df2016,
          df2017,
          df2018]

# df2015 doesn't have the variable retweeted_status, so we have to drop it
# here before the loop (we were going to drop it anyway
for df in dfList[1:]:
    df.drop(columns=['retweeted_status'],
            inplace=True)
# apparently this is also true for display_text_range, but only with 2017 and 2018
# same thing for full_text
for df in dfList[2:]:
    df.drop(columns=['display_text_range',
                     'full_text'],
            inplace=True)

# just 2016 has three columns that are unique and kind of useless, also dicts that mess up the merge
df2016.drop(columns=['scopes',
                     'withheld_scope',
                     'withheld_in_countries',
                     'withheld_copyright'],
            inplace=True)

# now we clean common problems at once:
# first we make the datetime an index
# the contributors coordinates, and geo column have no non-null entries in all tables, so we will drop them
# I'm going to go ahead and drop the lang column, I checked to see whether Trump was a secret
# polyglot. He is not. Twitter has been misclassifying his tweet language, perhaps because covfefe sounds French
# I'm going to do some cleaning to keep only the rows with text, but I am going to do that separately in case
# I want to do stuff like count total tweets including retweets in between
# looks like pandas interpreted id_str as a number, so let's convert it to a string and drop id
# favorited is only false so we'll drop that column
# user is a dict containing the other values so lets drop it
# in_reply_to_user_id_str is not a string. so lets change it to a string and drop in_reply_to_user_id
# in_reply_to_status_id_str is not a string, so we will make it one
# place is a dict containing coordinates, don't need that
# extended_entities is a dict containing meta-data for media like GIFs used in a tweet, going to drop that
# quoted_status_id_str is not a string, let's change it to a string and then drop quoted_status_id
# quoted_status contains a tweet object for any tweet that is a quote, don't think we are going to use that
# entities is a dict we are dropping
# all of the id strings don't really serve our purpose
# possibly sensitive is a whole bunch of NA values and couldn't find too much documentation on this variable
for df in dfList:
    # This might run on the side of being long-winded, but I think this is worth going into detail on and
    # I'm fairly certain I will forget why I did this unless I mention it somewhere. Basically, our VIX
    # data shows the closing price that occurs that day and it starts at midnight (logical since there
    # is only one closing price per day. In reality the market is open from 9:30 to 4:00, with a little
    # room for error on either side, as there are early orders and late order, especially for options,
    # which take a little longer to close, and which the VIX price is based on, in addition, some
    # brokerages offer "after-hours" trading within the comfort of their dark pools. This being said, If the
    # Don decides to tweet a though-provoking novela after five, prices aren't really going to change until
    # the next day, and our closing price doesn't happen until 5pm of that next day. Practically speaking,
    # this means our trading "day" should probably start at 5pm after the markets close, and end 5pm the
    # next day at market close. To account for this, we will shift all of the Don's tweets forward 2 hours,
    # so that 5pm becomes the new "midnight" (start/end of our day). This way if he tweets something after
    # market close, those tweets will be the independent variables to predict then next available close price.
    # Sorry if that was really wordy for what is essentially just the one line of code that follows this comment,
    # but this is a pretty big modification to the data if you think about it, and if I pushed all the tweets
    # two hours forward I figured like that would trigger some alarms about that being suspicious.
    df['created_at'] = df['created_at'] + DateOffset(hours=2)
    df.set_index('created_at',
                 inplace=True)
    df.drop(columns=['contributors',
                     'geo',
                     'coordinates',
                     'lang',
                     'id',
                     'favorited',
                     'user',
                     'in_reply_to_user_id',
                     'in_reply_to_status_id',
                     'place',
                     'extended_entities',
                     'quoted_status_id',
                     'quoted_status',
                     'entities',
                     'source',
                     'in_reply_to_screen_name',
                     'id_str',
                     'in_reply_to_user_id_str',
                     'in_reply_to_status_id_str',
                     'quoted_status_id_str',
                     'possibly_sensitive'],
            inplace=True)

# now let's append our dataframes to make a big one
myData = df2015.append(dfList[1:])\
    .sort_index()
myData.drop_duplicates(inplace=True)

# let's transfer the truncated column to an int64 to get it ready for the machine learning process
myData['truncated'] = myData['truncated'].astype(int)

# let's repeat this process for is_quote_status
myData['is_quote_status'] = myData['is_quote_status'].astype(int)

# and same as above for retweeted
myData['retweeted'] = myData['retweeted'].astype(int)

# Now let's join our VIX Data to our tweet data
myData = myData.merge(VIXdf,
                      how='outer',
                      left_index=True,
                      right_index=True)

# June 16th 2015 is the day Trump announced his campaign. Let's start here
myData = myData.loc['2015-06-16':]

# we are going to forward fill the closing values for every day
# note that the closing values occur at market close, while these show
# the value as occuring at 0:00 of that day
# thus the tweets will be predictive of the closing direction for that day
myData['volatilityUp'] = myData['volatilityUp'].fillna(method='ffill')
myData['volatilityUp'] = myData['volatilityUp'].astype('category')

# let's refine our data to just texts that are filled in (no retweets) here
myData = myData.dropna(subset=['text'])

# let's drop all rows with text in the format "@USERNAME:"
# these are other people tweeting @ing him, not his direct tweets
myData = myData[~myData['text'].str.contains(r"\@.*:")]

# this function outputs a column per regex search count, see below
def regexCountColumn(regex, df, newColumnName):
    df[newColumnName] = df['text'].str.count(regex)

# we want to create a variable that counts the number of all caps words in a tweet
# this might be an indication of strong emotional, or more volatile behavior
# we'll do this with a regex search for all all caps words
allCaps = r'\b[A-Z]+\b'
regexCountColumn(allCaps,
                 myData,
                 'allCaps')
print(myData['allCaps'].max())

# create a count of all exclamation points
exclamationPoints = r'!'
regexCountColumn(allCaps,
                 myData,
                 'exclamationPoints')
print(myData['exclamationPoints'].max())

# create a count of all hash tags
hashtags = r'#'
regexCountColumn(hashtags,
                 myData,
                 'hashtags')
print(myData['hashtags'].max())

# create a count of all other user mentions
userHandleCount = r'@[^\s]'
regexCountColumn(userHandleCount,
                 myData,
                 'userHandleCount')
print(myData['userHandleCount'].max())

# let's make another variable that is the number of words in a tweet
tweetWordCount = r'\b[A-Za-z]+\b'
regexCountColumn(tweetWordCount,
                 myData,
                 'tweetWordCount')
print(myData['tweetWordCount'].max())

# before we tokenize our words let's change contractions to full words
myData['noContractions'] = myData['text'].apply(contractions.fix)
print(myData['noContractions'].head())

# we want to split tweets into tokens (words) so we can perform sentiment analysis
# first we need to initiate a tokenizer
# nltk toolkit provides a tokenizer specifically for tweets to deal with dumb shit like hashtags
# we are also going to get into the habit of dropping the column we created before
# for the sake of A) not having a bunch of clutter when we already will have a
# bunch of columns at the end, and B) not using excessive memory
tokenizer = TweetTokenizer()
myData['tokenizedTweets'] = myData['noContractions'].apply(tokenizer.tokenize)
myData = myData.drop(columns='noContractions')

# perform bag of words for topic discovery
# first let's convert everything to lowercase
# each column entry contains a list of the tokens so we need to make a list comprehension
# then apply it to the column, either that or there is some more straightforward way I am totally missing
def listLowerer(list):
    return [token.lower() for token in list]

myData['loweredTokens'] = myData['tokenizedTweets'].apply(listLowerer)
myData = myData.drop(columns='tokenizedTweets')
print(myData['loweredTokens'].head())

# we also want to remove our stop words from the list. I'm just going to define
# another list comprehension then apply that function to the column
# note  that in order to use stop words you must first download the nltk data.
# If you haven't already, go to the Python console and enter "nltk.download()"
def listNoStopWords(list):
    return[token for token in list if token not in stopwords.words('english')]

myData['noStopWordsTokens'] = myData['loweredTokens'].apply(listNoStopWords)
myData = myData.drop(columns='loweredTokens')
print(myData['noStopWordsTokens'].head())

# pretty much going to do the same thing for punctuation as we did right above
def listNoPunct(list):
    return [token for token in list if token not in string.punctuation]

myData['noPunctTokens'] = myData['noStopWordsTokens'].apply(listNoPunct)
myData = myData.drop(columns='noStopWordsTokens')

# next we want to lemmatize our words to convert conjugations ect.
# to root versions of words (i.e. "runs", "run", "ran")
# first we need to instantiate our lemmatizer
lemmatizer = WordNetLemmatizer()

# next we want to go through the entire column, so we will have
# to apply this to the old column in the form of a list
# comprehension function like we did before
def listLemmatize(list):
    return [lemmatizer.lemmatize(token) for token in list]

myData['lemmatizedTokens'] = myData['noPunctTokens'].apply(listLemmatize)
myData = myData.drop(columns='noPunctTokens')
print(myData['lemmatizedTokens'].head())




#########################
### Train Test Split ####
#########################



# before we do any counting of words for the bag of words approach below
# we should split our data into train and test, as we want to avoid
# over-fitting our feature engineering to our data in addition to
# over-fitting it to our model

# split dependent and independent variables
y = myData['volatilityUp']
X = myData.drop(columns=['volatilityUp'])

# here we will randomly split the data into a 70-30 train and test sets
xTrain, xTest, yTrain, yTest = train_test_split(X,
                                                y,
                                                random_state=42,
                                                test_size=0.3,
                                                stratify=y)





#########################
#########################
## Feature Engineering ##
#########################
#########################




# we can build a list that contains all of our words in the entire column here
tokenList = []
tokenList.extend(xTrain['lemmatizedTokens'])
# this is a list of lists so I want to make it into a single list here
tokenList = [item for list in tokenList for item in list]
print(tokenList[0:100])

# lets create a list of just the nouns to see what subjects get mentioned the most
tokensNouns = pos_tag(tokenList)
tokensNouns = [token[0] for token in tokensNouns if token[1] == 'NN']
print(tokensNouns[0:10])

# look over count of all the words
allTextCount = Counter(tokenList)
print(allTextCount.most_common(50))

# let's turn this Counter object into a dataframe so we can explore it,
allTextCountDF = pd.DataFrame\
    .from_records(allTextCount,
                  index=[0])
print(allTextCountDF.info())
print(allTextCountDF.head())

# we will need to melt the dataframe above to be able to visualize the counts against one another
allTextCountMeltedDF = allTextCountDF.melt(var_name='Word',
                                           value_name='Count')
allTextCountMeltedDF.sort_values(by=['Count'],
                                 ascending=False,
                                 inplace=True)
print(allTextCountMeltedDF.head())

# a look over the count of all nouns
allTextNounCount = Counter(tokensNouns)
print(allTextNounCount.most_common(50))

# same for nouns as we did with all words above
allTextNounCountDF = pd.DataFrame.from_records(allTextNounCount,
                                               index=[0])
print(allTextNounCountDF.info())
print(allTextNounCountDF.head())


# we will need to melt the dataframe above to be able to visualize the counts against one another
allTextNounCountMeltedDF = allTextNounCountDF.melt(var_name='Word',
                                                   value_name='Count')
allTextNounCountMeltedDF.sort_values(by=['Count'],
                                     ascending=False,
                                     inplace=True)
print(allTextNounCountMeltedDF.head())

# let's go ahead and create a list of every word that appears
# originally were going to do that but it makes to large of a dataframe for memory
# instead let's do the top 2500 most mentioned nouns to make our dataframe more manageable
popularNounsList = [word for word in allTextNounCountMeltedDF.nlargest(2500, 'Count')['Word']]

print("Firsts ten entries to popularNounsList: " + str(popularNounsList[0:10]))

# same as above, but let's do the top 2500 most mentioned words to make our dataframe more manageable
popularWordsList = [word for word in allTextCountMeltedDF.nlargest(2500, 'Count')['Word']]

print("Firsts ten entries to popularWordsList: " + str(popularWordsList[0:10]))

# I want to create a column for every word in all the tweets and
# count how many times it occurs in that tweet
def countWords(list, word):
    return list.count(word)

for word in popularWordsList:
    xTrain['newCol'] = xTrain['lemmatizedTokens'].apply(countWords,
                                                        args=(word, ))
    xTrain = xTrain.rename(columns={'newCol' : word})

# we need to apply the above to the test data too
for word in popularWordsList:
    xTest['newCol'] = xTest['lemmatizedTokens'].apply(countWords,
                                                      args=(word, ))
    xTest = xTest.rename(columns={'newCol' : word})

print(xTrain.columns)
print(xTrain.head(10))
print(xTrain.info())

# now that we have our counter for our lemmatized tokens and original text, we can drop this column
dropList = ['lemmatizedTokens',
            'text']
xTrain = xTrain.drop(columns=dropList)
xTest = xTest.drop(columns=dropList)


#############################################
## Cluster Analysis for Feature Engineering #
#############################################


# first we want to find the optimal number of clusters to
# perform our KMeans clustering on our X variables

# here we are going to plot an elbow graph of our model
# inertia to determine our number of clusters
ks = np.arange(10,
               41,
               2)
inertias = []

for k in ks:
    # create steps for our pipeline
    pipelineSteps = [('scaler', StandardScaler()),
                     ('kmeans', ParallelPostFit(KMeans(n_clusters=k,
                                                       random_state=42,
                                                       algorithm='full')))]
    model = Pipeline(pipelineSteps)

    # Fit model to samples
    with joblib.parallel_backend('threading',
                                 n_jobs=-1):
        model.fit(xTrain.values)

    # Append the inertia to the list of inertias
    inertias.append(model
                    .named_steps['kmeans']
                    .inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.title("Varying inertia of k clusters\nthrough unsupervised k-means clustering")
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.savefig(outputFilepath+r"\trumpVolatilityVaryingInertiaKmeans.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# create steps for our pipeline
pipelineSteps = [('scaler', StandardScaler()),
                 ('kmeans', KMeans(n_clusters=7))]

# Create a KMeans model with 3 clusters: model
model = Pipeline(pipelineSteps)

# Use fit_predict to fit model and obtain cluster labels
clusterLabels = model.fit_predict(xTrain)

# create a df with labels and volatility outcomes
clusterLabelVoldf = pd.DataFrame({'Cluster Labels': clusterLabels,
                                  'volatilityUp': yTrain})

# Create crosstab table
crossTab = pd.crosstab(clusterLabelVoldf['Cluster Labels'],
                       clusterLabelVoldf['volatilityUp'])
print(crossTab)

# add this as a column to our training data
xTrain['kmeansClusterLabels'] = clusterLabels

# since cluster is a category we need to make dummy columns for every cluster minus 1
xTrain = pd.get_dummies(xTrain,
                        prefix='kmeansClusterLabel',
                        prefix_sep='_',
                        columns=['kmeansClusterLabels'])

# let's do this for our test data too, using the model
# created with the training data
xTest['kmeansClusterLabels'] = model.predict(xTest)

# now we need to do the same as above and create dummy columns for our test data
xTest = pd.get_dummies(xTest,
                       prefix='kmeansClusterLabel',
                       prefix_sep='_',
                       columns=['kmeansClusterLabels'])

# by creating dummy columns in both sets separately, there is a chance the test set may not
# contain all the dummy columns created in the test set. To remedy this so we don't have a
# mismatched number of columns for machine learning later, let's add the columns in the
# training set but not in the test set as zero values
missingColumns = set(xTrain.columns) - set(xTest.columns)
for column in missingColumns:
    xTest[column] = 0

# optimize data for memory consumption (downcast in64s to int16s)
ints = xTrain.select_dtypes(include=['int64',
                                     'int32'])\
    .columns\
    .tolist()
xTrain[ints] = xTrain[ints].apply(pd.to_numeric,
                                  downcast='integer')

ints = xTest.select_dtypes(include=['int64',
                                    'int32'])\
    .columns\
    .tolist()
xTest[ints] = xTest[ints].apply(pd.to_numeric,
                                downcast='integer')

# same for floats
floats = xTrain.select_dtypes(include=['float'])\
    .columns\
    .tolist()
xTrain[floats] = xTrain[floats].apply(pd.to_numeric,
                                      downcast='float')

floats = xTest.select_dtypes(include=['float'])\
    .columns\
    .tolist()
xTest[floats] = xTest[floats].apply(pd.to_numeric,
                                    downcast='float')




################################
################################
## Exploratory Data Analysis ###
################################
################################




# data dictionary for our json files is located here:
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/overview/tweet-object
#


xTrain.info()

# some exploratory questions:

# see if there is significant skew in volatilityUp
sns.catplot(data=myData,
            y='volatilityUp',
            kind='count')
plt.title('Count of number of days where volatility is up')
plt.savefig(outputFilepath+r"\trumpVolatilityCountDaysVolatilityUp.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# how often does Trump tweet per day?
textCountDF = myData.resample('D')\
    .apply({'text' : 'count'})
plt.style.use('ggplot')
plt.plot(textCountDF.text,
         c='blue')
plt.title('Count of Trump\'s tweets by day 2015-2017\nAn average of ' +
          str(int(round(textCountDF['text'].mean(), 1))) +
          ' tweets per day')
plt.xlabel('Date')
plt.xticks(fontsize=7,
           rotation=45)
plt.ylabel('Count of Tweets')
plt.savefig(outputFilepath+r"\trumpVolatilityCountTrumpTweetsPerDayLineplot.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# a probability distribution of the frequency of trumps tweets per day
plt.hist(textCountDF['text'],
         bins=textCountDF['text'].max(),
         density=True,
         align='left')
plt.title('PDF of count of Trump\'s tweets per day')
plt.ylabel('PDF')
plt.xlabel('Count of tweets')
plt.axvline(textCountDF['text'].mean(),
            color='black',
            linewidth=2)
plt.annotate("Mean of\n" + str(textCountDF['text'].mean()) + " tweets",
             xy=(textCountDF['text'].mean(), 0.5),
             xytext=(textCountDF['text'].mean() + 5, 0.1),
             arrowprops=dict(arrowstyle='->',
                             color='black',
                             linewidth=1))
plt.savefig(outputFilepath+r"\trumpVolatilityCountTrumpTweetsPerDayPDF.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# what time of day does trump usually tweet?
# first we resample to an hour by summing the count of tweets
textCountTimeDF = myData.resample('60min')\
    .apply({'text' : 'count'})
# then we subtract 7 hours to put us back in normal time (vs market time)
textCountTimeDF.index = textCountTimeDF.index - DateOffset(hours=7)
print(textCountTimeDF.head())
# we create an hour column based on the index's hour
textCountTimeDF['Hour'] = textCountTimeDF.index.hour
# then we group by our hour, summing the count of text
textCountTimeDF = textCountTimeDF.groupby('Hour').sum()
# we then modify the column to divide its value by the sum of the entire column (a percentage)
textCountTimeDF['text'] = textCountTimeDF['text'] / textCountTimeDF['text'].sum()
print(textCountTimeDF.head())
plt.style.use('ggplot')
plt.plot(textCountTimeDF['text'],
         c='blue',)
plt.title('Percent of Trump\'s tweets by hour of the day 2015-2017')
plt.xlabel('Hour of Day')
plt.ylabel('Percent of tweets by hour of the day')
plt.yaxis.set_major_formatter(mtk.PercentFormatter())
plt.savefig(outputFilepath+r"\trumpVolatilityPercentTrumpTweetsPerHourLineplot.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# histogram of the number of all caps words per tweet
plt.hist(myData['allCaps'],
         bins=myData['allCaps'].max(),
         density=True,
         align='left')
plt.title('Probability distribution of the number of all-caps words per Trump tweet')
plt.xlabel('Number of all caps words per Tweet')
plt.ylabel('PDF')
plt.savefig(outputFilepath+r"\trumpVolatilityAllCapsPerTweet.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# histogram of the number of exclamation points per tweet
plt.hist(myData['exclamationPoints'],
         bins=myData['exclamationPoints'].max(),
         density=True,
         align='left')
plt.title('Probability distribution of the number of exclamation points per Trump tweet')
plt.xlabel('Number of exclamation points per Tweet')
plt.ylabel('PDF')
plt.savefig(outputFilepath+r"\trumpVolatilityExclamationPointsPerTrumpTweet.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# histogram of the number of hashtags per tweet
plt.hist(myData['hashtags'],
         bins=myData['hashtags'].max(),
         density=True,
         align='left')
plt.title('Probability distribution of the number of hashtags per Trump tweet')
plt.xlabel('Number of all hashtags per Tweet')
plt.ylabel('PDF')
plt.savefig(outputFilepath+r"\trumpVolatilityHashtagsPerTweet.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# histogram of the number of userHandles per tweet
plt.hist(myData['userHandleCount'],
         bins=myData['userHandleCount'].max(),
         density=True,
         align='left')
plt.title('Probability distribution of the number of user handle mentions per Trump tweet')
plt.xlabel('Number of user handle mentions per Tweet')
plt.ylabel('PDF')
plt.savefig(outputFilepath+r"\trumpVolatilityUserHandleMentionsPerTweet.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# let's make our word clouds in the shape of Trump's head because, why not?
# I found a free png here: https://www.freeimg.net/photo/868386/trump-donaldtrump-president-usa
# I downloaded it, now I will direct it to that path
# tutorial on how to fit a custom mask here: https://www.datacamp.com/community/tutorials/wordcloud-python
imagePath = r"C:\Users\sebid\OneDrive\Desktop\trump-2069581_1280.png"

trumpMask = np.array(Image.open(imagePath))
trumpColors = ImageColorGenerator(trumpMask)

# let's do a wordcloud of the words used by Trump's tweets
tweetCloud = WordCloud(background_color='gray',
                       max_words=1000,
                       mask=trumpMask,
                       contour_width=1,
                       contour_color='orange',
                       color_func=trumpColors)\
    .generate_from_frequencies(allTextCount)
plt.imshow(tweetCloud,
           interpolation='bilinear')
plt.axis('off')
plt.title("Most popular words in Trump's Tweets")
plt.savefig(outputFilepath+r"\trumpVolatilityMostPopularWords.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# let's do a wordcloud of the nouns used by Trump's tweets, with a different picture
imagePath = r"C:\Users\sebid\OneDrive\Desktop\Donald_Trump_(25320945544).jpg"

# create image colors with a 3 level subsampling
# source code to do this edge finding stuff: https://amueller.github.io/word_cloud/auto_examples/parrot.html
trumpColors = np.array(Image.open(imagePath))
trumpColors = trumpColors[::3, ::3]

# create trump mask
trumpMask = trumpColors.copy()

# do some edge detection
trumpEdges = np.mean([gaussian_gradient_magnitude(trumpColors[:, :, i] / 255., 2) for i in range(3)], axis=0)
trumpMask[trumpEdges > .08] = 255

tweetCloud = WordCloud(background_color='gray',
                       max_words=2000,
                       mask=trumpMask,
                       contour_width=1,
                       contour_color='orange')\
    .generate_from_frequencies(allTextNounCount)

# add colors
trumpColors = ImageColorGenerator(trumpColors)
tweetCloud.recolor(color_func=trumpColors)
plt.imshow(tweetCloud,
           interpolation="bilinear")
plt.axis('off')
plt.title("Most popular nouns in Trump's tweets")
plt.savefig(outputFilepath+r"\trumpVolatilityMostPopularNouns.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()




###########################################
###########################################
##### Model Building ######################
###########################################
###########################################




###############################
#### Initiate Dask Cluster ####
###############################



# now let's convert our pandas dataFrames to Dask dataFrames
xTrain = dd.from_pandas(xTrain,
                        npartitions=multiprocessing.cpu_count() * 2)
xTest = dd.from_pandas(xTest,
                       npartitions=multiprocessing.cpu_count() * 2)
yTrain = dd.from_pandas(yTrain,
                        npartitions=multiprocessing.cpu_count() * 2)
yTest = dd.from_pandas(yTest,
                       npartitions=multiprocessing.cpu_count() * 2)

# here we want to make a list containing all the column names in order from left to right
# so that we can look at feature names on our baseline log reg variable importance graph later
xColNames = xTrain.columns

# convert these to dask arrays to save memory
xTrain = xTrain.values
xTest = xTest.values

# This is a super sparse matrix so we should probably house our
# info in a sparse matrix for memory's sake
xTrain = csr_matrix(xTrain)
xTest = csr_matrix(xTest)


######################################
# Baseline Logistic regression model #
######################################


# we are first going to build a baseline logistic regression
# model that predicts whether volatility goes up or down.

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', ParallelPostFit(StandardScaler(with_mean=False))),
                 ('logReg', ParallelPostFit(estimator=LogisticRegression(max_iter=10000)))]

# establish a pipeline object that will perform the above steps
baselineLogRegPipeline = Pipeline(pipelineSteps)

# create our baseline logistic regression model
with joblib.parallel_backend('threading',
                             n_jobs=-1):
    baselineLogReg = baselineLogRegPipeline.fit(xTrain,
                                                yTrain)

# create our prediction data
yPred = baselineLogReg.predict(xTest)

# create our predicted probabilities
yPredProb = baselineLogReg.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest,
                                 yPredProb)

# score our baseline logistic regression model
baselineLogRegScore = accuracy_score(yTest,
                                     yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest,
                       yPredProb)

print("\nBASELINE LOGISTIC REGRESSION\nACCURACY SCORE:\n")
print(baselineLogRegScore)
print("\nBASELINE LOGISTIC REGRESSION\nAUC SCORES:\n")
print(rocAuc)
print("\nBASELINE LOGISTIC REGRESSION\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest,
                       yPred))
print("\nBASELINE LOGISTIC REGRESSION\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest,
                            yPred))

# plot ROC curve for our model
plt.plot([0, 1],
         [0, 1],
         'k--')
plt.plot(fpr,
         tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Baseline Logistic Regression')
plt.savefig(outputFilepath+r"\trumpVolatilityROCBaselineLogisticRegression.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()

# plot feature importance for the 20 most important features
# create dataframe containing column names and feature coefficients
featureImportance = pd.DataFrame({'variable' : xColNames,
                                  'featureImportance' : baselineLogReg['logReg'].coef_[0]})

# let's narrow this dataFrame down to the 20 most
featureImportance = featureImportance.nlargest(20,
                                               'featureImportance')

# now let's plot our coefficients
plt.plot(featureImportance['variable'],
         featureImportance['featureImportance'])
plt.title("Top twenty most important independent variables for\n"
          "Baseline Logistic Regression Model\n"
          "(coefficient values)")
plt.xlabel("Variable Name")
plt.xticks(rotation=45)
plt.ylabel("Coefficient Value")
plt.margins(0.02)
plt.savefig(outputFilepath+r"\trumpVolatilityFeatureImportanceBaselineLogisticRegression.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()


################################
# SVD Feature Selection Tuning #
################################


# we want to reduce the number of columns we have for our models,
# so we will use SVD to reduce our features
# we are using SVD instead of PCA because SVD is ideal for sparse
# matrices, like we have in bag of words

# to determine the ideal number of features for
# SVD to reduce our data down to, we will first iterate through a
# for loop ranging the number of possible features to reduce to
# and then see which model performs best.
# the model we are using for our comparison is our baseline logistic regression

# create a parameter grid for our grid search CV
paramGrid = {
    'svd__estimator__n_components' : np.arange(40,
                                               501,
                                               20)
}

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', ParallelPostFit(StandardScaler(with_mean=False))),
                 ('svd', ParallelPostFit(TruncatedSVD(random_state=42))),
                 ('logReg', ParallelPostFit(LogisticRegression(max_iter=10000)))]

pipeline = Pipeline(pipelineSteps)

svdSearch = GridSearchCV(pipeline,
                         param_grid=paramGrid,
                         n_jobs=-1,
                         cv=5)

with joblib.parallel_backend('threading',
                             n_jobs=-1):
    svdSearch.fit(xTrain,
                  yTrain)

# Print the tuned parameters and score
print("SVD selection for number of features: {}".format(svdSearch.best_params_))

# no we can define our variable that holds the selected number of features
nComponents = svdSearch.best_params_['svd__estimator__n_components']


####################################################
# Logistic Regression with SVD Dimension Reduction #
####################################################


# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', ParallelPostFit(StandardScaler(with_mean=False))),
                 ('svd', ParallelPostFit(TruncatedSVD(n_components=nComponents,
                                                      random_state=42))),
                 ('logReg', ParallelPostFit(LogisticRegression(max_iter=10000)))]

# establish a pipeline object that will perform the above steps
SVDPipeline = Pipeline(pipelineSteps)

# create our SVD logistic regression model
with joblib.parallel_backend('threading',
                             n_jobs=-1):
    logRegSVD = SVDPipeline.fit(xTrain,
                                yTrain)

# create our prediction data
yPred = logRegSVD.predict(xTest)

# create our predicted probabilities
yPredProb = logRegSVD.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest,
                                 yPredProb)

# score our baseline logistic regression model
logRegSVDScore = accuracy_score(yTest,
                                yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest,
                       yPredProb)

print("\nLOGISTIC REGRESSION WITH SVD\nACCURACY SCORE:\n")
print(baselineLogRegScore)
print("\nLOGISTIC REGRESSION WITH SVD\nAUC SCORES:\n")
print(rocAuc)
print("\nLOGISTIC REGRESSION WITH SVD\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest,
                       yPred))
print("\nLOGISTIC REGRESSION WITH SVD\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest,
                            yPred))

# plot ROC curve for our model
plt.plot([0, 1],
         [0, 1],
         'k--')
plt.plot(fpr,
         tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Logistic Regression with SVD')
plt.savefig(outputFilepath+r"\trumpVolatilityROCBaselineLogisticRegressionWithSVD.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()


###################################
# Tuned Logistic regression model #
###################################


# this is a logistic regression with hyperparameter tuning

# create a parameter grid for our grid search CV
paramGrid = {
    'logReg__estimator__C' : np.logspace(-4,
                                         4,
                                         20)
}

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', ParallelPostFit(StandardScaler(with_mean=False))),
                 ('svd', ParallelPostFit(TruncatedSVD(n_components=nComponents,
                                                      random_state=42))),
                 ('logReg', ParallelPostFit(LogisticRegression(max_iter=20000)))]

# establish a pipeline object that will perform the above steps
tunedLogRegPipeline = Pipeline(pipelineSteps)

# now we go through a grid search crossvalidation on our model using the hyper parameters
tunedLogReg = GridSearchCV(tunedLogRegPipeline,
                           param_grid=paramGrid,
                           n_jobs=-1,
                           cv=5)

# fit our tuned logistic regression model
with joblib.parallel_backend('threading',
                             n_jobs=-1):
    tunedLogReg.fit(xTrain,
                    yTrain)

# create our prediction data
yPred = tunedLogReg.predict(xTest)

# create our predicted probabilities
yPredProb = tunedLogReg.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest,
                                 yPredProb)

# score our tuned logistic regression model
tunedLogRegScore = accuracy_score(yTest,
                                  yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest,
                       yPredProb)

# Print the tuned parameters and score
print("TUNED LOGISTIC REGRESSION HYPERPARAMETERS: {}".format(tunedLogReg.best_params_))
print("Best score is {}".format(tunedLogReg.best_score_))
print("\nTUNED LOGISTIC REGRESSION\nACCURACY SCORE:\n")
print(tunedLogRegScore)
print("\nTUNED LOGISTIC REGRESSION\nAUC SCORE:\n")
print(rocAuc)
print("\nTUNED LOGISTIC REGRESSION\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest,
                       yPred))
print("\nTUNED LOGISTIC REGRESSION\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest,
                            yPred))

# plot ROC curve for our model
plt.plot([0, 1],
         [0, 1],
         'k--')
plt.plot(fpr,
         tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Tuned Logistic Regression')
plt.savefig(outputFilepath+r"/trumpVolatilityROCTunedLogisticRegression.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()


############################
# Decision Tree Classifier #
############################


# this is a decision tree classifier with hyperparameter tuning

# create a parameter grid for our grid search CV
paramGrid = {
    'decisionTree__estimator__criterion' : ["gini",
                                            "entropy"],
    'decisionTree__estimator__max_depth' : np.arange(2,
                                                     101,
                                                     1),
    'decisionTree__estimator__min_samples_split' : np.arange(2,
                                                             41,
                                                             1)
}

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', ParallelPostFit(StandardScaler(with_mean=False))),
                 ('svd', ParallelPostFit(TruncatedSVD(n_components=nComponents,
                                                      random_state=42))),
                 ('decisionTree', ParallelPostFit(DecisionTreeClassifier(random_state=42)))]

# establish a pipeline object that will perform the above steps
decisionTreePipeline = Pipeline(pipelineSteps)

# now we go through a grid search crossvalidation on our model using the hyper parameters
decisionTree = RandomizedSearchCV(decisionTreePipeline,
                                  param_distributions=paramGrid,
                                  n_jobs=-1,
                                  cv=5,
                                  n_iter=60)

# fit our decision tree classifier model
with joblib.parallel_backend('threading',
                             n_jobs=-1):
    decisionTree.fit(xTrain,
                     yTrain)

# create our prediction data
yPred = decisionTree.predict(xTest)

# create our predicted probabilities
yPredProb = decisionTree.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest,
                                 yPredProb)

# score our tuned logistic regression model
decisionTreeScore = accuracy_score(yTest,
                                   yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest,
                       yPredProb)

# Print the tuned parameters and score
print("TUNED DECISION TREE CLASSIFIER HYPERPARAMETERS: {}".format(decisionTree.best_params_))
print("Best score is {}".format(decisionTree.best_score_))
print("\nTUNED DECISION TREE CLASSIFIER\nACCURACY SCORE:\n")
print(decisionTreeScore)
print("\nTUNED DECISION TREE CLASSIFIER\nAUC SCORE:\n")
print(rocAuc)
print("\nTUNED DECISION TREE CLASSIFIER\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest,
                       yPred))
print("\nTUNED DECISION TREE CLASSIFIER\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest,
                            yPred))

# plot ROC curve for our model
plt.plot([0, 1],
         [0, 1],
         'k--')
plt.plot(fpr,
         tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Tuned Decision Tree Classifier')
plt.savefig(outputFilepath+r"\trumpVolatilityROCDecisionTree.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()


############################
# Random Forest Classifier #
############################


# this is a random forest classifier with hyperparameter tuning

# create a parameter grid for our grid search CV
paramGrid = {
    'randomForest__estimator__n_estimators' : [500,
                                               750,
                                               1000],
    'randomForest__estimator__max_features' : ['log2',
                                               'sqrt']
}

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', ParallelPostFit(StandardScaler(with_mean=False))),
                 ('svd', ParallelPostFit(TruncatedSVD(n_components=nComponents,
                                                      random_state=42))),
                 ('randomForest', ParallelPostFit(RandomForestClassifier(random_state=42)))]

# establish a pipeline object that will perform the above steps
randomForestPipeline = Pipeline(pipelineSteps)

# now we go through a grid search crossvalidation on our model using the hyper parameters
randomForest = GridSearchCV(randomForestPipeline,
                            param_grid=paramGrid,
                            n_jobs=-1,
                            cv=5)

# fit our random forest classifier model
with joblib.parallel_backend('threading',
                             n_jobs=-1):
    randomForest.fit(xTrain,
                     yTrain)

# create our prediction data
yPred = randomForest.predict(xTest)

# create our predicted probabilities
yPredProb = randomForest.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest,
                                 yPredProb)

# score our tuned logistic regression model
randomForestScore = accuracy_score(yTest,
                                   yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest,
                       yPredProb)

# Print the tuned parameters and score
print("TUNED RANDOM FOREST CLASSIFIER HYPERPARAMETERS: {}".format(randomForest.best_params_))
print("Best score is {}".format(randomForest.best_score_))
print("\nTUNED RANDOM FOREST CLASSIFIER\nACCURACY SCORE:\n")
print(randomForestScore)
print("\nTUNED RANDOM FOREST CLASSIFIER\nAUC SCORE:\n")
print(rocAuc)
print("\nTUNED RANDOM FOREST CLASSIFIER\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest,
                       yPred))
print("\nTUNED RANDOM FOREST CLASSIFIER\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest,
                            yPred))

# plot ROC curve for our model
plt.plot([0, 1],
         [0, 1],
         'k--')
plt.plot(fpr,
         tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Tuned Random Forest Classifier')
plt.savefig(outputFilepath+r"\trumpVolatilityROCRandomForest.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()


########################
# AdaBoost Classifier #
#######################


# this is an adaboost classifier with hyperparameter tuning

# create a parameter grid for our grid search CV
paramGrid = {
    'adaBoost__estimator__n_estimators' : np.arange(50,
                                                    1001,
                                                    25)
}

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', ParallelPostFit(StandardScaler(with_mean=False))),
                 ('svd', ParallelPostFit(TruncatedSVD(n_components=nComponents,
                                                      random_state=42))),
                 ('adaBoost', ParallelPostFit(AdaBoostClassifier(random_state=42)))]

# establish a pipeline object that will perform the above steps
adaBoostPipeline = Pipeline(pipelineSteps)

# now we go through a grid search crossvalidation on our model using the hyper parameters
adaBoost = GridSearchCV(adaBoostPipeline,
                        param_grid=paramGrid,
                        n_jobs=-1,
                        cv=5)

# fit our random forest classifier model
with joblib.parallel_backend('threading',
                             n_jobs=-1):
    adaBoost.fit(xTrain,
                 yTrain)

# create our prediction data
yPred = adaBoost.predict(xTest)

# create our predicted probabilities
yPredProb = adaBoost.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest,
                                 yPredProb)

# score our tuned adaboost model
adaBoostScore = accuracy_score(yTest,
                               yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest,
                       yPredProb)

# Print the tuned parameters and score
print("TUNED ADABOOST CLASSIFIER HYPERPARAMETERS: {}".format(adaBoost.best_params_))
print("Best score is {}".format(adaBoost.best_score_))
print("\nTUNED ADABOOST CLASSIFIER\nACCURACY SCORE:\n")
print(adaBoostScore)
print("\nTUNED ADABOOST CLASSIFIER\nAUC SCORE:\n")
print(rocAuc)
print("\nTUNED ADABOOST CLASSIFIER\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest,
                       yPred))
print("\nTUNED ADABOOST CLASSIFIER\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest,
                            yPred))

# plot ROC curve for our model
plt.plot([0, 1],
         [0, 1],
         'k--')
plt.plot(fpr,
         tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Tuned AdaBoost Classifier')
plt.savefig(outputFilepath+r"\trumpVolatilityROCAdaBoost.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()


##############################################
# Gradient Boosted Decision Trees Classifier #
##############################################


# this is a gradient boosted decision tree classifier with hyperparameter tuning

# create a parameter grid for our grid search CV
paramGrid = {
    'gradientBoost__estimator__n_estimators' : np.arange(10,
                                                         101,
                                                         5),
    'gradientBoost__estimator__subsample' : np.arange(0.4,
                                                      1.01,
                                                      0.1)
}

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', ParallelPostFit(StandardScaler(with_mean=False))),
                 ('svd', ParallelPostFit(TruncatedSVD(n_components=nComponents,
                                                      random_state=42))),
                 ('gradientBoost', ParallelPostFit(GradientBoostingClassifier(random_state=42)))]

# establish a pipeline object that will perform the above steps
gradientBoostPipeline = Pipeline(pipelineSteps)

# now we go through a grid search crossvalidation on our model using the hyper parameters
gradientBoost = GridSearchCV(gradientBoostPipeline,
                             param_grid=paramGrid,
                             n_jobs=-1,
                             cv=5)

# fit our random forest classifier model
with joblib.parallel_backend('threading',
                             n_jobs=-1):
    gradientBoost.fit(xTrain,
                      yTrain)

# create our prediction data
yPred = gradientBoost.predict(xTest)

# create our predicted probabilities
yPredProb = gradientBoost.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest,
                                 yPredProb)

# score our tuned adaboost model
gradientBoostScore = accuracy_score(yTest,
                                    yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest,
                       yPredProb)

# Print the tuned parameters and score
print("TUNED GRADIENT BOOSTED DECISION TREE CLASSIFIER HYPERPARAMETERS: {}".format(gradientBoost.best_params_))
print("Best score is {}".format(gradientBoost.best_score_))
print("\nTUNED GRADIENT BOOSTED DECISION TREE CLASSIFIER\nACCURACY SCORE:\n")
print(gradientBoostScore)
print("\nTUNED GRADIENT BOOSTED DECISION TREE CLASSIFIER\nAUC SCORE:\n")
print(rocAuc)
print("\nTUNED GRADIENT BOOSTED DECISION TREE CLASSIFIER\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest,
                       yPred))
print("\nTUNED GRADIENT BOOSTED DECISION TREE CLASSIFIER\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest,
                            yPred))

# plot ROC curve for our model
plt.plot([0, 1],
         [0, 1],
         'k--')
plt.plot(fpr,
         tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Tuned Gradient Boosted Decision Tree Classifier')
plt.savefig(outputFilepath+r"\trumpVolatilityROCGradientBoost.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()


####################################
# Ensemble Model Voting Classifier #
####################################

# this is an ensemble model consisting of our earlier models input into a
# voting classifier with hyperparameter tuning

# here is a list of our previous estimators.
classifierModels = [('tunedLogReg', LogisticRegression(C=tunedLogReg.best_params_['logReg__estimator__C'],
                                                       max_iter=20000,
                                                       n_jobs=-1)),
                 ('decisionTree', DecisionTreeClassifier(random_state=42,
                                                         criterion=decisionTree.best_params_['decisionTree__estimator__criterion'],
                                                         max_depth=decisionTree.best_params_['decisionTree__estimator__max_depth'],
                                                         min_samples_split=decisionTree.best_params_['decisionTree__estimator__min_samples_split'])),
                    ('randomForest', RandomForestClassifier(random_state=42,
                                                            n_estimators=randomForest.best_params_['randomForest__estimator__n_estimators'],
                                                            max_features=randomForest.best_params_['randomForest__estimator__max_features'],
                                                            n_jobs=-1)),
                    ('adaBoost', AdaBoostClassifier(random_state=42,
                                                    n_estimators=adaBoost.best_params_['adaBoost__estimator__n_estimators'])),
                    ('gradientBoost', GradientBoostingClassifier(random_state=42,
                                                                 n_estimators=gradientBoost.best_params_['gradientBoost__estimator__n_estimators'],
                                                                 subsample=gradientBoost.best_params_['gradientBoost__estimator__subsample']))]

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', ParallelPostFit(StandardScaler(with_mean=False))),
                 ('svd', ParallelPostFit(TruncatedSVD(n_components=nComponents,
                                                      random_state=42))),
                 ('votingClassifier', ParallelPostFit(VotingClassifier(estimators=classifierModels,
                                                                       n_jobs=-1,
                                                                       voting='soft')))]

# now we go through a grid search crossvalidation on our model using the hyper parameters
votingClassifier = Pipeline(pipelineSteps)

# fit our tuned logistic regression model
with joblib.parallel_backend('threading',
                             n_jobs=-1):
    votingClassifier.fit(xTrain,
                         yTrain)

# create our prediction data
yPred = votingClassifier.predict(xTest)

# create our predicted probabilities
yPredProb = votingClassifier.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest,
                                 yPredProb)

# score our tuned logistic regression model
votingClassifierScore = accuracy_score(yTest,
                                       yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest,
                       yPredProb)

# Print the tuned parameters and score
print("\nMODEL ENSEMBLE VOTING CLASSIFIER\nACCURACY SCORE:\n")
print(decisionTreeScore)
print("\nMODEL ENSEMBLE VOTING CLASSIFIER\nAUC SCORE:\n")
print(rocAuc)
print("\nMODEL ENSEMBLE VOTING CLASSIFIER\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest,
                       yPred))
print("\nMODEL ENSEMBLE VOTING CLASSIFIER\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest,
                            yPred))

# plot ROC curve for our model
plt.plot([0, 1],
         [0, 1],
         'k--')
plt.plot(fpr,
         tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Model Ensemble Voting Classifier')
plt.savefig(outputFilepath+r"\trumpVolatilityROCVotingClassifier.jpeg",
            dpi=dpiSettings,
            bbox_inches='tight')
plt.close()
