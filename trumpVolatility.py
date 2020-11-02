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

# libraries for parallel processing
import dask.dataframe as dd
import dask.array as da
import multiprocessing
from dask.distributed import Client, LocalCluster

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude

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

# model building
from sklearn.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler as daskStandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from dask_ml.wrappers import ParallelPostFit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD









##################
##################
# Data Gathering #
##################
##################



# print our number of cpu cores
print(multiprocessing.cpu_count())

# json files sourced from https://github.com/bpb27/trump_tweet_data_archive
df2015 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2015.json")
df2016 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2016.json")
df2017 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2017.json")
df2018 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2018.json")

# gather our VIX data using Yahoo Finance API
tickerRetrieve = yf.Ticker('^VIX')
VIXdf = pd.DataFrame(tickerRetrieve.history(period='max')['Close'])
VIXdf.rename(columns={'Close' : 'VIX_Daily_Close'}, inplace=True)

# convert the timeseries to UTC
VIXdf = VIXdf.tz_localize(timezone.utc)



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
dfList = [df2015, df2016, df2017, df2018]

# df2015 doesn't have the variable retweeted_status, so we have to drop it
# here before the loop (we were going to drop it anyway
for df in dfList[1:]:
    df.drop(columns=['retweeted_status'], inplace=True)
# apparently this is also true for display_text_range, but only with 2017 and 2018
# same thing for full_text
for df in dfList[2:]:
    df.drop(columns=['display_text_range',
                     'full_text'], inplace=True)

# just 2016 has three columns that are unique and kind of useless, also dicts that mess up the merge
df2016.drop(columns=['scopes',
                     'withheld_scope',
                     'withheld_in_countries',
                     'withheld_copyright'], inplace=True)

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
    df.set_index('created_at', inplace=True)
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
                     'possibly_sensitive'], inplace=True)

# now let's append our dataframes to make a big one
myData = df2015.append(dfList[1:]).sort_index()
myData.drop_duplicates(inplace=True)

# let's transfer the truncated column to an int64 to get it ready for the machine learning process
myData['truncated'] = myData['truncated'].astype(int)

# let's repeat this process for is_quote_status
myData['is_quote_status'] = myData['is_quote_status'].astype(int)

# and same as above for retweeted
myData['retweeted'] = myData['retweeted'].astype(int)

# Now let's join our VIX Data to our tweet data
myData = myData.merge(VIXdf, how='outer', left_index=True, right_index=True)

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
regexCountColumn(allCaps, myData, 'allCaps')
print(myData['allCaps'].max())

# create a count of all exclamation points
exclamationPoints = r'!'
regexCountColumn(allCaps, myData, 'exclamationPoints')
print(myData['exclamationPoints'].max())

# create a count of all hash tags
hashtags = r'#'
regexCountColumn(hashtags, myData, 'hashtags')
print(myData['hashtags'].max())

# create a count of all other user mentions
userHandleCount = r'@[^\s]'
regexCountColumn(userHandleCount, myData, 'userHandleCount')
print(myData['userHandleCount'].max())

# let's make another variable that is the number of words in a tweet
tweetWordCount = r'\b[A-Za-z]+\b'
regexCountColumn(tweetWordCount, myData, 'tweetWordCount')
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
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)





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
allTextCountDF = pd.DataFrame.from_records(allTextCount, index=[0])
print(allTextCountDF.info())
print(allTextCountDF.head())

# we will need to melt the dataframe above to be able to visualize the counts against one another
allTextCountMeltedDF = allTextCountDF.melt(var_name='Word', value_name='Count')
allTextCountMeltedDF.sort_values(by=['Count'], ascending=False, inplace=True)
print(allTextCountMeltedDF.head())

# a look over the count of all nouns
allTextNounCount = Counter(tokensNouns)
print(allTextNounCount.most_common(50))

# same for nouns as we did with all words above
allTextNounCountDF = pd.DataFrame.from_records(allTextNounCount, index=[0])
print(allTextNounCountDF.info())
print(allTextNounCountDF.head())


# we will need to melt the dataframe above to be able to visualize the counts against one another
allTextNounCountMeltedDF = allTextNounCountDF.melt(var_name='Word', value_name='Count')
allTextNounCountMeltedDF.sort_values(by=['Count'], ascending=False, inplace=True)
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
    xTrain['newCol'] = xTrain['lemmatizedTokens'].apply(countWords, args=(word, ))
    xTrain = xTrain.rename(columns={'newCol' : word})

# we need to apply the above to the test data too
for word in popularWordsList:
    xTest['newCol'] = xTest['lemmatizedTokens'].apply(countWords, args=(word, ))
    xTest = xTest.rename(columns={'newCol' : word})

print(xTrain.columns)
print(xTrain.head(10))
print(xTrain.info())

# now that we have our counter for our lemmatized tokens and original text, we can drop this column
dropList = ['lemmatizedTokens', 'text']
xTrain = xTrain.drop(columns=dropList)
xTest = xTest.drop(columns=dropList)


#############################################
## Cluster Analysis for Feature Engineering #
#############################################


# first we want to find the optimal number of clusters to
# perform our KMeans clustering on our X variables

# here we are going to plot an elbow graph of our model
# inertia to determine our number of clusters
ks = range(1, 20)
inertias = []

for k in ks:
    # create steps for our pipeline
    pipelineSteps = [('scaler', StandardScaler()),
                     ('kmeans', KMeans(n_clusters=k))]
    model = Pipeline(pipelineSteps)

    # Fit model to samples
    model.fit(xTrain)

    # Append the inertia to the list of inertias
    inertias.append(model
                    .named_steps['kmeans']
                    .inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# create steps for our pipeline
pipelineSteps = [('scaler', StandardScaler()),
                 ('kmeans', KMeans(n_clusters=5))]

# Create a KMeans model with 3 clusters: model
model = Pipeline(pipelineSteps)

# Use fit_predict to fit model and obtain cluster labels
clusterLabels = model.fit_predict(xTrain)

# create a df with labels and volatility outcomes
clusterLabelVoldf = pd.DataFrame({'Cluster Labels': clusterLabels, 'volatilityUp': yTrain})

# Create crosstab table
crossTab = pd.crosstab(clusterLabelVoldf['Cluster Labels'], clusterLabelVoldf['volatilityUp'])
print(crossTab)

# add this as a column to our training data
xTrain['kmeansClusterLabels'] = clusterLabels

# let's do this for our test data too, using the model
# created with the training data
xTest['kmeansClusterLabels'] = model.predict(xTest)

# optimize data for memory consumption (downcast in64s to int16s)
ints = xTrain.select_dtypes(include=['int64', 'int32']).columns.tolist()
xTrain[ints] = xTrain[ints].apply(pd.to_numeric, downcast='integer')

ints = xTest.select_dtypes(include=['int64', 'int32']).columns.tolist()
xTest[ints] = xTest[ints].apply(pd.to_numeric, downcast='integer')

floats = xTrain.select_dtypes(include=['float']).columns.tolist()
xTrain[floats] = xTrain[floats].apply(pd.to_numeric, downcast='float')

floats = xTest.select_dtypes(include=['float']).columns.tolist()
xTest[floats] = xTest[floats].apply(pd.to_numeric, downcast='float')



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
sns.catplot(data=myData, y='volatilityUp', kind='count')
plt.title('Count of number of days where volatility is up')
plt.show()

# how often does Trump tweet per day?
textCountDF = myData.resample('D').apply({'text' : 'count'})
plt.style.use('ggplot')
plt.plot(textCountDF.text, c='blue')
plt.title('Count of Trump\'s tweets by day 2015-2017\nAn average of ' +
          str(round(textCountDF['text'].mean(), 1)) +
          ' tweets per day')
plt.xlabel('Date')
plt.ylabel('Count of Tweets')
plt.show()

# a probability distribution of the frequency of trumps tweets per day
plt.hist(textCountDF['text'], bins=textCountDF['text'].max(), density=True)
plt.title('PDF of count of Trump\'s tweets per day')
plt.ylabel('PDF')
plt.xlabel('Count of tweets')
plt.show()

# histogram of the number of all caps words per tweet
plt.hist(myData['allCaps'], bins=myData['allCaps'].max(), density=True)
plt.title('Probability distribution of the number of all-caps words per Trump tweet')
plt.xlabel('Number of all caps words per Tweet')
plt.ylabel('PDF')
plt.show()

# histogram of the number of exclamation points per tweet
plt.hist(myData['exclamationPoints'], bins=myData['exclamationPoints'].max(), density=True)
plt.title('Probability distribution of the number of exclamation points per Trump tweet')
plt.xlabel('Number of exclamation points per Tweet')
plt.ylabel('PDF')
plt.show()

# histogram of the number of hashtags per tweet
plt.hist(myData['hashtags'], bins=myData['hashtags'].max(), density=True)
plt.title('Probability distribution of the number of hashtags per Trump tweet')
plt.xlabel('Number of all hashtags per Tweet')
plt.ylabel('PDF')
plt.show()

# histogram of the number of userHandles per tweet
plt.hist(myData['userHandleCount'], bins=myData['userHandleCount'].max(), density=True)
plt.title('Probability distribution of the number of user handle mentions per Trump tweet')
plt.xlabel('Number of user handle mentions per Tweet')
plt.ylabel('PDF')
plt.show()

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
                       color_func=trumpColors).generate_from_frequencies(allTextCount)
plt.imshow(tweetCloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most popular words in Trump's Tweets")
plt.show()

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
                       contour_color='orange').generate_from_frequencies(allTextNounCount)

# add colors
trumpColors = ImageColorGenerator(trumpColors)
tweetCloud.recolor(color_func=trumpColors)
plt.imshow(tweetCloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most popular nouns in Trump's tweets")
plt.show()




###########################################
###########################################
##### Model Building ######################
###########################################
###########################################




###############################
#### Initiate Dask Cluster ####
###############################


# since we are using a pretty huge sparse matrix,
# we are going to take advantage of parallel programming by using Dask
#if __name__ == "__main__":
    # Create a local cluster for Dask with as many workers as cores
    #cluster = LocalCluster()
    # create client
   # client = Client(cluster)

# now let's convert our pandas dataFrames to Dask dataFrames
xTrain = dd.from_pandas(xTrain, npartitions=multiprocessing.cpu_count() * 2)
xTest = dd.from_pandas(xTest, npartitions=multiprocessing.cpu_count() * 2)
yTrain = dd.from_pandas(yTrain, npartitions=multiprocessing.cpu_count() * 2)
yTest = dd.from_pandas(yTest, npartitions=multiprocessing.cpu_count() * 2)


######################################
# Baseline Logistic regression model #
######################################


# we are first going to build a baseline logistic regression
# model that predicts whether volatility goes up or down.

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', daskStandardScaler()),
                 ('logReg', ParallelPostFit(estimator=LogisticRegression(max_iter=10000)))]

# establish a pipeline object that will perform the above steps
pipeline = Pipeline(pipelineSteps)

# create our baseline logistic regression model
baselineLogReg = pipeline.fit(xTrain, yTrain)

# create our prediction data
yPred = baselineLogReg.predict(xTest)

# create our predicted probabilities
yPredProb = baselineLogReg.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest.compute(), yPredProb.compute())

# score our baseline logistic regression model
baselineLogRegScore = accuracy_score(yTest.compute(), yPred.compute())

# create our ROC AUC score
rocAuc = roc_auc_score(yTest.compute(), yPredProb.compute())

print("\nBASELINE LOGISTIC REGRESSION\nACCURACY SCORE:\n")
print(baselineLogRegScore)
print("\nBASELINE LOGISTIC REGRESSION\nAUC SCORES:\n")
print(rocAuc)
print("\nBASELINE LOGISTIC REGRESSION\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest.compute(), yPred.compute()))
print("\nBASELINE LOGISTIC REGRESSION\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest.compute(), yPred.compute()))

# plot ROC curve for our model
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Baseline Logistic Regression')
plt.show()


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

# first we define a dictionary to hold our model
# names (a string of the number of components for SVD)
modelDict = dict()

# then we iterate through 1 and 1000 desired features (n_components) to see what the
# optimal number of components is for our SVD dimension reduction
for i in np.arange(100, 2000, 100):
    # here is a list of tuples for our pipeline steps.
    pipelineSteps = [('scaler', daskStandardScaler()),
                     ('svd', ParallelPostFit(TruncatedSVD(n_components=i))),
                     ('logReg', ParallelPostFit(LogisticRegression(max_iter=10000)))]
    modelDict[str(i)] = Pipeline(pipelineSteps)

# evaluate the models and store results
results, names = list(), list()
for name, model in modelDict.items():
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    scores = accuracy_score(yTest.compute(), yPred.compute())
    results.append(scores)
    names.append(name)
    print(name, scores)

# create a datframe of our results
svdResults = pd.DataFra

# plot model performance for comparison
plt.bar(results, height=results, labels=names)
plt.xticks(rotation=45)
plt.show()


##############
# Baseline Logistic Regression with SVD Dimension Reduction
# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', StandardScaler()),
                 ('svd', TruncatedSVD(n_components=100)),
                 ('logReg', LogisticRegression(max_iter=10000))]

# establish a pipeline object that will perform the above steps
pipeline = Pipeline(pipelineSteps)

# create our baseline logistic regression model
baselineLogReg = pipeline.fit(xTrain, yTrain)

# create our prediction data
yPred = baselineLogReg.predict(xTest)

# create our predicted probabilities
yPredProb = baselineLogReg.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest, yPredProb)

# score our baseline logistic regression model
baselineLogRegScore = accuracy_score(yTest, yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest, yPredProb)

print("\nBASELINE LOGISTIC REGRESSION\nACCURACY SCORE:\n")
print(baselineLogRegScore)
print("\nBASELINE LOGISTIC REGRESSION\nAUC SCORES:\n")
print(rocAuc)
print("\nBASELINE LOGISTIC REGRESSION\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest, yPred))
print("\nBASELINE LOGISTIC REGRESSION\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest, yPred))

# plot ROC curve for our model
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Baseline Logistic Regression')
plt.show()


###################################
# Tuned Logistic regression model #
###################################


# this is a logistic regression with hyperparameter tuning

# create a parameter grid for our grid search CV
paramGrid = {
    'C' : np.logspace(-4, 4, 20)
}

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', StandardScaler()),
                 ('gridSearchLogReg', GridSearchCV(LogisticRegression(max_iter=20000), param_grid=paramGrid))]

# establish a pipeline object that will perform the above steps
pipeline = Pipeline(pipelineSteps)

# create our baseline logistic regression model
tunedLogReg = pipeline.fit(xTrain, yTrain)

# create our prediction data
yPred = tunedLogReg.predict(xTest)

# create our predicted probabilities
yPredProb = tunedLogReg.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest, yPredProb)

# score our baseline logistic regression model
tunedLogRegScore = accuracy_score(yTest, yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest, yPredProb)

# Print the tuned parameters and score
print("TUNED LOGISTIC REGRESSION HYPERPARAMETERS: {}".format(tunedLogReg
                                                             .named_steps['gridSearchLogReg']
                                                             .best_params_))
print("Best score is {}".format(tunedLogReg
                                .named_steps['gridSearchLogReg']
                                .best_score_))
print("\nTUNED LOGISTIC REGRESSION\nACCURACY SCORE:\n")
print(tunedLogRegScore)
print("\nTUNED LOGISTIC REGRESSION\nAUC SCORE:\n")
print(rocAuc)
print("\nTUNED LOGISTIC REGRESSION\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest, yPred))
print("\nTUNED LOGISTIC REGRESSION\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest, yPred))

# plot ROC curve for our model
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Tuned Logistic Regression')
plt.show()


###################################
# Multinomial Bayesian Classifier #
###################################


# this is a multinomial bayesian classifier with hyperparameter tuning

# create a parameter grid for our grid search CV
paramGrid = {
    'alpha' : [1, 1e-1, 1e-2]
}

# here is a list of tuples for our pipeline steps.
pipelineSteps = [('scaler', StandardScaler()),
                 ('naiveBayes', GridSearchCV(MultinomialNB(), param_grid=paramGrid))]

# establish a pipeline object that will perform the above steps
pipeline = Pipeline(pipelineSteps)

# create our baseline logistic regression model
naiveBayes = pipeline.fit(xTrain, yTrain)

# create our prediction data
yPred = naiveBayes.predict(xTest)

# create our predicted probabilities
yPredProb = naiveBayes.predict_proba(xTest)[:, 1]

# create ROC curve values from predicted probabilities
fpr, tpr, thresholds = roc_curve(yTest, yPredProb)

# score our baseline logistic regression model
naiveBayesScore = accuracy_score(yTest, yPred)

# create our ROC AUC score
rocAuc = roc_auc_score(yTest, yPredProb)

# Print the tuned parameters and score
print("TUNED NAIVE BAYES HYPERPARAMETERS: {}".format(naiveBayes
                                                             .named_steps['naiveBayes']
                                                             .best_params_))
print("Best score is {}".format(naiveBayes
                                .named_steps['naiveBayes']
                                .best_score_))
print("\nTUNED NAIVE BAYES\nACCURACY SCORE:\n")
print(naiveBayesScore)
print("\nTUNED NAIVE BAYES\nAUC SCORE:\n")
print(rocAuc)
print("\nTUNED NAIVE BAYES\nCONFUSION MATRIX:\n")
print(confusion_matrix(yTest, yPred))
print("\nTUNED NAIVE BAYES\nCLASSIFICATION REPORT:\n")
print(classification_report(yTest, yPred))

# plot ROC curve for our model
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.title('ROC for Naive Bayes')
plt.show()
