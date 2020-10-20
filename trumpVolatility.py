###############################################
###############################################
###############################################
### TITLE: trumpVolatility ####################
### AUTHOR: Miguel Sebastian de la Mata #######
### DATE: October 3rd, 2020 ###################
###############################################
###############################################
###############################################


# The purpose of this project is to gage what affect, if any,
# Trump's tweets may have on market volatility

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contractions
from nltk import TweetTokenizer
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image




##################
# Data Gathering #
##################

# json files sourced from https://github.com/bpb27/trump_tweet_data_archive
df2015 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2015.json")
df2016 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2016.json")
df2017 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2017.json")
df2018 = pd.read_json(r"C:\Users\sebid\OneDrive\Desktop\master_2018.json")



####################
# Data Cleaning ####
####################



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
                     'entities'], inplace=True)
    df['id_str'] = df['id_str'].astype('str')
    df['in_reply_to_user_id_str'] = df['in_reply_to_user_id_str'].astype('str')
    df['in_reply_to_status_id_str'] = df['in_reply_to_status_id_str'].astype('str')
    df['quoted_status_id_str'] = df['quoted_status_id_str'].astype('str')

# now let's append our dataframes to make a big one
myData = df2015.append(dfList[1:]).sort_index()
myData.drop_duplicates(inplace=True)

# June 16th 2015 is the day Trump announced his campaign. Let's start here
myData = myData.loc['2015-06-16':]

# refine our data to just texts that are filled in (no retweets) here
myData.dropna(subset=['text'], inplace=True)

# let's drop all rows with text in the format "@USERNAME:"
# these are other people tweeting Trump, not his direct tweets
myData = myData[~myData['text'].str.contains(r"\@.*:")]

# we want to create a variable that counts the number of all caps words in a tweet
# this might be an indication of strong emotional, or more volatile behavior
# we'll do this with a regex search for all all caps words
allCaps = r'\b[A-Z]+\b'
myData['allCaps'] = myData['text'].str.count(allCaps)
print(myData['allCaps'].max())

# let's make another variable that is the number of words in a tweet
myData['tweetWordCount'] = myData['text'].str.count(r'\b[A-Za-z]+\b')

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
myData.drop(columns='noContractions', inplace=True)

# perform bag of words for topic discovery
# first let's convert everything to lowercase
# each column entry contains a list of the tokens so we need to make a list comprehension
# then apply it to the column, either that or there is some more straightforward way I am totally missing
def listLowerer(list):
    return [token.lower() for token in list]

myData['loweredTokens'] = myData['tokenizedTweets'].apply(listLowerer)
myData.drop(columns='tokenizedTweets', inplace=True)
print(myData['loweredTokens'].head())

# we also want to remove our stop words from the list. I'm just going to define
# another list comprehension then apply that function to the column
# note  that in order to use stop words you must first download the nltk data.
# If you haven't already, go to the Python console and enter "nltk.download()"
def listNoStopWords(list):
    return[token for token in list if token not in stopwords.words('english')]

myData['noStopWordsTokens'] = myData['loweredTokens'].apply(listNoStopWords)
myData.drop(columns='loweredTokens', inplace=True)
print(myData['noStopWordsTokens'].head())

# pretty much going to do the same thing for punctuation as we did right above
def listNoPunct(list):
    return [token for token in list if token not in string.punctuation]

myData['noPunctTokens'] = myData['noStopWordsTokens'].apply(listNoPunct)
myData.drop(columns='noStopWordsTokens', inplace=True)

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
myData.drop(columns='noPunctTokens', inplace=True)
print(myData['lemmatizedTokens'].head())

# we can build a list that contains all of our words in the entire column here
tokenList = []
tokenList.extend(myData['lemmatizedTokens'])
# this is a list of lists so I want to make it into a single list here
tokenList = [item for list in tokenList for item in list]
print(tokenList[0:100])

# lets create a list of just the nouns to see what subjects get mentioned the most
tokensNouns = pos_tag(tokenList)
tokensNouns = [token[0] for token in tokensNouns if token[1] == 'NN']
print(tokensNouns[0:10])

# build a counter for a bag of words approach
print(myData['lemmatizedTokens'].apply(Counter))

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
# instead let's do the top 200 most mentioned nouns to make our dataframe more manageable
popularNounsList = [word for word in allTextNounCountMeltedDF.nlargest(250, 'Count')['Word']]

print("Firsts ten entries to popularNounsList: " + str(popularNounsList[0:10]))

# I want to create a column for every word in all the tweets and
# count how many times it occurs in that tweet
def countWords(list, word):
    return list.count(word)

for word in popularNounsList:
    myData['newCol'] = myData['lemmatizedTokens'].apply(countWords, args=(word, ))
    myData.rename(columns={'newCol' : word}, inplace=True)
print(myData.columns)
print(myData['clinton'].head(10))




################################
## Exploratory Data Analysis ###
################################



# data dictionary for our json files is located here:
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/overview/tweet-object
#


myData.info()

# some exploratory questions:

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
plt.hist(textCountDF.text, bins=40, density=True)
plt.title('PDF of count of Trump\'s tweets per day')
plt.ylabel('PDF')
plt.xlabel('Count of tweets')
plt.show()

# histogram of the number of all caps words per tweet
plt.hist(myData['allCaps'], bins=30, density=True)
plt.title('Probability distribution of the number of all-caps words per Trump tweet')
plt.xlabel('Number of all caps words per Tweet')
plt.ylabel('PDF')
plt.xticks(np.arange(0, 30))
plt.show()

# let's look at the most common words used in tweets
# there are obviously a lot of unique words used so let's keep it to like 25ish
plt.bar(allTextCountMeltedDF['Word'].head(n=20), allTextCountMeltedDF['Count'].head(n=20))
plt.title("Frequency of all words used in Trump's tweets")
plt.ylabel('Count')
plt.xlabel('Word')
plt.show()

# let's look at the most common nouns used in tweets
# there are obviously a lot of unique words used so let's keep it to like 25ish
plt.bar(allTextNounCountMeltedDF['Word'].head(n=20), allTextNounCountMeltedDF['Count'].head(n=20))
plt.title("Frequency of all nouns used in Trump's tweets")
plt.ylabel('Count')
plt.xlabel('Word')
plt.show()

# let's make our word clouds in the shape of Trump's head because, why not?
# I found a free png here: https://www.freeimg.net/photo/868386/trump-donaldtrump-president-usa
# I downloaded it, now I will direct it to that path
# tutorial on how to fit a custom mask here: https://www.datacamp.com/community/tutorials/wordcloud-python
imagePath = r"C:\Users\sebid\OneDrive\Desktop\trump-2069581_1280.png"

trumpMask = np.array(Image.open(imagePath))
trumpColors = ImageColorGenerator(trumpMask)
print(trumpMask)


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

# let's do a wordcloud of the nouns used by Trump's tweets
tweetCloud = WordCloud(background_color='gray',
                       max_words=1000,
                       mask=trumpMask,
                       contour_width=1,
                       contour_color='orange',
                       color_func=trumpColors).generate_from_frequencies(allTextNounCount)
plt.imshow(tweetCloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most popular nouns in Trump's tweets")
plt.show()
