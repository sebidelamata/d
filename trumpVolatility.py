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
import tweepy
import json


##################
# Data Gathering #
##################


# first we want to import trump's sweet, sweet tweets
# to do this we will use the tweepy library.
# in order to gain access to Twitter's REST API,
# we first need to go to twitter apps and register
# for a developer account before adding a new app to our profile.

# next we need to authenticate our API.
# I have stored this info in a csv on the desktop. When you get your API
# keys for this project, create a csv with column titles on one line
# and values on the next.
APIkeys = pd.read_csv(r'C:\Users\sebid\OneDrive\Desktop\trumpVolatilityTwitterAPIKey.csv')
print(APIkeys.columns)
print(APIkeys['API_KEY'][0])

# first we create an instance of the authentication handler with our
# consumer API key and consumer secret API key
auth = tweepy.OAuthHandler(APIkeys['API_KEY'][0],
                             APIkeys['API_SECRET_KEY'][0])
# next we set our access token
auth.set_access_token(APIkeys['ACCESS_TOKEN'][0],
                      APIkeys['ACCESS_TOKEN_SECRET'][0])
# now that we are authenticated we can instantiate our API
api = tweepy.API(auth)
# lets test our authentication
try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

# initialize our donnyList
donnyDF = pd.DataFrame()

# now we can grab our donny-tweets
for tweet in tweepy.Cursor(api.user_timeline,id='realDonaldTrump').items(5):
    twit = tweet._json
    donnyDF = donnyDF.append(twit, ignore_index=True)

print(donnyDF.info())
