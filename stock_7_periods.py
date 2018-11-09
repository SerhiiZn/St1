# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 20:15:07 2018

@author: FC
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 07:50:53 2018

@author: FC

"""
import time
import sys
#import datetime
import argparse
import os
import numpy as np
import scipy
#from scipy.stats import poisson
#import gensim
import string
import nltk as nl
#from nltk import bigrams
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from nltk.corpus import names
#from nltk.tokenize import wordpunct_tokenize 
#from nltk.stem.porter import PorterStemmer
import nltk.chunk
#from nameparser.parser import HumanName
#from nltk import pos_tag, ne_chunk
#import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#rom sklearn.cluster import KMeans
#from gensim.models import word2vec
#from sklearn import metrics
import json
from operator import itemgetter
#import difflib
#import pylab as pl
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import scale
#from scipy.sparse import csr_matrix
#from collections import OrderedDict
#from pprint import pprint
import csv 
import re
#from functools import reduce

    
    
stop_words = set(stopwords.words('english')) 

stop_words.update(['crap', 'hell', 'damn', 'shit', 'anal','bullshit', 'ass', 'screw', 'bastard', 'bitch', 'piss', 'fuck', 'cunt', 'cryptoupd','cryptocurr','etn', 'eth','blockchain', 'cryptoupdate','btcusd', 'cryptographicmonitor','altcoins','ico','dash', 'bitcoincash','litecoin', 'zcash', 'cryptocurrency', 'cryptocoin', 'ripple', 'xrp', 'buy', 'bitcoin', 'crypto', 'stellar', 'btc', 'mining', 'money', 'exchange', 'ethereum', 'stellar', 'cryptic', 'bt', 'coin', 'cryptocurrencies', 'satoshi', 'airdrop', 'ltc', 'forex', 'altcoin', '$', '#', '%', '.', ',', '"', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "'ve", 'longer', 'have', "'ll", 'have been', 'to', 'say', 'with', 'got', 'as longer', 'https:', 'if']) 

start_time = time.clock()    
                  
                


    
def extract_email_addresses(string):
    r = re.compile(r'@[\w\.-]+')
    return r.findall(string)

def extract_under_(string):
    r = re.compile(r'[\w\.-]+_[\w\.-]+')
    return r.findall(string)   

def extract_http(string):
    r = re.compile(r'http[\w\.-]+')
    return r.findall(string)
  
def extract_doll(string):
    r = re.compile(r'![\w\.-]+')
    return r.findall(string)    

def extract_imp(string):
    r = re.compile(r'&[\w\.-]+')
    return r.findall(string)    

def Tokenisation (summary_input):
    #tokenisation of sentense
    
    #global a
    #global words
    #global words0
    #global stripped
    #global table
    
    #tokens = word_tokenize(summary_input)
    
    tokens = [t for t in summary_input.split()]
    # convert to lower case

    tokens = [w.lower() for w in tokens]
    words0 = [word for word in tokens if (not extract_email_addresses(word) and not extract_under_(word) and not extract_http(word) and not extract_imp(word) and not word[0]=='$' and len(word)>3)]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words0]
   # words0 = [word for word in stripped if not extract_email_addresses(word) or not extract_under_(word)]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    #stop_words = set(stopwords.words('english'))
    
    tagged = nltk.pos_tag(words)
    length = len(tagged) - 1
    list_nouns = list()
    for i in range(0, length):
        log = (tagged [i][1][0] == 'N')
        if log == True:
           list_nouns.append(tagged [i][0])
     
    list_nouns = [w for w in list_nouns if not w in stop_words]       
    return list_nouns
    
def HASTAG(json_input):
   
  # global tg
  # global hashtags_tweet
  # global entit
  # global tokens
 
   #json_input    
   tweets=[]    
   with open(json_input, encoding='utf-8') as f:
      for line in f:
          tweets.append(json.loads(line))

#   tweet=tweets[0]
   # pull out various data from the tweets
   #tweet_id = [tweet['id'] for tweet in tweets]
   #tweet_text = [tweet['text'] for tweet in tweets]
  # tweet_time = [tweet['created_at'] for tweet in tweets]
   #tweet_teg = [tweet['object']['summary']for tweet in tweets]


   hashtags_tweet = []
   entit=tweets[0]
    
   for entit in tweets:
         if 'object' in entit:
             tokens = Tokenisation(entit['object']['summary'])
             if len(tokens)>0:
                for i in range(0, len(tokens)-1):  
                    token_list = tokens[i] 
                    hashtags_tweet.append(token_list)
             #hashtags_tweet.extend(entit['object']['summary'])
 
   
  # hashs = [tag['id'] for tag in hashtags_tweet]
    
   filttext=nl.FreqDist(hashtags_tweet)
   tg = sorted(filttext.items(), key=itemgetter(1), reverse=True)[0:round(len(filttext)*0.02)]
   sentences=dict(sorted(filttext.items(), key=itemgetter(1), reverse=True)[round(len(filttext)*0.02):(round(len(filttext)*0.4))])
   
   # CSV files with sorted hashtags and trend characteristic NDI
   
   
   def csv_reader(file_obj,tg):
       #Read a csv file
        reader = list(csv.reader(file_obj))
        i=0
        data_read1=[]
        for k in reader:
             string1 =''
             string1 = str(reader[i])[2:-2]
             data_read1.append(string1)
             i=i+1
           
        data_read2 = [row[0] for row in tg]
        mergedlist = list(set(data_read1 + data_read2))
        return mergedlist
                  
   csv_path = "STOPWORDS_stock.csv"   
   
   with open(csv_path, "r") as f_obj:
        row_n=csv_reader(f_obj,tg)
        row_n.sort()
   with open(csv_path, 'w', encoding="utf-8") as csv_file:
                 csv_writer = csv.writer(csv_file)
                 for item1 in row_n:
                        csv_writer.writerow([item1])
   
   def dictMinus(dct, val):
       global copy
       copy = dct.copy()
       if val in copy:
            del copy[val]
       return copy
    
   sentences1=sentences
   for wrd1 in sentences:
       for wrd2 in row_n:
           if wrd1.lower() == wrd2.lower():
                  sentences1=dictMinus(sentences1, wrd1)
                  
   sentences=sentences1                             
   hash_all={}  
   for wrd1 in sentences:
       sum_hash=0
       for wrd2 in sentences:
           if wrd1.lower() == wrd2.lower():
                   sum_hash = sum_hash + sentences[wrd2]
                   hash_all[wrd1.lower()]=hash_all.get(wrd2.lower(), sum_hash)
                   hash_all[wrd1.lower()]=sum_hash
                
 
 
   return hash_all


def exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # time series
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result  

# Getting the trend characteristic NDI:  if NDI > 1 then hashtags have a trend 
 
def N_DI (parametr1,parametr2,parametr3):
    series=[]
    DI=scipy.stats.poisson.interval(0.99, parametr1)
    series1=round((parametr2/((DI[1]-DI[0])/2+DI[0])),1)
    DI=scipy.stats.poisson.interval(0.99, parametr2)
    series2=round((parametr3/((DI[1]-DI[0])/2+DI[0])),1)
    series = [series1,series2]
    NDI=exponential_smoothing(series, 0.6, 0.9)
    return NDI[0],parametr1,parametr2


# Forming a dictionary "out" with hashtags data from two neighboring files and trend characteristic NDI
    
def trend (hash_period,n):
    out = {}
    for key1 in hash_period[n-3]:
        for key2 in hash_period[n-2]:
            for key3 in hash_period[n-1]:
                if key1.lower() == key2.lower() and key2.lower() == key3.lower():
                        out[key1]=out.get(key1, N_DI(hash_period[n-3][key1],hash_period[n-2][key2],hash_period[n-1][key3]))
                    
    return out              
                    

def exponential_smoothing_six(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result



def main():
    global prnt_out
    i=0
    
    hash_period=[]
    if (len(sys.argv)-1)<3:
        print ("Error - number of JSON files have to 3  ")
        sys.exit()


    for i in range (len(sys.argv)-1):
        parametr = sys.argv[i+1:i+2]
        name=parametr[0]     
        hash_period.append(HASTAG(name))
    #  tweet_text.append(HASTAG(name)[1])
        i=i+1
  

    if (len(sys.argv)-1)>2:
          out1=trend(hash_period,3)
          prnt_out= list(zip(out1.keys(),out1.values()))
          prnt_out.sort(key=lambda elem: elem[1], reverse=True)
# CSV files with sorted hashtags and trend characteristic NDI
          with open('stocktwits_trend_words_3h.csv', 'w', encoding="utf-8") as csv_file:
                 csv_writer = csv.writer(csv_file)
                 csv_writer.writerow(["Word", " Trend 3h"])
                 for item1 in prnt_out:
                      csv_writer.writerow(*[item1])
    else: print ("Number of JSON files have to 3  ")

    if (len(sys.argv)-1)>4:
           out2=trend(hash_period,6)
           prnt_out= list(zip(out2.keys(),out2.values()))
           prnt_out.sort(key=lambda elem: elem[1], reverse=True)
# CSV files with sorted hashtags and trend characteristic NDI         
           with open('stocktwits_trend_words_next_3h.csv', 'w', encoding="utf-8") as csv_file:
                  csv_writer = csv.writer(csv_file)
                  csv_writer.writerow(["Word", " Trend next 3h"])
                  for item1 in prnt_out:
                       csv_writer.writerow(*[item1])
           out_list={}
           for key1 in out1:
                  for key2 in out2:
                           if key1.lower() == key2.lower():
                                 out_list[key1]=out_list.get(key1, round(exponential_smoothing_six([out1[key1][0],out2[key2][0]], 0.7)[1],1))
           prnt_out= list(zip(out_list.keys(),out_list.values()))
           prnt_out.sort(key=lambda elem: elem[1], reverse=True)
# CSV file of 6h trend        
           with open('stocktwits_trend_words_6h.csv', 'w', encoding="utf-8") as csv_file:
                 csv_writer = csv.writer(csv_file)
                 csv_writer.writerow(["Word", " Trend for 6h"])
                 for item1 in prnt_out:
                      csv_writer.writerow(*[item1])
    else: print ("Number of JSON files have to 6  ")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Produce TOP LIST")
    parser.add_argument('-i','--input-files',dest='input_files',nargs="+", default=None, help="input files; if unspecified, stdin is used")

    main()
 #   print (time.clock() - start_time)

      
