# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:20:27 2018

@author: Wesley
"""

import pandas
# from pattern.en import sentiment
# import HTMLParser
import re
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import string
from collections import OrderedDict
from nltk import bigrams
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
# import plotly.plotly as py

# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
import requests
from bs4 import BeautifulSoup
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
import os


os.getcwd()
#os.chdir(r'D:\Data Science\Text for Findings')


data = pd.read_csv("C:\Shashank Reddy\DataSet.csv", encoding="ISO-8859-1")
print(data.count())
data = data[data.columns[0:8]]
# data = data.dropna(subset=['Random_ID','Year','Month','dt','Indications','Findings'])
# data = data.dropna(subset=['Findings'])
#print(data.count())

# print(data)
data['PredictionColumn'] = data["Findings"].map(str)



print(data.head(1))
print(data.count())
# data = data.dropna(subset=['PredictionColumn'])
# print(data.count())
# data['PredictionColumn'].to_csv(r'D:\Data Science\Cancer dataset', header=None, index=None, sep=' ')

list_data = list(data['PredictionColumn'])




type(list_data)
print(list_data[0:5])

from collections import Counter

Counter(list_data)
# Remove Nan from the list, otherwise it throws error while doing further operations
# cleanedList = [x for x in list_data if str(x) != 'nan']
Counter()

# Regex Cleansing of data - Links,Hashtags,UserTags,ReTweet tags,Converting emoticons,Punctuation,Parenthesis
Cleansed_data = []
for j in list_data:
    Special_chars = re.sub(r'[\-\!\@\#\$\%\^\&\*\(\)\_\+\[\]\;\'\.\,\/\{\}\:\"\<\>\?\|]', '', j)
    lower = Special_chars.lower()
    Widespace = lower.strip()
    Cleansed_data.append(Widespace)
print(Cleansed_data[0:5])
print(len(Cleansed_data))

data1 = []
words = ['piermeal', 'cold Snare', 'hot Snare', 'snare', 'electocautery snare', 'excisional biopsy', 'biopsy forcep',
         'cold biopsy', 'resection', 'removed', 'not removed', 'retrieval', 'not retrieval']

for w in Cleansed_data:
    tokens = re.findall(
        'piermeal|cold Snare|hot Snare|snare|electocautery snare|excisional biopsy|biopsy forcep|cold biopsy',
        w)
    data1.append(tokens)
print("DATA 1**************************************")
#print(data1[0:5])
print(len(data1))
print(data1)
data_mat1 = pd.DataFrame(data1)
data_mat1.to_dense().to_csv(r"C:\Shashank Reddy\1.csv")


data2 = []

for w in Cleansed_data:
    tokens = re.findall('removed|not removed|retrieval|non retrieval',w)
    data2.append(tokens)

data_mat2 = pd.DataFrame(data2)
data_mat2.to_dense().to_csv(r"C:\Shashank Reddy\2.csv")


data3 = []

for w in Cleansed_data:
    tokens = re.findall('right|left',w)
    data3.append(tokens)
data_mat3 = pd.DataFrame(data3)
data_mat3.to_dense().to_csv(r"C:\Shashank Reddy\3.csv")

data4 = []
for w in Cleansed_data:
    tokens = re.findall('cecal|ascending|ileum|ileocecal|hepatic|transverse|splenic|descending|sigmoid|recto-sigmoid|rectal|appendix|Cecum',w)
    data4.append(tokens)
data_mat4 = pd.DataFrame(data4)
data_mat4.to_dense().to_csv(r"C:\Shashank Reddy\4.csv")


data5 =[]
for w in Cleansed_data:
    tokens = re.findall('sessile|pedunculated|flat|mass|smooth|serrated',w)
    data5.append(tokens)
data_mat5 = pd.DataFrame(data5)
data_mat5.to_dense().to_csv(r"C:\Shashank Reddy\5.csv")


data6 =[]
for w in Cleansed_data:
    tokens = re.findall('small|medium|large|diminutive',w)
    data6.append(tokens)
data_mat6 = pd.DataFrame(data6)
data_mat6.to_dense().to_csv(r"C:\Shashank Reddy\6.csv")


data7 =[]

for w in Cleansed_data:
    tokens = re.findall('one|two|three|four|five|six|seven|eight|nine|ten',w)
    data7.append(tokens)
data_mat7 = pd.DataFrame(data7)
data_mat7.to_dense().to_csv(r"C:\Shashank Reddy\SessileNumber.csv")















#df = pd.DataFrame(Cleansed_data)
#Full_data = pd.concat([data['Random_ID'], df, data_mat], axis=1)

# data.to_excel("CountVectorizerOutput.xls",index=False)
#Full_data.to_dense().to_csv(r"C:\Shashank Reddy\DataSet_Final.csv", sep='\t', index=False)
