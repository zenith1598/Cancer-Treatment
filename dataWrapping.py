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

data_outcome = pd.read_csv("C:\Shashank Reddy\Outcome.csv",sep='\s*,\s*',header=0, encoding='ascii', engine='python')



#print(data_outcome)

data_outcome = data_outcome.fillna("zero")





# outcome dictionary
outcome = {'removed': 1, 'not removed': 2, 'retrieval': 3, 'non retrieval': 4, 'zero' : 0}
data_outcome["Outcome"] = [outcome[item] for item in data_outcome["Outcome"]]

list_outcome = pd.DataFrame(list(data_outcome["Outcome"]))

list_outcome.to_csv(r"C:\Shashank Reddy\FinalOutcome.csv",sep='\t', index=False)

#print(data_outcome)




#data_outcome.to_dense().to_csv(r"C:\Shashank Reddy\FinalOutcome.csv")


sessilelocation = pd.read_csv("C:\Shashank Reddy\SessileLocation.csv",sep='\s*,\s*',header=0, encoding='ascii', engine='python').fillna("zero")
#print(sessilelocation.columns.tolist())
#print(sessilelocation)

location = {'cecal': 1, 'ascending': 2, 'ileum': 3, 'ileocecal': 3, 'hepatic': 4, 'transverse': 5, 'splenic': 6, 'descending': 7, 'sigmoid': 8, 'recto-sigmoid': 9, 'rectal': 10, 'appendix': 11,'zero': 0}
sessilelocation["PositionA"] = [location[item] for item in sessilelocation["PositionA"]]
sessilelocation["PositionB"] = [location[item] for item in sessilelocation["PositionB"]]
sessilelocation["PositionC"] = [location[item] for item in sessilelocation["PositionC"]]
sessilelocation["PositionD"] = [location[item] for item in sessilelocation["PositionD"]]
sessilelocation["PositionE"] = [location[item] for item in sessilelocation["PositionE"]]
sessilelocation["PositionF"] = [location[item] for item in sessilelocation["PositionF"]]
sessilelocation["PositionG"] = [location[item] for item in sessilelocation["PositionG"]]

#print(sessilelocation)


sessileshape = pd.read_csv("C:\Shashank Reddy\SessileShape.csv",sep='\s*,\s*',header=0, encoding='ascii', engine='python').fillna("zero")
#print(sessileshape)
shape = {'zero':0,'sessile':1,'pedunculated':2,'flat':3,'mass':4,'smooth':5,'serrated':6}

sessileshape["Shape"] = [shape[item] for item in sessileshape["Shape"]]
#print(sessileshape)


sessilesize = pd.read_csv("C:\Shashank Reddy\SessileSize.csv",sep='\s*,\s*',header=0, encoding='ascii', engine='python').fillna("zero")
#print(sessilesize)

size = {'zero':0,'diminutive':1,'small':2,'medium':3,'large':4}
sessilesize["Size"] = [size[item] for item in sessilesize["Size"]]
#print(sessilesize)


sessileside = pd.read_csv("C:\Shashank Reddy\Sides.csv",sep='\s*,\s*',header=0, encoding='ascii', engine='python').fillna("zero")
#print(sessileside)

side = {'zero':0,'left':1,'right':2}
sessileside["Sides"] = [side[item] for item in sessileside["Sides"]]
#print(sessileside)


cancer_treatment = pd.read_csv("C:\Shashank Reddy\Treatment.csv",sep='\s*,\s*',header=0, encoding='ascii', engine='python').fillna("zero")
#print(cancer_treatment)

treatment = {'zero':0,'piermeal':1,'cold snare':2,'hot snare':3,'snare':4,'electocautery snare':5,'excisional biopsy':6,'biopsy forcep':7,'cold biopsy':8}
cancer_treatment["Treatment"] = [treatment[item] for item in cancer_treatment["Treatment"]]

list_treatment = pd.DataFrame(list(cancer_treatment["Treatment"]))

list_treatment.to_csv(r"C:\Shashank Reddy\FinalTreatment.csv",sep='\t', index=False)
#print(cancer_treatment)


sessile_number = pd.read_csv("C:\Shashank Reddy\SessileNumber.csv",sep='\s*,\s*',header=0, encoding='ascii', engine='python').fillna("zero")
#print(sessile_number)

number = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':7,'eight':8,'nine':9,'ten':10}

sessile_number["Number1"] = [number[item] for item in sessile_number["Number1"]]
sessile_number["Number2"] = [number[item] for item in sessile_number["Number2"]]
sessile_number["Number3"] = [number[item] for item in sessile_number["Number3"]]
sessile_number["Number4"] = [number[item] for item in sessile_number["Number4"]]

#print(sessile_number)

#************************************* Data Union *********************************************************************************

list_mat1= pd.DataFrame(list(sessilelocation["PositionA"]))
list_mat2= pd.DataFrame(list(sessilelocation["PositionB"]))
list_mat3= pd.DataFrame(list(sessilelocation["PositionC"]))
list_mat4= pd.DataFrame(list(sessilelocation["PositionD"]))
list_mat5= pd.DataFrame(list(sessilelocation["PositionE"]))
list_mat6= pd.DataFrame(list(sessilelocation["PositionF"]))
list_mat7= pd.DataFrame(list(sessilelocation["PositionG"]))

list_mat8= pd.DataFrame(list(sessileshape["Shape"]))
list_mat9= pd.DataFrame(list(sessilesize["Size"]))
list_mat10= pd.DataFrame(list(sessileside["Sides"]))

list_mat11= pd.DataFrame(list(sessile_number["Number1"]))
list_mat12= pd.DataFrame(list(sessile_number["Number2"]))
list_mat13= pd.DataFrame(list(sessile_number["Number3"]))
list_mat14= pd.DataFrame(list(sessile_number["Number4"]))


Final_Data = pd.concat([list_mat1,list_mat2,list_mat3,list_mat4,list_mat5,list_mat6,list_mat7,list_mat8,list_mat9,list_mat10,list_mat11
                        ,list_mat12,list_mat13,list_mat14],axis = 1)


print(Final_Data)


Final_Data.to_csv(r"C:\Shashank Reddy\DataSet_Final.csv",index = False)


#print(Final_Data)







