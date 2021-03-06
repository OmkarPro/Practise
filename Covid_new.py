import numpy as np
import pandas as pd
import warnings
import pickle
import seaborn as sns #for Plotting Graphs
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

names=["Gender","Age","S1_Cough","S2_Fever","S3_Dyspnea","pO2","Temperature","Contact_with_COVID_positive_patient","Swab_test_result","Risk_of_contracting_virus"]
dataframe = pd.read_csv('covid_new.csv', names=names)
array = dataframe.values
X = array[1:,0:9]
y = array[1:,-1]

from imblearn.over_sampling import RandomOverSampler

os = RandomOverSampler(random_state=0)
X_train_res,y_train_res = os.fit_resample(X,y)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_train_res, y_train_res, test_size=0.3, random_state=1)

print(X_train_res[-5])

abc = AdaBoostClassifier()
abc.fit(X_train_res, y_train_res)

pickle.dump(abc,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))