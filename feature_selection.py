import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle
import json

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('medical_insurance.csv')
df = pd.get_dummies(df, columns=['region'], dtype = int)

df.info()