import numpy as np
import pandas as pd

data1 = pd.read_csv("submission00.csv")
data2 = pd.read_csv("submission01.csv")
data = data1.copy()
data['TARGET'] = (data1['TARGET'] + data2['TARGET']) / 2

data.to_csv('Blend.csv', index = False)