import time
import json
import numpy as np
import pandas as pd

# extract data from npy to csv

source = np.load("/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_constant_wa.npy")
# data = (10 seconds * 100Hz) + Time + HR + RR + SV + TPR + MAP. 

# extract the data from the source into one csv file
df = pd.DataFrame(source)
df.to_csv("/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_constant_wa.csv", index=False)
