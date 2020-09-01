import csv, sys
from statistics import mean

# Function to find majority element 
from collections import Counter 

'''with open(sys.argv[1], newline='') as f:
    results = csv.reader(f, delimiter=";")
    avg_cluster=[]
    for row in results:
        pri=list(map(float,row[1:]))
        avg_cluster.append(pri.index(max(pri))+1)
    print(Counter(avg_cluster))'''
    
import pandas as pd
df = pd.read_csv(sys.argv[1]).drop("image",axis=1)
dfcols=df[["3","8","12","19","24"]]
print(dfcols.mean())
print(mean(dfcols.max(axis=1)))
