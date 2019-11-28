import glob
import os
import sys
import pandas as pd

sys.maxsize = 2147483647

in_path = 'C:/Users/sglvladi/Documents/TrackAnalytics/data/exact_earth/id'
out_path = 'C:/Users/sglvladi/Documents/TrackAnalytics/data/exact_earth/id_sorted'
for file_path in glob.iglob(os.path.join(in_path, r'*.csv')):
    print(file_path)
    filename = os.path.basename(file_path)
    print(filename)

    (pd.read_csv(file_path, low_memory=False)
     .sort_values(['Time', 'Millisecond'])
     .to_csv(os.path.join(out_path, filename), index=False))