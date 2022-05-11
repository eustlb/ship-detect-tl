import pandas as pd
import numpy as np

def generate_od_csv_boats(path_original_csv, path_new_csv):

    df = pd.read_csv(path_original_csv)
    
    indexNames = df[df['xmax'] != df['xmax']].index
    df.drop(indexNames, inplace=True)
    del df['filename']
    del df['class']
    print(df.head())

    df.to_csv(path_new_csv, index=False)

if __name__ == "__main__":
    generate_od_csv_boats('/tf/ship_data/train_ship_segmentations_OD.csv', '/tf/ship_data/train_ship_segmentations_OD_boats.csv')

