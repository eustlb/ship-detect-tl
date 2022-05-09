import pandas as pd
from tqdm import tqdm

df = pd.read_csv('/tf/ship_data/train_ship_segmentations_OD.csv')

df_ship = df[df.xmax == df.xmax] # dataframe des images avec bateau

sizes = pd.DataFrame(columns = ['filename','W' , 'H'])

for index, row in tqdm(df_ship.iterrows(), total=df_ship.shape[0]):
    new_row = pd.DataFrame.from_dict({'filename': row['filename'], 'W': [row['xmax']-row['xmin']], 'H': [row['ymax']-row['ymin']]})
    sizes = pd.concat([sizes, new_row],  ignore_index=True)

print(sizes.astype(int).describe())

