import pandas as pd

df = pd.read_csv('/tf/ship_detect_tl/scripts/train/train_labels.csv')

sizes = pd.DataFrame(columns = ['xSize' , 'ySize'])

for index, row in df.iterrows():
    new_row = pd.DataFrame.from_dict({'xSize': [row['xmax']-row['xmin']], 'ySize': [row['ymax']-row['ymin']]})
    sizes = pd.concat([sizes, new_row],  ignore_index=True)

print(sizes.astype(int).describe())

