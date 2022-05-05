import pandas as pd 
from tqdm import tqdm

df = pd.read_csv('/tf/ship_data/train_ship_segmentations_OD.csv')

df_ship = df[df.xmax == df.xmax] # dataframe des images avec bateau
df_no_ship = df[df.xmax != df.xmax] # dataframe des images sans bateau

nb_images = 100
boat_rate = 0.8
cut_rate = 0.5

df_train = pd.DataFrame(columns=df.columns.tolist())
df_test = pd.DataFrame(columns=df.columns.tolist())

i, j = 0, 0
print('Création des dataframes à partir des images avec bateau.')
for image_name in tqdm(df_ship['filename'].unique()[:int(nb_images*boat_rate)]):
    if i < int(cut_rate*int(nb_images*boat_rate)):
        image_df = df_ship[df_ship.filename == image_name]
        df_train = pd.concat([df_train,image_df], ignore_index = True)
        i+=1
    else :
        image_df = df_ship[df_ship.filename == image_name]
        df_test = pd.concat([df_test,image_df], ignore_index = True)

print('Création des dataframes à partir des images sans bateau.')
for image_name in tqdm(df_no_ship['filename'].unique()[:nb_images-int(nb_images*boat_rate)]):
    if j < int(cut_rate*(nb_images-int(nb_images*boat_rate))):
        image_df = df_no_ship[df_no_ship.filename == image_name]
        df_train = pd.concat([df_train,image_df], ignore_index = True)
        j+=1
    else :
        image_df = df_no_ship[df_no_ship.filename == image_name]
        df_test = pd.concat([df_test,image_df], ignore_index = True)

# analyser dataframe :
def analyser_df(df):
    nb_total = len(df['filename'].unique())
    avec_bateau = nb_total- len(df[df.xmax != df.xmax])
    print(("nombre d'images : {}").format(nb_total))
    print('pourcentage avec bateau : {}'.format(avec_bateau/nb_total))

analyser_df(df_test)
analyser_df(df_train)

filename = 'train_'+str(nb_images)+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord'
print(filename)

