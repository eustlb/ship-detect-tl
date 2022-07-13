import pandas as pd
from tqdm import tqdm

def imgs_per_b_csv(path_hash_csv, path_new_csv):
    """
    Genère le CSV de toutes les images qui contiennent un certain bateau (identifié par son hash).
    """
    df_hash = pd.read_csv(path_hash_csv)
    boats = df_hash['BoatHash'].unique()
    print(len(boats))
    boats_h_dict = {'BoatHash':[], 'ImageIds':[]}
    for boat in tqdm(boats) : 
        imgs = [img for img in df_hash[df_hash.BoatHash == boat]['ImageId']]
        imgs_str = ' '.join(imgs)
        boats_h_dict['BoatHash'].append(boat)
        boats_h_dict['ImageIds'].append(imgs_str)
    df = pd.DataFrame.from_dict(boats_h_dict)
    pd.DataFrame.to_csv(df, path_new_csv)

if __name__=='__main__':
    print()