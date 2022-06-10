import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def draw_distrib(path_to_csv, nb_bar, saving_path):
    """
    Crée un diagramme à barres (format pdf) de la distribution des largeurs et hateurs des bateaux sur les images.

    :param path_to_csv: str, path du csv pascal VOC des images à analyser.
    :param nb_bar: int, nombre de barres du diagramme.
    :param saving_dir: str, path où sera enregistré le pdf.
    :return: Void.
    """

    df = pd.read_csv(path_to_csv)

    df_ship = df[df.xmax == df.xmax] # dataframe des images avec bateau

    sizes = pd.DataFrame(columns = ['filename','W' , 'H'])

    for index, row in tqdm(df_ship.iterrows(), total=df_ship.shape[0]):
        new_row = pd.DataFrame.from_dict({'filename': row['filename'], 'W': [row['xmax']-row['xmin']], 'H': [row['ymax']-row['ymin']]})
        sizes = pd.concat([sizes, new_row],  ignore_index=True)
        
    bar_width = 768 // nb_bar
    counts_w = np.zeros(nb_bar)
    counts_h = np.zeros(nb_bar)
    
    for index, row in tqdm(sizes.iterrows(), total=sizes.shape[0]):
        counts_w[int(row['W']//bar_width)]+=1
        counts_h[int(row['H']//bar_width)]+=1
        
    x = np.linspace(bar_width/2,768+bar_width/2,nb_bar,endpoint=False)
    
    fig, ax = plt.subplots(2, 1, figsize=(10,12))
    plt.subplots_adjust(hspace=0.25)
    
    ax[0].set_xticks(np.arange(0, 776, 25))
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].set_xlabel('taille en pixels')
    ax[0].set_ylabel("Nombre de bateaux")
    ax[0].set_title('Distribution de la largeur des bateaux sur les images')
    ax[0].bar(x, counts_w, bar_width, color='#6495ed' )
    
    ax[1].set_xticks(np.arange(0, 776, 25))
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].set_xlabel('taille en pixels')
    ax[1].set_ylabel("Nombre de bateaux")
    ax[1].set_title('Distribution de la hauteur des bateaux sur les images')
    ax[1].bar(x, counts_h, bar_width, color='#6495ed' )
    
    plt.savefig(saving_path+'_'+str(nb_bar)+'.pdf')