import os 
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageDraw
import pandas as pd

def save_image_bbox(img_path, saving_path, bboxs, scores):
    """
    Dessine sur une image des bboxs avec leurs scores.

    :param img_path: str, path de l'image à éditer
    :param saving_path: str, path où sera enregistrée l'image 
    :param bboxs: list, liste de bboxs au format [[ymin_rate, xmin_rate, ymax_rate, xmax_rate]] (valeurs relatives : xmin_rate=xmin/w)
    :param scores: list, liste des scores au format [0.8]
    :return: Void.
    """
    image = Image.open(img_path)
    h, w = image.size
    draw = ImageDraw.Draw(image)
    for i in range(len(bboxs)) :
        x0 = bboxs[i][1]*w
        y0 = bboxs[i][0]*h
        x1 = bboxs[i][3]*w
        y1 = bboxs[i][2]*h
        font_file = "/tf/ship_detect_tl/data/BebasNeue-Regular.ttf"
        font_size = 12
        font = ImageFont.truetype(font_file, font_size)
        text = str(round(scores[i]*100))+'%'
        height = font.getsize(text)[1]
        draw.rectangle([x0,y0,x1,y1], outline='#20d200')
        draw.text((x0,y0-height-3), text, font = font, fill=(32, 210, 0))
    image.save(saving_path, "PNG")

def draw_bboxs(img_name, img_dir, path_od_csv, saving_dir):
    """
    Crée à partir d'une image et du csv au format pascal VOC une nouvelle image du même avec les bboxes déssinées.

    :param img_name: str, nom de l'image de départ.
    :param img_dir: str, répertoire où se trouve l'image de départ.
    :param path_od_csv: str, chemin du CSV au format pascal VOC où se trouve les données sur les bboxes de l'image.
    :param saving_dir: str, répertoire où sera enregistrée la nouvelle image. Différent de img_dir (sinon l'image de départ sera écrasée)
    """
    df = pd.read_csv(path_od_csv)
    l = []
    scores = []

    for index, row in df[df.filename == img_name].iterrows():
        l.append([row['ymin']/768, row['xmin']/768, row['ymax']/768, row['xmax']/768])
        scores.append(1.)

    save_image_bbox(os.path.join(img_dir,img_name),os.path.join(saving_dir,img_name), l, scores)

if __name__ == '__main__':

    img_name = '4e564fc9b_t_horiz_rot90.jpg'
    img_dir = '/tf/ship_data/augmented_data/imgs'
    path_od_csv = '/tf/ship_data/augmented_data/augmented_data_OD.csv'
    saving_dir = '/tf/test/pred_test'
    draw_bboxs(img_name, img_dir, path_od_csv, saving_dir)