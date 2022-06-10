import pandas as pd
import numpy as np

def rle2bbox(rle, shape):
    '''
    taken from : https://www.kaggle.com/code/eigrad/convert-rle-to-bounding-box-x0-y0-x1-y1/notebook
    '''
    
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))  # an array of (start, length) pairs
    a[:,0] -= 1  # `start` is 1-indexed
    
    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    if np.any(y1 > shape[0]):
        # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)
    
    x0 = a[:,0] // shape[0]
    x1 = (a[:,0] + a[:,1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)
    
    if x1 > shape[1]:
        # just went out of the image dimensions
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    return x0, y0, x1, y1

def generate_od_csv(path_original_csv, path_new_csv):
    """
    Generates a new csv according to Tensorflow object detection format (Pascal VOC) from a dataframe build according to the structure of the original CSV (from kaggle dataset)

    :param path_to_original_csv: str, path du csv de départ fourni par kaggle, où la position des bateaux est codée au format RLE.
    :param path_new_csv: str, path du nouveau csv que l'on souhaite créer.
    :return: Void.
    """

    df = pd.read_csv(path_original_csv)
    H = 768
    W = 768

    image_names = df['ImageId'].unique() # list of image names
    od_dict = {'filename':[], 'width': [], 'height': [], 'class': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}

    i=0 # will be printed to show progression
    for image in image_names:
        print(i)
        for rle in df.loc[df['ImageId'] == image]['EncodedPixels']:
            if rle == rle : # testing here if rle is not a 'NaN'
                xmin, ymin, xmax, ymax = rle2bbox(rle, (H, W))
                od_dict['class'].append('ship')
            else :
                xmin, ymin, xmax, ymax = '', '', '', ''
                od_dict['class'].append('')
            od_dict['filename'].append(image)
            od_dict['width'].append(W)
            od_dict['height'].append(H)
            od_dict['xmin'].append(xmin)
            od_dict['ymin'].append(ymin)
            od_dict['xmax'].append(xmax)
            od_dict['ymax'].append(ymax)
        i+=1

    df_od_train= pd.DataFrame(od_dict)
    df_od_train.to_csv(path_new_csv, index=False)