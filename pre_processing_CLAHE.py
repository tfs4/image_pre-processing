
import numpy as np
import cv2
import pandas as pd
SEED = 42
import warnings
warnings.filterwarnings("ignore")




def circle_crop(p, sigmaX=10):
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 5)
    img = clahe.apply(img)+30


    height, width  = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    if img.ndim == 2:
        mask = img > 7
        img = img[np.ix_(mask.any(1), mask.any(0))]
    #img = clahe.apply(img) + 30
    #img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img










df_messidor = pd.read_csv('messidor_2/data.csv')

df_idrid_test = pd.read_csv('IDRID/test.csv')

df_idrid_train = pd.read_csv('IDRID/train.csv')

df_aptos = pd.read_csv('APTOS/train.csv')

def run():
    for idx, row in df_aptos.iterrows():
        path = f"APTOS/train/{row['id_code']}"
        image = circle_crop(path, sigmaX=30)
        cv2.imwrite(f"pre_processing_2/APTOS/train/{row['id_code']}", image)

    for idx, row in df_messidor.iterrows():
        path = f"messidor_2/dataset/{row['id_code']}"
        image = circle_crop(path, sigmaX=30)
        cv2.imwrite(f"pre_processing_2/messidor_2/train/{row['id_code']}", image)

    for idx, row in df_idrid_train.iterrows():
        path = f"IDRID/train/{row['id_code']}.jpg"
        image = circle_crop(path, sigmaX=30)
        cv2.imwrite(f"pre_processing_2/IDRID/train/{row['id_code']}.jpg", image)

    for idx, row in df_idrid_test.iterrows():
        path = f"IDRID/test/{row['id_code']}.jpg"
        image = circle_crop(path, sigmaX=30)
        cv2.imwrite(f"pre_processing_2/IDRID/test/{row['id_code']}.jpg", image)



