import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.io import imread

def join_datasets(datasets, datasets_test=None):
    dfs = []
    for key in datasets.keys():
        datasets[key][0].reset_index(inplace=True)
        dfs.append(datasets[key][0])

    return dfs

def calc_aug(datasets):
    datalabel = pd.concat(join_datasets(datasets))

    c = datalabel['level'].value_counts()
    max = c.max()

    img_add = {}
    img_add[0] = max - c[0]
    img_add[1] = max - c[1]
    img_add[2] = max - c[2]
    img_add[3] = max - c[3]
    img_add[4] = max - c[4]

    return img_add


def do_augmentation(datasets, img_add=None, path='datasets/augmentation/', angulo = 7):
    rotation = int(angulo)
    if img_add is None:
        img_add = calc_aug(datasets)

    augmentation_dict = {}
    max_value = img_add[max(img_add, key=img_add.get)]

    while max_value > 0:
        img_add, augmentation_dict = augmentation(datasets, img_add, augmentation_dict, angulo, path)
        max_value = img_add[max(img_add, key=img_add.get)]
        angulo = angulo+rotation
    data_items = augmentation_dict.items()
    data_list = list(data_items)
    df = pd.DataFrame(data_list, columns=['id_code', 'level'])
    df.to_csv(path+'../augmentation.csv', index=False)



def augmentation(data_lebel, img_add, augmentation_dict, ang, path):
    for x in data_lebel:
        file_path = data_lebel[x][1]
        dfs = data_lebel[x][0]
        for i, row in dfs.iterrows():
            if img_add[row['level']] == 0:
                continue
            augmentation_dict[str(str(ang)+'_aug_'+row['id_code'])] = row['level']
            create_image(row['id_code'], file_path+'/', ang, path)
            img_add[row['level']] = img_add[row['level']]-1
    return img_add, augmentation_dict

def create_image(img_name , file_path,  ang, path):
    img = imread(file_path+img_name)
    img = img / 255
    plt.imsave(path + '/' + str(ang) + '_aug_' + str(img_name), rotate(img, angle=ang, center=None, mode='wrap'))