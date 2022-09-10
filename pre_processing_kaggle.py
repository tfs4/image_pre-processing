import cv2, glob, numpy
import os

def scaleRadius(img,scale):

    k = img.shape[0]/2
    x = img[int(k), :, :].sum(1)
    r=(x>x.mean()/10).sum()/2
    if r == 0:
        r = 1
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)
def preprocessing(dir, destino):
    for scale in [500]:
        cont = 0
        #print(len(glob.glob(dir+"*")))
        for f in (glob.glob(dir+"*")):

            dir = f.split('/')

            extn = dir[-1].split('.')

            if extn[-1] == 'csv':
                continue

            uri = dir[0] + "/" + str(scale) + "/" + dir[1] + "/" + dir[2]

            cont = cont+1

            #uri_final = dir[0] + "/" + str(scale) + "/" + dir[2]
            uri_final = destino+dir[-1]
            if not os.path.isfile(uri_final):
                #print(uri)
                # print(f)
                print(f)
                a=cv2.imread(f)
                a = scaleRadius(a, scale)
                b = numpy.zeros(a.shape)
                x = a.shape[1] / 2
                y = a.shape[0] / 2
                center_coordinates = (int(x), int(y))
                cv2.circle(b, center_coordinates, int(scale * 0.9), (1, 1, 1), -1, 8, 0)
                aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128) * b + 128 * (1 - b)


                cv2.imwrite(destino+dir[-1], aa)

