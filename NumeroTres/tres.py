from random import randint

from PIL import Image
from PIL import ImageOps
import math
import numpy as np

def rotate(x,y,z,alpha,beta,gamma):
    shift = np.array([0.0   ,-0.04,0.5])
    v=np.array([x,y,z])
    alpha = alpha*3.14/180
    beta = beta*3.14/180
    gamma = gamma*3.14/180
    matrixX =np.array([ [1,0,0],
            [0, math.cos(alpha) , math.sin(alpha)],
            [0,-math.cos(alpha),math.cos(alpha)]])

    matrixY =np.array( [ [math.cos(beta), 0, math.sin(beta)],
                [0,1,0],
                [-math.sin(beta), 0 , math.cos(beta)]])

    matrixZ = np.array([[math.cos(gamma), math.sin(gamma), 0],
               [-math.sin(gamma), math.cos(gamma), 0],
               [0,0,1]])

    R=matrixX @ matrixY @ matrixZ
    v = R @ v
    v=v+shift
    return v




def draw_triangle(img,z_buffer,x0, y0, z0, x1, y1, z1, x2, y2, z2,color):

    scale = 3500

    n = norm(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cosA = scalar(n)

    if (cosA < 0):
        x0p = (scale/z0 * x0 + 600)
        y0p = (scale/z0 * y0 + 600)
        x1p = (scale/z1 * x1 + 600)
        y1p = (scale/z1 * y1 + 600)
        x2p = (scale/z2 * x2 + 600)
        y2p = (scale/z2 * y2 + 600)
        xmin = int(min(x0p, x1p, x2p) - 1)
        xmax = int(max(x0p, x1p, x2p) + 1)
        ymin = int(min(y0p, y1p, y2p) - 1)
        ymax = int(max(y0p, y1p, y2p) + 1)
        if (xmin < 0): xmin = 0
        if (ymin < 0): ymin = 0
        if (xmax > 1200): xmax = 1200
        if (ymax>1200): ymax = 1200
        color = [-color[0] * cosA, -color[1] * cosA, -color[2] * cosA]
        for x in range(xmin,xmax):
            for y in range(ymin,ymax):
                lambd0,lambd1,lambd2=barCenter(x,y,x0p,y0p,x1p,y1p,x2p,y2p)
                if(lambd0>=0 and lambd1>=0 and lambd2>=0):
                    Z = lambd0*z0 +lambd1*z1+lambd2*z2
                    if(Z<z_buffer[y,x]):

                        img[y,x] = color
                        z_buffer[y, x]=Z



def barCenter(x, y, x0, y0, x1, y1, x2, y2):
    lambd0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) /  ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambd1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /  ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambd2 = 1.0 - lambd0 - lambd1
    return lambd0, lambd1, lambd2

#алгоритм брезенхема
def draw_line(img, x0, y0, x1, y1,color):
    swap=False
    if( abs(x0-x1) < abs(y0-y1)):
        x0,y0=y0,x0
        x1,y1=y1,x1
        swap = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if(swap):
            img[x,y]=color
        else:
            img[y,x] = color
        derror += dy
        if (derror > (x1-x0)):
            derror -= 2*(x1 - x0)
            y += y_update

def norm(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    n = [0] * 3
    n[0] = ((y1-y2) * (z1-z0)) - ((z1-z2) * (y1-y0))
    n[1] = ((x1-x2) * (z1-z0)) - ((z1-z2) * (x1-x0))
    n[2] = ((x1-x2) * (y1-y0)) - ((y1-y2) * (x1-x0))
    return n

def scalar(n):
    cosA = n[2]/ math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    return cosA


file =open('model_1.obj', 'r+')
v=[]
p=[]
img_mat= np.zeros((1200,1200,3),dtype=np.uint8)
z_buffer = np.full((1200,1200), np.inf, dtype = np.float32)

for str in file:
    splitted_str = str.split()
    if(len(splitted_str)==0):
        continue
    if(splitted_str[0] == 'v'):
        v.append([float(splitted_str[1]),float(splitted_str[2]),float(splitted_str[3])])
    if(splitted_str[0] == 'f'):
        p.append([int(splitted_str[1].split('/')[0]),int(splitted_str[2].split('/')[0]),int(splitted_str[3].split('/')[0])])


for i in v:
    i[0],i[1],i[2]=rotate(i[0],i[1],i[2],0,0,0)




for tr in p:


    color=[randint(0,255),randint(0,255),randint(0,255)]
    # color =[255,255,255]
    draw_triangle(img_mat,z_buffer,
                  v[tr[0] - 1][0], v[tr[0] - 1][1], v[tr[0] - 1][2],
                  v[tr[1] - 1][0], v[tr[1] - 1][1], v[tr[1] - 1][2],
                  v[tr[2] - 1][0], v[tr[2] - 1][1], v[tr[2] - 1][2],
                  color)

img =  Image.fromarray(img_mat, mode= 'RGB')
img = ImageOps.flip(img)
img.save('img1.png')