from PIL import Image
from PIL import ImageOps
import math
import numpy as np

# def draw_image(img,x0,y0,x1,y1,color):
#

#
#
#
#     for t in np.arange(0,1,0.01):
#         x = round((1-t)*x0+ t*x1)
#         y = round((1-t)*y0 + t*y1)
#

# def draw_line(img, x0, y0, x1, y1,color):
#
#     swap=False
#     if( abs(x0-x1) < abs(y0-y1)):
#         x0,y0=y0,x0
#         x1,y1=y1,x1
#         swap = True
#
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#
#     for x in range(x0,x1):
#         t= (x-x0)/(x1-x0)
#         y = round((1.0 - t) * y0 + t * y1)
#
#         if(swap):
#             img[x,y]=color
#         else:
#             img[y,x] = color

# def draw_line(img, x0, y0, x1, y1,color):
#     swap=False
#     if( abs(x0-x1) < abs(y0-y1)):
#         x0,y0=y0,x0
#         x1,y1=y1,x1
#         swap = True
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#     y = y0
#     dy = abs(y1 - y0) / (x1 - x0)
#     derror = 0.0
#     y_update = 1 if y1 > y0 else -1
#     for x in range(x0, x1):
#         if(swap):
#             img[x,y]=color
#         else:
#             img[y,x] = color
#         derror += dy
#         if (derror > 0.5):
#             derror -= 1.0
#             y += y_update

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



file =open('model_1.obj','r+')
v=[]
p=[]
img_mat= np.zeros((1000,1000,3),dtype=np.uint8)
img_mat[0:1000,0:1000] = 255
for str in file:
    splitted_str = str.split()
    if(splitted_str[0] == 'v'):
        v.append([float(splitted_str[1]),float(splitted_str[2]),float(splitted_str[3])])
    if(splitted_str[0] == 'f'):
        p.append([int(splitted_str[1].split('/')[0]),int(splitted_str[2].split('/')[0]),int(splitted_str[3].split('/')[0])])


for vertex in v:
    img_mat[int(7000*vertex[1]+250),int(7000*vertex[0]+500)] = 0

for tr in p:
    draw_line(img_mat, int(7000*v[tr[0]-1][0]+500), int(7000*v[tr[0]-1][1]+250) , int(7000*v[tr[1]-1][0]+500) , int(7000*v[tr[1]-1][1]+250),[0,0,0])
    draw_line(img_mat, int(7000*v[tr[1]-1][0]+500), int(7000*v[tr[1]-1][1]+250) , int(7000*v[tr[2]-1][0]+500) , int(7000*v[tr[2]-1][1]+250),[0,0,0])
    draw_line(img_mat, int(7000*v[tr[2]-1][0]+500), int(7000*v[tr[2]-1][1]+250) , int(7000*v[tr[0]-1][0]+500) , int(7000*v[tr[0]-1][1]+250),[0,0,0])



img =  Image.fromarray(img_mat, mode= 'RGB')
img = ImageOps.flip(img)
img.save('img1.png')