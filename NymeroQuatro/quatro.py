from random import randint

from PIL import Image
from PIL import ImageOps
import math
import numpy as np


def rotate(x,y,z,alpha,beta,gamma):
    shift = np.array([0,-1.5,5])
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



def draw_triangle(img,z_buffer,t0,t1,t2,u0, v0, u1, v1, u2, v2,texture,cosA,cosB,cosC):

    scale = 2500

    W = 2048
    H = 2048


    if (cosA < 0 and cosB < 0 and cosC < 0):
        x0,y0,z0 = t0
        x1,y1,z1 = t1
        x2,y2,z2 = t2

        x0p = (scale/z0 * x0 + 1500)
        y0p = (scale/z0 * y0 + 1500)
        x1p = (scale/z1 * x1 + 1500)
        y1p = (scale/z1 * y1 + 1500)
        x2p = (scale/z2 * x2 + 1500)
        y2p = (scale/z2 * y2 + 1500)

        xmin = int(min(x0p, x1p, x2p) - 1)
        xmax = int(max(x0p, x1p, x2p) + 1)
        ymin = int(min(y0p, y1p, y2p) - 1)
        ymax = int(max(y0p, y1p, y2p) + 1)

        if (xmin < 0): xmin = 0
        if (ymin < 0): ymin = 0
        if (xmax > 3000): xmax = 3000
        if (ymax>3000): ymax = 3000

        for x in range(xmin,xmax):
            for y in range(ymin,ymax):
                lambd0,lambd1,lambd2=barCenter(x,y,x0p,y0p,x1p,y1p,x2p,y2p)

                Wt = int(W * (lambd0 * u0 + lambd1 * u1 + lambd2 * u2))
                Ht = int(H * (lambd0 * v0 + lambd1 * v1 + lambd2 * v2))

                if(lambd0>=0 and lambd1>=0 and lambd2>=0):
                    rgb_value = image_texture.getpixel((Wt, Ht))
                    Z = lambd0*z0 +lambd1*z1+lambd2*z2
                    if(z_buffer[y,x] > Z):

                        img[y,x] = rgb_value
                        z_buffer[y, x]=Z




def barCenter(x, y, x0, y0, x1, y1, x2, y2):
    lambd0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) /  ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambd1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /  ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambd2 = 1.0 - lambd0 - lambd1
    return lambd0, lambd1, lambd2

def triangulate_convex(polygon,p):
    triangles = []
    n = len(polygon)
    if n < 3:
        return []
    for i in range(1, n - 1):
        p.append([polygon[0], polygon[i], polygon[i + 1]])
    return triangles


def norm(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    n = [0] * 3
    n[0] = ((y1-y2) * (z1-z0)) - ((z1-z2) * (y1-y0))
    n[1] = ((x1-x2) * (z1-z0)) - ((z1-z2) * (x1-x0))
    n[2] = ((x1-x2) * (y1-y0)) - ((y1-y2) * (x1-x0))
    return n

def cos(n):
    cosA = n[2]/ math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    return cosA


file =open('Vivi_Final.obj', 'r+')
image_texture = Image.open('Vivi_Diffuse.tga.png')
image_texture = image_texture.convert("RGB")
image_texture = ImageOps.flip(image_texture)
v=[]
vt=[]
f =[]
ft = []
p=[]
pt = []
img_mat= np.zeros((3000,3000,3),dtype=np.uint8)
z_buffer = np.full((3000,3000), np.inf, dtype = np.float32)

for str in file:
    splitted_str = str.split()
    if(len(splitted_str)==0):
        continue
    if(splitted_str[0] == 'vt'):
        vt.append([float(splitted_str[1]), float(splitted_str[2])])
    if(splitted_str[0] == 'v'):
        v.append([float(splitted_str[1]),float(splitted_str[2]),float(splitted_str[3])])
    if(splitted_str[0] == 'f'):
        polyg = []
        textur = []
        for i in splitted_str[1:]:
            polyg.append(int(i.split('/')[0]))
            textur.append(int(i.split('/')[1]))
        f.append(polyg)
        ft.append(textur)

if len(polyg) > 3 :
    for i in range(len(f)):
        triangulate_convex(f[i],p)
        triangulate_convex(ft[i], pt)

else:
    p=f
    pt = ft



for i in v:
    i[0],i[1],i[2]=rotate(i[0],i[1],i[2],0,180,0)



vn_calc = np.zeros((len(v),3),dtype=np.float32)


for tr in p:
    n = norm( v[tr[0] - 1][0], v[tr[0] - 1][1], v[tr[0] - 1][2],
                  v[tr[1] - 1][0], v[tr[1] - 1][1], v[tr[1] - 1][2],
                  v[tr[2] - 1][0], v[tr[2] - 1][1], v[tr[2] - 1][2])

    vn_calc[tr[0] - 1][0] +=n[0]
    vn_calc[tr[0] - 1][1] +=n[1]
    vn_calc[tr[0] - 1][2] +=n[2]

    vn_calc[tr[1] - 1][0] +=n[0]
    vn_calc[tr[1] - 1][1] +=n[1]
    vn_calc[tr[1] - 1][2] +=n[2]

    vn_calc[tr[2] - 1][0] +=n[0]
    vn_calc[tr[2] - 1][1] +=n[1]
    vn_calc[tr[2] - 1][2] +=n[2]

for i in range(len(vn_calc)):
    vn_calc[i] = vn_calc[i] / np.linalg.norm(vn_calc[i])






for tr in range(len(p)):

    u0 = vt[pt[tr][0]-1][0]
    v0 = vt[pt[tr][0]-1][1]
    u1 = vt[pt[tr][1]-1][0]
    v1 = vt[pt[tr][1]-1][1]
    u2 = vt[pt[tr][2]-1][0]
    v2 = vt[pt[tr][2]-1][1]

    cosA = cos(vn_calc[p[tr][0] - 1])
    cosB = cos(vn_calc[p[tr][1] - 1])
    cosC = cos(vn_calc[p[tr][2] - 1])
    # color=[randint(0,255),randint(0,255),randint(0,255)]
    color =[255,255,255]
    draw_triangle(img_mat,z_buffer,
                  v[p[tr][0] - 1],
                  v[p[tr][1] - 1],
                  v[p[tr][2] - 1],
                  u0, v0, u1, v1, u2, v2,
                  image_texture,cosA,cosB,cosC)

img =  Image.fromarray(img_mat, mode= 'RGB')
img = ImageOps.flip(img)
img.save('img1.png')