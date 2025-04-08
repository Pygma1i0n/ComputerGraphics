from PIL import Image
from PIL import ImageOps
import math
import numpy as np


def quaternion_mult(a,b):
    w1,x1,y1,z1 = a
    w2,x2,y2,z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def quaternion_rotate(vertushka,ugol,v):
    vertushka = vertushka/np.linalg.norm(vertushka)
    q = [math.cos(ugol/2),vertushka[0]*math.sin(ugol/2),vertushka[1]*math.sin(ugol/2),vertushka[2]*math.sin(ugol/2)]
    q_conj=[q[0],-q[1],-q[2],-q[3]]
    v = [0,v[0],v[1],v[2]]
    return (quaternion_mult(quaternion_mult(q,v),q_conj))[1:]




def rotate(x, y, z, alpha, scale_obj,shift,vertushka):

    v = np.array([x*scale_obj, y*scale_obj, z*scale_obj])
    alpha = alpha * 3.14 / 180


    v =quaternion_rotate(vertushka,alpha,v)


    v = v + shift
    return v


def draw_triangle(img, z_buffer, t0, t1, t2, u0, v0, u1, v1, u2, v2, texture, cosA, cosB, cosC):
    scale = 2500
    if(texture != None):
        W,H = texture.size


    if (cosA < 0 and cosB < 0 and cosC < 0):
        x0, y0, z0 = t0
        x1, y1, z1 = t1
        x2, y2, z2 = t2

        x0p = (scale / z0 * x0 + img.shape[1]/2)
        y0p = (scale / z0 * y0 + img.shape[0]/2)
        x1p = (scale / z1 * x1 + img.shape[1]/2)
        y1p = (scale / z1 * y1 + img.shape[0]/2)
        x2p = (scale / z2 * x2 + img.shape[1]/2)
        y2p = (scale / z2 * y2 + img.shape[0]/2)

        xmin = int(min(x0p, x1p, x2p) - 1)
        xmax = int(max(x0p, x1p, x2p) + 1)
        ymin = int(min(y0p, y1p, y2p) - 1)
        ymax = int(max(y0p, y1p, y2p) + 1)

        if (xmin < 0): xmin = 0
        if (ymin < 0): ymin = 0
        if (xmax > img.shape[1]): xmax = img.shape[1]
        if (ymax > img.shape[0]): ymax = img.shape[0]

        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                lambd0, lambd1, lambd2 = barCenter(x, y, x0p, y0p, x1p, y1p, x2p, y2p)

                if(u0):
                    Wt = int(W * (lambd0 * u0 + lambd1 * u1 + lambd2 * u2))
                    Ht = int(H * (lambd0 * v0 + lambd1 * v1 + lambd2 * v2))

                if (lambd0 >= 0 and lambd1 >= 0 and lambd2 >= 0):

                    I = -255 * (cosA * lambd0 + cosB * lambd1 + cosC * lambd2)
                    if(texture == None):
                        rgb_value = [I,I,I]
                    else:
                        rgb_value = texture.getpixel((Wt, Ht))
                    Z = lambd0 * z0 + lambd1 * z1 + lambd2 * z2
                    if (z_buffer[y, x] > Z):
                        img[y, x] = rgb_value
                        z_buffer[y, x] = Z


def barCenter(x, y, x0, y0, x1, y1, x2, y2):
    lambd0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambd1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambd2 = 1.0 - lambd0 - lambd1
    return lambd0, lambd1, lambd2


def triangulate_convex(polygon, p):
    triangles = []
    n = len(polygon)
    if n < 3:
        return []
    for i in range(1, n - 1):
        p.append([polygon[0], polygon[i], polygon[i + 1]])
    return triangles


def norm(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = [0] * 3
    n[0] = ((y1 - y2) * (z1 - z0)) - ((z1 - z2) * (y1 - y0))
    n[1] = ((x1 - x2) * (z1 - z0)) - ((z1 - z2) * (x1 - x0))
    n[2] = ((x1 - x2) * (y1 - y0)) - ((y1 - y2) * (x1 - x0))
    return n


def cos(n):
    cosA = n[2] / math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    return cosA


def draw_model(file,image_texture,img_mat,z_buffer,scale, shift, alpha,vertushka):
    file = open(file, 'r+')
    if (image_texture != None):
        image_texture = Image.open(image_texture)
        image_texture = image_texture.convert("RGB")
        image_texture = ImageOps.flip(image_texture)
    v = []
    vt = []
    p = []
    pt = []
    flag = True
    for str in file:
        splitted_str = str.split()
        if (len(splitted_str) == 0):
            continue
        if (splitted_str[0] == 'vt'):
            vt.append([float(splitted_str[1]), float(splitted_str[2])])
        if (splitted_str[0] == 'v'):
            v.append([float(splitted_str[1]), float(splitted_str[2]), float(splitted_str[3])])
        if (splitted_str[0] == 'f'):
            polyg = []
            textur = []
            for i in splitted_str[1:]:
                polyg.append(int(i.split('/')[0]))
                if ((i.split('/')[1] != '') and flag):
                    textur.append(int(i.split('/')[1]))
                else:
                    flag = False
            if len(polyg) > 3:
                triangulate_convex(polyg, p)
                triangulate_convex(textur, pt)

            else:
                p.append(polyg)
                pt.append(textur)

    for i in v:
        i[0], i[1], i[2] = rotate(i[0], i[1], i[2], alpha, scale, shift,vertushka)

    vn_calc = np.zeros((len(v), 3), dtype=np.float32)

    for tr in p:
        n = norm(v[tr[0] - 1][0], v[tr[0] - 1][1], v[tr[0] - 1][2],
                 v[tr[1] - 1][0], v[tr[1] - 1][1], v[tr[1] - 1][2],
                 v[tr[2] - 1][0], v[tr[2] - 1][1], v[tr[2] - 1][2])

        vn_calc[tr[0] - 1][0] += n[0]
        vn_calc[tr[0] - 1][1] += n[1]
        vn_calc[tr[0] - 1][2] += n[2]

        vn_calc[tr[1] - 1][0] += n[0]
        vn_calc[tr[1] - 1][1] += n[1]
        vn_calc[tr[1] - 1][2] += n[2]

        vn_calc[tr[2] - 1][0] += n[0]
        vn_calc[tr[2] - 1][1] += n[1]
        vn_calc[tr[2] - 1][2] += n[2]

    for i in range(len(vn_calc)):
        vn_calc[i] = vn_calc[i] / np.linalg.norm(vn_calc[i])

    for tr in range(len(p)):
        if (flag):
            u0 = vt[pt[tr][0] - 1][0]
            v0 = vt[pt[tr][0] - 1][1]
            u1 = vt[pt[tr][1] - 1][0]
            v1 = vt[pt[tr][1] - 1][1]
            u2 = vt[pt[tr][2] - 1][0]
            v2 = vt[pt[tr][2] - 1][1]
        else:
            u0 = None
            v0 = None
            u1 = None
            v1 = None
            u2 = None
            v2 = None

        cosA = cos(vn_calc[p[tr][0] - 1])
        cosB = cos(vn_calc[p[tr][1] - 1])
        cosC = cos(vn_calc[p[tr][2] - 1])
        # color=[randint(0,255),randint(0,255),randint(0,255)]
        color = [255, 255, 255]
        draw_triangle(img_mat, z_buffer,
                      v[p[tr][0] - 1],
                      v[p[tr][1] - 1],
                      v[p[tr][2] - 1],
                      u0, v0, u1, v1, u2, v2,
                      image_texture, cosA, cosB, cosC)



img_mat = np.zeros((500, 1500, 3), dtype=np.uint8)
z_buffer = np.full((500, 1500), np.inf, dtype=np.float32)



for i in range(5):
    draw_model('jesus.obj',None,img_mat,z_buffer,0.5,[10*i-20, -1.5, 100],-45*i-90,[0,1,0])


draw_model('Avent.obj',None,img_mat,z_buffer,0.8,[-3, -1.5, 30],25,[3, -1.5, 30])
draw_model('Avent.obj',None,img_mat,z_buffer,0.8,[3, -1.5, 30],25,[3, -1.5, 30])


img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img1.png')