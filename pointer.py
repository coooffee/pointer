# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np

    

def distance_p2p(x1,y1,xr,yr):    #计算点到点的距离

    dis = ((x1-xr)**2+(y1-yr)**2)**0.5

    return dis


def distance_p2l(x1,y1,x2,y2,xr,yr):       #计算点到线段所在直线的距离

    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2 
    dis = abs(A * xr + B * yr + C) / np.sqrt(A**2+ B**2)

    return dis


def calK(x1,y1,x2,y2,rx,ry):       #计算两个点到中心点k值的比值

    k1 = (y1-ry)/(x1-rx)
    k2 = (y2-ry)/(x2-rx)

    return k1/k2


def farther_p(x1,y1,x2,y2,xr,yr):       # 获取线段两个端点中，距离中心点较远的点（用于计算角度）
    
    d1 = distance_p2p(x1,y1,xr,yr)
    d2 = distance_p2p(x2,y2,xr,yr)
    if d1>d2:
        return x1,y1
    return x2,y2


def close_p(x1,y1,x2,y2,xr,yr):          # 获取距离中心较近的点（用于切割数字）
    d1 = distance_p2p(x1,y1,xr,yr)
    d2 = distance_p2p(x2,y2,xr,yr)
    if d1<d2:
        return x1,y1
    return x2,y2


def cal_theta(x,y,xr,yr):             #计算角度 
    tan_theta = (y-yr)/(x-xr)
    theta = math.atan(tan_theta)
    angle = math.degrees(theta)
    if angle<0:
        angle += 180
    return angle


def img_cutting(img,x,y):        #图像初次切割，x，y是刻度的近点 

    cut_img = img[y-12:y+32, x-22:x+22]

    return cut_img


def img_cutting_2(img,x,y,w,h):    #图像进一步切割 将数字精确切割出来

    cut_img = img[y:y+h, x:x+w]

    return cut_img

def cal_per_angle_represent(dic):            
    
    keys = []
    values = []
    per_angle = []

    for k,v in dic.items():
        keys.append(k)                       
        values.append(v)                     

    for i in range(len(keys)-1):
        dev_keys = abs(keys[i+1] - keys[i])
        dev_values = abs(values[i+1] - values[i])
        per_angle.append(dev_keys/dev_values)

    per_angle_rep = round(np.mean(per_angle),2)


    print('keys: %s' % keys)
    print('values: %s' % values)


    return keys,values,per_angle_rep
    

                

def make_templates(num):                   #制作模板,为识别数字做准备

    path = 'template_%d.jpg' % num

    input_img = cv2.imread(path)
    gray_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    binary_img = cv2.resize(gray_img,(9,15))
    _a,binary_img = cv2.threshold(binary_img, 150, 255, cv2.THRESH_BINARY)
    binary_img = binary_img.reshape(1,-1)

    return binary_img

def num_identification(input_num):         #识别出单个数字

    gray_img = cv2.cvtColor(input_num,cv2.COLOR_BGR2GRAY)
    resize_img = cv2.resize(gray_img,(9,15)).reshape(1,-1)
    _a,binary_num = cv2.threshold(resize_img, 150, 255, cv2.THRESH_BINARY)

    num_0 = make_templates(0)
    num_1 = make_templates(1)
    num_2 = make_templates(2)
    num_4 = make_templates(4)
    num_8 = make_templates(8)

    identifi_result = {}
    identifi_result['0'] = (binary_num-num_0).sum()
    identifi_result['1'] = (binary_num-num_1).sum()
    identifi_result['2'] = (binary_num-num_2).sum() 
    identifi_result['4'] = (binary_num-num_4).sum()
    identifi_result['8'] = (binary_num-num_8).sum()

    predic_num = min(identifi_result, key=identifi_result.get)

    if identifi_result[predic_num]<5000:       #5000是一个阈值,设定小于这个阈值算识别出数字
        return int(predic_num)

    # else:
    #     print('this is not a number')



def rotate_img(src,center_x,center_y,degree):     #旋转图像

    (height,width)=src.shape[:2]
    center=(center_x,center_y)     
    angle = degree *np.pi/ 180
    a=np.sin(angle)
    b=np.cos(angle)
    width_rotate=int(height*np.fabs(a)+width*np.fabs(b))
    height_rotate=int(height*np.fabs(b)+width*np.fabs(a))
    M=cv2.getRotationMatrix2D(center,degree,1)
    img_rotate=cv2.warpAffine(src,M,(width_rotate,height_rotate))
    return img_rotate


def get_num(img):     #第二次切割,返回最后读出的数字


    finalnum = 0
    index = 0
    numbers =[40, 60, 80, 100, 120, 140]     #保存仪表上存在的数字,提高读数准确性


    gray_num = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _a,binary_img = cv2.threshold(gray_num, 150, 255, cv2.THRESH_BINARY)          #将灰度图转为二值化图像

    image, contours, hier = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   #在二值化图像中寻找单个数字

    for c in contours:
        
        x,y,w,h = cv2.boundingRect(c)

        
        if 5<w<15 and 10<h<20:                                                         #限制每个数字的高和宽 (9,15)
            cut_img_2 = img_cutting_2(img,x,y,w,h)

            # cv2.imshow('num',cut_img_2)                                                #将每个数字切割出来并保存
            # cv2.imwrite('precisenum_%d_%d.jpg' % (partnum, precisenum),cut_img_2)      #partnum是第几个图片中的数字，precisenum是这张图片中第几个数字
            # precisenum += 1

            scalenum = num_identification(cut_img_2)                                     

            finalnum += scalenum*(10**index)
            index += 1
        
    if finalnum in numbers:
        return finalnum

def pre_process(img):

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gauss_img = cv2.GaussianBlur(gray_img, (5, 5),3)  
    edges = cv2.Canny(gauss_img,30, 50)

    return gray_img, edges

            



def read_pointer_instrument(img):       #标注出表盘，圆心，刻度，指针

    global partnum
    global picnum

    partnum = 0
    picnum = 0

    lines = []
    num_angle_dic = {}

    gray_img, edges = pre_process(img)

    for i in range(10):
        line  = cv2.HoughLinesP(edges, 1, np.pi/180, 0, minLineLength=15, maxLineGap=i)   #寻找图片中直线，包括刻度和指针 
        lines.append(line)

    circle = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 1)                         #寻找图像中的圆
    

    if len(circle)>0 and len(lines)>0: 
    
        rx = circle[0][0][0]  
        ry = circle[0][0][1]
        r = circle[0][0][2]

        print('r is %d' % r)

        for difgap in lines:

            for line in difgap:

                x1 = line[0][0]
                y1 = line[0][1]
                x2 = line[0][2]
                y2 = line[0][3]

                if distance_p2l(x1,y1,x2,y2,rx,ry) < 10:   

                    if 2*r/3 <=distance_p2p(x1,y1,rx,ry)< r and 2*r/3 <=distance_p2p(x2,y2,rx,ry)< r:   #画刻度线
                        
                        gray_img = cv2.line(gray_img,(x1,y1),(x2,y2),(0,0,255),2)        #在灰度图上画刻度线
                        x_f,y_f = farther_p(x1,y1,x2,y2,rx,ry)                           #刻度远点
                        x_c,y_c = close_p(x1,y1,x2,y2,rx,ry)                             #刻度近点
                        scale_angle = round(cal_theta(x_f,y_f,rx,ry), 1)                 #计算刻度的角度,1位小数

                        rot_i_img = rotate_img(img, x_c, y_c, scale_angle-90)            #将图像按照刻度角度旋转，以便于切图
                        cut_img_1 = img_cutting(rot_i_img, x_c, y_c)                     #将数字区域初步切出来(已经预先设置好切的大小)
                        cut_img_1_rot = rotate_img(cut_img_1,22,22,-scale_angle-270)     #将切出的数字再次旋转，旋转回原位.两次旋转角度相加为360
                        
                        # cv2.imwrite('part_%d.jpg' % partnum,cut_img_rot)               #保存第一次切出来的数字
                        # partnum += 1
                        # cv2.imshow('cut_num',cut_img_rot)

                        try:
                            number = get_num(cut_img_1_rot)                              #输出切出的数字图像和最后的数字（多位数）

                            if number is not None:
                                
                                if number in num_angle_dic:
                                    num_angle_dic[number] = (num_angle_dic[number]+scale_angle)/2
                                else:
                                    num_angle_dic[number] = scale_angle

                            # print('number is : %d' % number)
                            # print('scale angle is: %f' % scale_angle)

                            # cv2.waitKey(0)
                        except:
                            continue


                        # per_angle_represent = round(number/scale_angle, 2)
                        # per.append(per_angle_represent)

                        
                    

                    if r/2 < distance_p2p(x1,y1,x2,y2) < r:          #画指针

                        gray_img = cv2.line(gray_img,(x1,y1),(x2,y2),(255,0,0),2)
                        x, y =farther_p(x1,y1,x2,y2,rx,ry)
                        pointer_angle = round(cal_theta(x,y,rx,ry), 1) 

                        print('pointer angle is: %f' % pointer_angle)

                        # cv2.waitKey(0)
                

        print(num_angle_dic)

        keys,values,per_angle_rep = cal_per_angle_represent(num_angle_dic)

        final_val_min = per_angle_rep*(pointer_angle-min(values))+min(keys)                        #ffffffffffffffffffffffinal result base on min
        final_val_max = max(keys)-per_angle_rep*(max(values) - pointer_angle)                        #ffffffffffffffffffffffinal result base on max
        final_val = (final_val_max+final_val_min)/2
        print('number is :%d' % final_val)

        img_withcircle = cv2.circle(gray_img,(int(rx),int(ry)),int(r),(0,0,0),3)            #画圆（表盘）
        img_withpoint = cv2.circle(img_withcircle,(int(rx),int(ry)),1,(255,255,255),3)      #画圆心
        
        cv2.imshow('final',img_withpoint)
        cv2.imwrite('finalimg_%d.jpg' % picnum ,img_withpoint)

        picnum += 1

        return img_withpoint

    else:
        return gray_img



def main(path):	
	
    input_img = cv2.imread(path)
    resize_img = cv2.resize(input_img,None,fx=0.5,fy=0.5)
    # h,w = resize_img.shape[:2]  
    # print('h is %d,w is %d' % (h,w))          h is 540,w is 960
    read_pointer_instrument(resize_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    for i in range(1,6):  
        path = '%d.jpg' % i
        print(i)
        main(path)
