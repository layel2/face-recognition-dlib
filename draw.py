import cv2
from tools import *

def face_bbox(img, bboxes, newImg=False):
    if newImg :
        img = img.copy()
    for bbox in bboxes:
        cv2.rectangle(img, *bbox, (0, 0, 255), 2)
    if newImg:
        return img

def face_bbox_name(img, bboxes, names, newImg=False):
    if newImg :
        img = img.copy()
    for bbox,name in zip(bboxes,names):
        t_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 2, 3)[0]
        cv2.rectangle(img, *bbox, (0, 0, 255), 2)
        cv2.rectangle(img, (bbox[0][0],bbox[1][1]),(bbox[0][0]+t_size[0]+5,bbox[1][1]+t_size[1]+5), (0, 0, 255), -1)
        cv2.putText(img,name,(bbox[0][0],bbox[1][1]+20),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3 )
    if newImg:
        return img

def crop_face(img, bboxes):
    crop_imgs = []
    for bbox in bboxes:
        crop_img = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        crop_imgs.append(crop_img)
    return crop_imgs