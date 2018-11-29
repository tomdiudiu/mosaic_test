import os
import cv2
import numpy as np
import h5py
from keras.models import load_model
from keras.models import Model
from keras.preprocessing import image
from PIL import Image,ImageStat
import time
# load model
time1=time.time()
model=load_model('./model/resnet34_final.h5')
# 加载训练好的模型
time2=time.time()
print('成功加载模型，用时%d秒'%(time2-time1))
#load picture                                                 #这部分不知道怎么加载RTSP流视频
# videos_src_path=input("please input the RTSP of the video:")
videos_src_path='./test.mp4'
time3=time.time()
cap=cv2.VideoCapture(videos_src_path)
time4=time.time()
if cap.isOpened():
 print('成功加载视频，用时%d秒'%(time4-time3))
mak_path='./msk/'
time_interval = 5
def video2frame(video_src_path,interval,mak_path):
    cap=cv2.VideoCapture(video_src_path)
    frame_index=0
    frame_count=0
    mask_count=0
    time5=time.time()
    while cap.isOpened():
        success,frame=cap.read()
        print('-->正在读取第%d帧：'%frame_index,success)
        if success:
        #打印物理指标:
        # 大小
         size=frame.shape
         print(size)
        if frame_index%interval==0:
            resize_frame=cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA)
            # 进行预测
            x = image.img_to_array(resize_frame)
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)
            print(preds)
            # if preds[0][0]>preds[0][1]:
                # cv2.imwrite(mak_path+'%d.jpg'%frame_count,frame)

            frame_count = frame_count + 1
        frame_index=frame_index+interval
    time6=time.time()
    print('一共读取%d帧图片，用时%d秒'%(frame_count,time6-time5))
if __name__ == '__main__':
    video2frame(videos_src_path,time_interval,mak_path)

