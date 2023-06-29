import pandas as pd
import cv2
from parinya import LINE
from datetime import datetime
import numpy as np
# import sqlite3
# import helper
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import time

#####
driver = 'ODBC Driver 17 for SQL Server'
server = '10.31.1.19'
database = 'FBS'
uid = 'fbs'
pwd = 'fbsrw@2023'
from sqlalchemy.engine import URL
connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={uid};PWD={pwd}'
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

from sqlalchemy import create_engine
engine = create_engine(connection_url)
####

os.chdir('D:\Projects\Ai\Flare\Codes\extraction')
print(os.getcwd())
line = LINE('QEBozjS32SIaiYpQX87e1LrMbUdSx97tYu3HdRcrbze')

def streaming(model,source,conf,project,name,save_txt,exist_ok,save_vid=False):
    dicsource={'Sea View':'rtsp://10.30.24.206','TF2':'rtsp://viewer:password@1@10.30.24.210/axis-media/media.amp?videocodec=jpeg&resolution=704x576&fps=24'}
    # inv_model=dict(zip(model.model.names.values(),model.model.names.keys()))
    classes=list(model.model.names.keys())
    s = -1
    a = 0
    st = False
    font = cv2.FONT_HERSHEY_SIMPLEX
    vid_cap=cv2.VideoCapture(dicsource[source])
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    table_name = 'result'
    df1=pd.DataFrame({key:[] for key in ['Camera','Model','DetectTime','Flag','Classes','x','y','w','h','Source','SubFlare','Steam']})
    
    while (vid_cap.isOpened()):
        success,image=vid_cap.read()
        if success:
            t=datetime.now()
            if t.second!=s:
                print(datetime.now())
                s=t.second
                # image = cv2.resize(image,(width, height))
                cv2.rectangle(image, (15, height-10), (525, height-50), (0, 0, 0), -1)
                cv2.putText(image, str(datetime.now()), (20, height-20),
                            font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # image=cv2.resize(image,(720,int(720*(9/16))))
                # image=cv2.rectangle(img=image,pt1=(0,385),pt2=(200,485),color=(0,0,0),thickness=-1)
                # image=cv2.putText(img=image,text=t.strftime('%Y-%m-%d %H:%M:%S'),org=(5,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=.5,color=(255,255,255),thickness=1)
                res=model.predict(image,conf=conf,project=project,name=name,save_txt=save_txt,exist_ok=exist_ok,classes=classes)
                setobj=[int(c) for r in res for c in r.boxes.cls]
                xy=np.array([box.xywh.numpy() for box in res[0].boxes])
                xy=xy.reshape(-1,4).T
                class_series=[np.nan] if len(setobj)==0 else [model.names[i] for i in setobj]
                flag='N' if len(setobj)==0 else 'Y'
                n=1 if len(setobj)==0 else len(setobj)
                res_plotted=res[0].plot()
                cv2.imshow('detect',res_plotted)
                # if save_vid:cv2.imwrite(f"{savepath}/{datetime.now().strftime('%Y%m%d%H%M%S')}.png",res_plotted, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                if save_vid:plt.imsave(f"{savepath}/{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",res_plotted[:,:,::-1])

                #sent line notify
                if flag == 'Y':
                    if  a >= 3 and a <= 5:
                        # exec(open('D:/Projects/Ai/Flare/flarelinenoti_seaview_shot.py').read())
                        line.sendimage(res_plotted[:,:,::-1],f'{datetime.now()} {source} !!! Start !!!')
                        st = True
                    a += 1
                elif flag == 'N'  and st == False:
                    a = 0
                elif flag == 'N' and a > 5:
                    # exec(open("D:/Projects/Ai/Flare/flarelinenoti_seaview_shot.py").read())
                    line.sendimage(res_plotted[:,:,::-1],f'{datetime.now()} {source} --- Stop ----')
                    st = False
                #################
                df2=pd.DataFrame({'Camera':[source]*n,'Model':[modeldict[modelweight]]*n,'DetectTime':[t.strftime('%Y-%m-%d %H:%M:%S')]*n,'Flag':[flag]*n,'Classes':class_series,'x':xy[0],'y':xy[1],'w':xy[2],'h':xy[3],'Source':[np.nan]*n}) if flag=='Y' else pd.DataFrame({'Camera':[source]*n,'Model':[modeldict[modelweight]]*n,'DetectTime':[t.strftime('%Y-%m-%d %H:%M:%S')]*n,'Flag':[flag]*n,'Classes':class_series,'x':[np.nan]*n,'y':[np.nan]*n,'w':[np.nan]*n,'h':[np.nan]*n,'Source':[np.nan]*n})
                df1=pd.concat([df1,df2],axis=0)
                # time.sleep(1. - t.second - t.microsecond / 1e6)
                time.sleep(1 - t.microsecond/1e6)

                if s==0:
                    df1.drop_duplicates(subset=['Camera','Model','DetectTime','Flag','Classes'],inplace=True)
                    # df1.to_sql('result',con,if_exists='append',index=False)
                    df1['DetectTime'] = pd.to_datetime(df1.DetectTime)
                    df1.to_sql(name=table_name, con=engine, if_exists='append', index=False)
                    print(f'write sql on {t}')
                    print(df1)
                    df1=pd.DataFrame({key:[] for key in ['Camera','Model','DetectTime','Flag','Classes','x','y','w','h','Source','SubFlare','Steam']})
                    # print(df1)
                    # print(df1.dtypes)
                    # time.sleep(60.0 - t.second - t.microsecond / 1e6)

            if cv2.waitKey(1) & 0xFF==ord('q'):break
        else:
            print(f'No input CCTV signal {datetime.now()}')
            exec(open("D:/Projects/Ai/Flare/Codes/extraction/flaredetect.py").read())
            # continue
    vid_cap.release()
    cv2.destroyAllWindows()
# source_rtsp='TF2'
# modelweight='FlareTF2(Day)'
source_rtsp='Sea View'
modelweight='FlareOverview'
modeldict={'General':'yolov8s','FlareOverview':'SeaView','FlareTF2(Day)':'TF2Day','FlareTF2(Night)':'TF2night'}
# con=sqlite3.connect('./Flaredatabase.db')
model_path=f'./weights/{modeldict[modelweight]}.pt'
# model=helper.load_model(model_path)
model=YOLO(model_path)
savepath=os.path.join(os.path.curdir,f'RTSP1/{source_rtsp}')
if os.path.exists(savepath)==False:os.makedirs(savepath)

try:
    streaming(model=model,source=source_rtsp,conf=0.25,project='savelabelroot',name='savelabelfolder',save_txt=True,exist_ok=True,save_vid=True)
except:
    print(f'RESTART {datetime.now()}')
    exec(open("D:/Projects/Ai/Flare/Codes/extraction/flaredetect.py").read())
# streaming(model=model,source=source_rtsp,conf=0.25,project='savelabelroot',name='savelabelfolder',save_txt=True,exist_ok=True,save_vid=True)

