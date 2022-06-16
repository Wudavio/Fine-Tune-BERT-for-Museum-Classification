# -*- coding: utf-8 -*-
import json
import time
import requests
from datetime import datetime,timezone,timedelta
from requests.adapters import HTTPAdapter


try:
    to_unicode = unicode
except NameError:
    to_unicode = str


class RecordTrainLog():
    #init 建立json檔案
    def __init__(self):
        self.fileName = time.strftime(""+self.getNowTime()) + "-log"
        with open("./TraingLogDir/"+self.fileName+'.json', 'w') as f:
            f.write(to_unicode({}))
        print(self.fileName)

    #writeLogJson 寫入json檔案
    def writeLogJson(self, data):
        #load資料
        with open("./TraingLogDir/"+self.fileName+'.json') as f:
            fileData = json.load(f)
            f.close()

        fileData.update(data)
        #call request write log
        #if self.requestRecordToServer(fileData) == 200:
        #    print(self.requestRecordToServer(fileData))

        #write資料
        with open("./TraingLogDir/"+self.fileName+'.json', 'w') as f:
            str_ = json.dumps(fileData,
                              indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            f.write(to_unicode(str_))
            f.close()
        return True

    #將log寫進去server
    def requestRecordToServer(self, dataJson = ''):
        apiUrl = 'https://lab0726.at.tw/LogView/Home/updateModelLog'
        s = requests.session()
        # s.mount('https://', HTTPAdapter(max_retries=3))
        s.adapters.DEFAULT_RETRIES = 5 #增加重新連線次數
        s.keep_alive = False
        response = s.post(apiUrl, data= json.dumps(dataJson), timeout=10)
        return response.text

    #python get now time
    def getNowTime(self):
        #設定時區
        dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
        dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
        return dt2.strftime("%Y-%m-%d %H:%M:%S")
        # print('UTC \t%s\nUTC+8\t%s'%(dt1,dt2))
        #print(dt2.strftime("%Y-%m-%d %H:%M:%S")) # 將時間轉換為 string

    def buildLogStyle(self, split, epoch, step, loss, acc):
        return {f"{split}_Epoch_{epoch}_Step_{step}" : {
            'epoch' : epoch,
            'loss' : loss,
            'acc' : acc,
            'time' : self.getNowTime()
        }}

