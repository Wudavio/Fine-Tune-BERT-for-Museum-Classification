# -*- coding: utf-8 -*-
import json
import numpy as np
import random
import torch

class MuseumDatasetBuilder():
    def __init__(self):
      self.DatasetJson = []
      self.DatasetArr = []
    #read Dataset.json
    def build(self):
        with open('./MuseumMainDataset.json') as f:
            self.DatasetJson = json.load(f)

    #padding array
    def padding(self):
        self.DatasetArr = []
        for item in self.DatasetJson:
            #過濾特殊字元 Filiter special token
            item['title'] = ''.join(char for char in item['title'] if char.isalnum())
            item['content'] = ''.join(char for char in item['content'] if char.isalnum())
            self.DatasetArr.append([item['title'] + item['content'], item['type']]) #(title + content) -> tokenizer

    #資料分割方法(訓練70%、驗證15%、測試資料15%)
    def segmentationData(self, arr, start_num, end_num):
        data = []
        for item in range(start_num,end_num):
            data.append(arr[item])
        return data

    #回傳資料分割(訓練70%、驗證15%、測試資料15%)
    def getProcessedDataset(self):
        self.same_seeds(2022)
        self.build()
        self.padding()
        np.random.shuffle(self.DatasetArr) #資料打亂(亂數排序)
        #text
        trainingSet = self.segmentationData(self.DatasetArr, 0, 56896)  #70
        validationSet = self.segmentationData(self.DatasetArr, 56896, 69088) #15
        testSet = self.segmentationData(self.DatasetArr, 69089, 81281) #15
        #label
        train_label = [int(train_data[1]) for train_data in trainingSet]
        dev_label = [int(dev_data[1]) for dev_data in validationSet]
        test_label = [int(test_data[1]) for test_data in testSet]

        #顯示資料集內容
        print("total len:", (len(trainingSet)+len(validationSet)+len(testSet)))
        print("trainingSet:", len(trainingSet))
        print("validationSet:", len(validationSet))
        print("testSet:", len(testSet))
        print(trainingSet[0])
        print(validationSet[0])
        return trainingSet, validationSet, testSet, train_label, dev_label, test_label
    #在神經網絡中，參數默認是進行隨機初始化的。
    #不同的初始化參數往往會導致不同的結果，當得到比較好的結果時我們通常希望這個結果是可以復現的，
    #在pytorch中，通過設置隨機數種子也可以達到這麼目的。
    def same_seeds(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True