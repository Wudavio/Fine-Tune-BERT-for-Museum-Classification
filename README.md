# Fine-Tune-BERT-for-Museum-Classification

## Introduction
使用來自台灣文化部全國藝文活動資訊網的標題、內容以及活動類別微調BERT-Base-Chinese，目的是微調出一個通用的藝文活動分類器，將零散的博物館、展覽館資訊進行分類。

Fine-tune BERT-Base-Chinese using the title, content ,and activity categories from the [cultural activities](https://event.moc.gov.tw/sp.asp?xdurl=HySearchG22016/SearchResult.asp&ctNode=676&mp=1),  with the aim of fine-tuning a universal categorizer of arts and cultural activities to classify scattered museum and exhibition information

## DataSource
2010~2020年所有藝文活動
[Link 台灣文化部全國藝文活動資訊網](https://event.moc.gov.tw/sp.asp?xdurl=HySearchG22016/SearchResult.asp&ctNode=676&mp=1)

## Requirements

```
Python             3.8.10

accelerate         0.5.1
numpy              1.22.2
requests           2.27.1
scikit-learn       1.0.2
sklearn            0.0
tokenizers         0.10.3
torch              1.8.2+cu111
tqdm               4.62.3
transformers       4.5.0
```

## Training (Fine-Tune)
```
unzip MuseumMainDataset.zip
python3 Museum.py
```

## Testing Experimental Result
```
classification_report
              precision    recall  f1-score   support

         1.0     0.8954    0.9022    0.8988       920
         2.0     0.7764    0.8303    0.8025       389
         3.0     0.7278    0.6181    0.6685       199
         4.0     0.8883    0.9435    0.9151      1045
         5.0     0.9289    0.8809    0.9043       949
         6.0     0.8492    0.9033    0.8754       424
         7.0     0.8986    0.9155    0.9070       213
         8.0     0.4286    0.8571    0.5714         7
         9.0     0.9697    0.9668    0.9683      1325
        10.0     0.7212    0.7560    0.7382       455
        11.0     0.9571    0.9686    0.9628      5386
        12.0     0.6454    0.6570    0.6512       554
        13.0     0.7714    0.4141    0.5389       326

    accuracy                         0.9041     12192
   macro avg     0.8045    0.8164    0.8002     12192
weighted avg     0.9032    0.9041    0.9019     12192

```

## Prediction
```
unzip saved_museum_model.zip
cd UseTrainedModel
```

```
python3 Prediction.py "Your text"
```