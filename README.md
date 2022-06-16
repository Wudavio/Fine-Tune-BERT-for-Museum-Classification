# Fine-Tune-BERT-for-Museum-Classification

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

## Prediction
```
unzip saved_museum_model.zip
cd UseTrainedModel
```

```
python3 Prediction.py "Your text"
```