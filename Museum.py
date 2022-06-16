import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertModel, BertConfig, BertTokenizerFast, BertForSequenceClassification

from tqdm.auto import tqdm

#模組化寫法 -> from檔案import方法
from MuseumDataset import MuseumDatasetBuilder #dataset module
from RecordTrainLog import RecordTrainLog #record log module
from MuseumDatasetPreProcess import MuseumDatasetPreProcess #MuseumDatasetPreProcess
from EvalMethod import classification_scores
#Transformers 有一個集中的日誌記錄系統，因此您可以輕鬆設置庫的詳細程度。
from transformers import logging
logging.set_verbosity_error()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 檢查是否使用GPU訓練模型??
def use_gpu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print(torch.version.cuda)
    return device
device = use_gpu()

#深度學習的自動混合精度
#深度神經網絡訓練傳統上依賴於 IEEE 單精度格式，但是對於混合精度，
#您可以使用半精度進行訓練，同時保持單精度實現的網絡精度。這種同時使用單精度和半精度表示的技術稱為混合精度技術。
#混合精度訓練的好處
#通過使用張量核心加速數學密集型運算，例如線性和卷積層。
#與單精度相比，通過訪問一半的字節來加速內存受限的操作。
#減少訓練模型的內存需求，支持更大的模型或更大的小批量(batch)。
fp16_training = True
#fp16_training = False
if fp16_training:
    from accelerate import Accelerator
    accelerator = Accelerator(fp16=True)
    device = accelerator.device

#載入BERT與tokenizer
print("Downloading BERT Model start...")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=14,
                                                      output_attentions = False,
                                                      output_hidden_states = False,
                                                      attention_probs_dropout_prob=0.3,
                                                      hidden_dropout_prob=0.3).to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
print("Downloading BERT Model end...")

#要不要調整BERT內權重 (需要就註解)
for param in model.base_model.parameters():
    param.requires_grad = True

#打印模型結構
print("""
name            module
----------------------""")
for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print(f"{name}:{n}")
    else:
        print("{:15} {}".format(name, module))

#MuseumDataset 整理後
MuseumDatasetBuilder = MuseumDatasetBuilder()
trainDatasetForText, validationDatasetForText, testDatasetForText , trainLabel, validationLabel, testLabel = MuseumDatasetBuilder.getProcessedDataset()

#Input tokenizer
trainDataTokenized = tokenizer([train_data[0] for train_data in trainDatasetForText], add_special_tokens=True)
validationDataTokenized = tokenizer([dev_data[0] for dev_data in validationDatasetForText], add_special_tokens=True)
testDataTokenized = tokenizer([test_data[0] for test_data in testDatasetForText], add_special_tokens=True)


#PyTorch DataLoader工作原理可視化(可參考網址) -> https://mp.weixin.qq.com/s/Uc2LYM6tIOY8KyxB7aQrOw
MaxTextLangth = 150
trainSet = MuseumDatasetPreProcess(trainDatasetForText, trainDataTokenized, trainLabel, MaxTextLangth)
validationSet = MuseumDatasetPreProcess(validationDatasetForText, validationDataTokenized, validationLabel, MaxTextLangth)
testSet = MuseumDatasetPreProcess(testDatasetForText, testDataTokenized, testLabel, MaxTextLangth)

trainBatchSize = 64
validationBatchSize = 32
testBatchSize = 32

trainLoader = DataLoader(trainSet, batch_size=trainBatchSize, shuffle=True, pin_memory=True, num_workers=4)
validationLoader = DataLoader(validationSet, batch_size=validationBatchSize, shuffle=False, pin_memory=True)
testLoader = DataLoader(testSet, batch_size=testBatchSize, shuffle=False, pin_memory=True)



num_epoch = 10 #Fine tune 幾次
validation = True #增加驗證集來觀察訓練集overfitting
loggingStep = 155
loggingValiStep = 1000
loggingTestStep = 31
learningRate = 3e-5 #2E-5, 5E-5實驗acc差不多
optimizer = AdamW(model.parameters(), lr=learningRate)


print("optimizer lr:", learningRate)
print("fp16_training", fp16_training)


if fp16_training:
    model, optimizer, trainLoader = accelerator.prepare(model, optimizer, trainLoader)
RecordTrainLog = RecordTrainLog() #Record the training log to json
def training():
    print("Start Training ...")
    model.train()
    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0

        for data in tqdm(trainLoader):
            # Load all data into GPU
            data = [i.to(device) for i in data]

            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], labels=data[3])

            # train accuracy
            label_ids = data[3].cpu().numpy()
            pred = np.argmax(output.logits.detach().cpu(), axis=1).numpy()

            #計算訓練loss, accuracy
            train_acc += (pred == label_ids).mean()
            train_loss += output.loss

            if fp16_training:
                accelerator.backward(output.loss)
            else:
                output.loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % loggingStep == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / loggingStep:.3f} , acc = {train_acc / loggingStep:.3f}")
                log = RecordTrainLog.buildLogStyle("Training", epoch + 1, step, round((train_loss.item() / loggingStep), 3), round((train_acc / loggingStep), 3))
                RecordTrainLog.writeLogJson(log)
                train_loss = train_acc = 0.0

        if validation:
            print("Evaluating Dev Set ...")
            eval_num = 0
            model.eval()
            with torch.no_grad():
                dev_acc = dev_loss = 0.0
                dev_total_acc = dev_total_loss = 0.0
                for i, eval_data in enumerate(tqdm(validationLoader)):
                    output = model(input_ids=eval_data[0].to(device),
                                    token_type_ids=eval_data[1].to(device),
                                    attention_mask=eval_data[2].to(device),
                                    labels=eval_data[3].to(device))
                    # prediction is correct only if answer text exactly matches
                    validationLabel = eval_data[3].cpu().numpy()
                    dev_pred = np.argmax(output.logits.detach().cpu(), axis=1).numpy()
                    dev_acc += (dev_pred == validationLabel).mean()
                    dev_total_acc += (dev_pred == validationLabel).mean()
                    dev_loss += output.loss
                    dev_total_loss += output.loss
                    eval_num += 1

                    if(eval_num % loggingValiStep == 0) :
                        print(f"Validation | Epoch {epoch + 1} | loss = {dev_loss.item() / loggingValiStep:.3f} , acc = {dev_acc / loggingValiStep:.3f}")
                        log = RecordTrainLog.buildLogStyle(
                            "Validation",
                            epoch + 1,
                            eval_num,
                            round((dev_loss.item() / eval_num), 3),
                            round((dev_acc / eval_num), 3)
                        )
                        RecordTrainLog.writeLogJson(log)
                        dev_acc = dev_loss = 0.0

                print(f"Validation | Epoch {epoch + 1} | loss = {dev_total_loss.item() / len(validationLoader):.3f} , acc = {dev_total_acc / len(validationLoader):.10f}")
                log = RecordTrainLog.buildLogStyle(
                    "Validation",
                    epoch + 1,
                    "Last",
                    round((dev_total_loss.item() / len(validationLoader)), 3),
                    round((dev_total_acc / len(validationLoader)), 10)
                )
                RecordTrainLog.writeLogJson(log)
            model.train()

def testing():
    print("Evaluating Test Set ...")
    model.eval()
    test_num = 0
    with torch.no_grad():
        test_acc = test_loss = 0.0
        test_total_acc = test_total_loss = 0.0

        testLabel_arr = []
        test_pred_arr = []


        for i, test_data in enumerate(tqdm(testLoader)):
            output = model(input_ids=test_data[0].to(device),
                            token_type_ids=test_data[1].to(device),
                            attention_mask=test_data[2].to(device),
                            labels=test_data[3].to(device))
            # prediction is correct only if answer text exactly matches
            testLabel = test_data[3].cpu().numpy()
            test_pred = np.argmax(output.logits.detach().cpu(), axis=1).numpy()
            test_loss += output.loss
            test_total_loss += output.loss
            test_acc += (test_pred == testLabel).mean()
            test_total_acc += (test_pred == testLabel).mean()

            #合併所有label
            testLabel_arr = np.hstack((testLabel_arr, testLabel))
            test_pred_arr = np.hstack((test_pred_arr, test_pred))


            test_num += 1
            if(test_num % loggingTestStep == 0) :
                print(f"Testing | acc = {test_acc / test_num:.3f}")
                log = RecordTrainLog.buildLogStyle(
                    "Testing",
                    num_epoch,
                    test_num,
                    round((test_loss.item() / test_num), 3),
                    round((test_acc / test_num), 3)
                )
                print(f"Testing | loss = {test_loss.item() / loggingTestStep:.3f} ,acc = {test_acc / loggingTestStep:.10f}")
                RecordTrainLog.writeLogJson(log)
                test_acc = test_loss = 0.0

        print(f"Testing | loss = {test_total_loss.item() / len(testLoader):.3f} ,acc = {test_total_acc / len(testLoader):.10f}")
        classification_scores(testLabel_arr, test_pred_arr)

        log = RecordTrainLog.buildLogStyle(
            "Testing",
            num_epoch,
            "Last",
            round((test_total_loss.item() / len(testLoader)), 3),
            round((test_total_acc / len(testLoader)), 3)
        )
        RecordTrainLog.writeLogJson(log)

training()
testing()

# Save Trained Model
print("Saving Model ...")
model_save_dir = "saved_museum_model"
model.save_pretrained(model_save_dir)