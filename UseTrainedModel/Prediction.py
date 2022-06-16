import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
import numpy as np
from tqdm.auto import tqdm
import sys
from PreProcessForPrediction import MuseumDatasetPreProcess
filename = sys.argv[0] #filename
parameter_1_text = sys.argv[1] #parameter[0]

#Whether to use GPU for training
def use_gpu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print(torch.version.cuda)
    return device
device = use_gpu()

# loading model and config
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
config = BertConfig.from_json_file("../saved_museum_model/config.json")
model = BertForSequenceClassification.from_pretrained("../saved_museum_model/", config=config).to(device)


model.eval()

parameter_1_text = ''.join(char for char in parameter_1_text if char.isalnum())

predictionText = [[parameter_1_text]] # parameter or json

predictionText_Tokenized = tokenizer([textState[0] for textState in predictionText], add_special_tokens=True)


predictionSet = MuseumDatasetPreProcess(predictionText, predictionText_Tokenized)

predictionSetLoader = DataLoader(predictionSet, batch_size=1, shuffle=False, pin_memory=True)

#Prediction Label
with torch.no_grad():
    for i, test_data in enumerate(tqdm(predictionSetLoader)):
        output = model(input_ids=test_data[0].to(device),
                        token_type_ids=test_data[1].to(device),
                        attention_mask=test_data[2].to(device))

        test_pred = np.argmax(output.logits.detach().cpu(), axis=1).numpy()
    print(test_pred)
    #final result
    fp = open("predictionResult.txt", "w+")
    fp.writelines(str(test_pred[0]))
    fp.close()


