import torch
'''
:param text: array (原始輸入字串 Original Input String)
:param tokenized_text: array (已查找完BERT內建字典後 每個Token的ID Array)
:param label: int (原始輸入字串的Real Label)
:param max_text_len: int (最大允許原始輸入字串的長度)
:return: input_ids:array (將不足於max_text_len進行padding zero)
:return: token_type_ids:array (將文字進行分段[0]為第一段[1]第二段 本專案是分類問題所以Default [0])
:return: attention_mask:array (需要被Attention的字串標註為[1])
'''
class MuseumDatasetPreProcess():
    def __init__(self, text, tokenized_text, label, max_text_len):
        self.text = text
        self.tokenized_text = tokenized_text
        self.label = label
        self.max_text_len = max_text_len
        # Input sequence length = [CLS] + text + [SEP]
        self.max_seq_len = 1 + self.max_text_len + 1
    #跑訓練的時候會用到的length
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        input_ids_text = self.tokenized_text[index].ids
        input_ids, token_type_ids, attention_mask = self.padding(input_ids_text)

        return torch.as_tensor(input_ids), torch.as_tensor(token_type_ids), torch.as_tensor(attention_mask), self.label[index]
    def padding(self, input_ids_text):
        input_ids_text = input_ids_text[0:self.max_text_len]
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_text)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_text + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_text) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * len(input_ids_text) + [0] * padding_len
        return input_ids, token_type_ids, attention_mask