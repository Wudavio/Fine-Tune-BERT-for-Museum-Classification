import torch
class MuseumDatasetPreProcess():
    def __init__(self, text, tokenized_text):
        self.text = text
        self.tokenized_text = tokenized_text
        self.max_text_len = 150
        # Input sequence length = [CLS] + text + [SEP]
        self.max_seq_len = 1 + self.max_text_len + 1

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        #Select bert wordbook ids(number)
        input_ids_text = self.tokenized_text[index].ids
        #Pad zeros
        input_ids, token_type_ids, attention_mask = self.padding(input_ids_text)

        return torch.as_tensor(input_ids), torch.as_tensor(token_type_ids), torch.as_tensor(attention_mask)
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