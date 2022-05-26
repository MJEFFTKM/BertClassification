import transformers
import torch
import numpy as np
import re
import pandas as pd
import torch
import random
import os
import transformers as ppb
from tqdm import tqdm

# model_transformer.py
from torch.utils.data import Dataset, random_split, Subset, Sampler, DataLoader
from torch import nn
from transformers import DistilBertConfig, DistilBertModel

from config import (
    VAL_DATA_PATH,
    TOKENIZER_PATH,
    CONFIG_BERT_PATH,
    PRETRAINED_WEIGHTS_PATH,
    BATCH_SIZE
)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def activate_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def clean_str_sst(string):
    string = re.sub(r"[^A-Za-zА-Яа-я0-9@+-/(/)]", " ", string)
    return string.strip().lower()


def tokenize(example, tokenizer):
    tokens = tokenizer.encode(example, add_special_tokens=True)
    return [101] * (len(tokens) > 512) + tokens[-511:]


def preprocess_function(string, tokenizer):
    return tokenize(clean_str_sst(string), tokenizer)


class TextsDataset(Dataset):  # для пандаса
    def __init__(self,
                 df,
                 tokenizer,
                 have_label=True):
        self.have_label = have_label
        self.texts_column_name = 'description'
        self.title_column_name = 'title'
        self.df_indices = df.index.values
        self.tokenized = list(map(lambda text: preprocess_function(text, tokenizer), tqdm(df[self.texts_column_name] + ' ' + df[self.title_column_name])))
        if have_label:
            self.target_name = 'is_bad'
            self.labels = df[self.target_name].values  # name of target column

    def __getitem__(self, idx):
        if self.have_label:
            return {"df_index": self.df_indices[idx], "tokenized": self.tokenized[idx], "label": self.labels[idx]}
        return {"df_index": self.df_indices[idx], "tokenized": self.tokenized[idx]}

    def __len__(self):
        return len(self.tokenized)


class TextsSubset(Subset):
    def __init__(self,
                 textsDataset):
        super().__init__(textsDataset, np.arange(len(textsDataset)))
        self.df_indices = textsDataset.df_indices


class TextsSampler(Sampler):
    def __init__(self,
                 subset,
                 batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.subset = subset
        self.indices = subset.indices
        self.df_indices = subset.df_indices
        self.tokenized = np.array(subset.dataset.tokenized)[self.indices]

    def __iter__(self):

        batch_idx = []
        for index in np.argsort(list(map(len, self.tokenized))):
            batch_idx.append(index)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []

        if len(batch_idx) > 0:
            yield batch_idx

    def __len__(self):
        return len(self.dataset)


def get_padded(values):
    max_len = 0
    for value in values:
        if len(value) > max_len:
            max_len = len(value)

    padded = np.array([value + [0] * (max_len - len(value)) for value in values])

    return padded


class func():
    def __init__(self, is_train):
        self.is_train = is_train

    def collate_fn(self, batch):

        inputs = []
        labels = []
        indices = []
        for elem in batch:
            indices.append(elem['df_index'])
            inputs.append(elem['tokenized'])
            if self.is_train:
                labels.append(elem['label'])

        inputs = get_padded(inputs)
        attention_mask = (inputs != 0).astype('int')
        if self.is_train:
            return {"df_index": np.array(indices), "inputs": torch.tensor(inputs), "labels": torch.FloatTensor(labels),
                    'attention_mask': torch.tensor(attention_mask)}
        return {"df_index": np.array(indices), "inputs": torch.tensor(inputs),
                'attention_mask': torch.tensor(attention_mask)}


class BertClassifier(nn.Module):  # оставляю
    def __init__(self, pretrained_model, dropout=0.3):
        super().__init__()

        self.bert = pretrained_model
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(768, affine=False)

        sizes = [768, 768]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Tanh())
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.Linear(sizes[-1], 1))
        self.linear = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, attention_mask):
        afterbert = self.bert(inputs, attention_mask)[0][:, 0, :]
        afterbert = self.dropout(afterbert)
        afterbert = self.relu(afterbert)
        afterbert = self.bn(afterbert)

        afterlinear = self.linear(afterbert)
        proba = self.sigmoid(afterlinear)
        return proba


def model_predict(model, iterator, device, is_train=True):
    model.eval()
    y_pred = []
    labels = []
    indices = []
    with torch.no_grad():

        for i, batch in tqdm(enumerate(iterator)):
            inputs = batch['inputs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(inputs, attention_mask)
            indices.append(batch['df_index'])
            y_pred.append(output.detach().cpu().numpy().reshape(1, -1)[0])
            if is_train:
                labels = batch['labels'].reshape(-1, 1).to(device)
        y_pred = np.concatenate(y_pred).reshape(1, -1)[0]
        indices = np.concatenate(indices)

    return indices, y_pred

def get_model():
    seed_everything(42)
    device = activate_device()
    tokenizer = torch.load(TOKENIZER_PATH)
    configuration = DistilBertConfig.from_json_file(CONFIG_BERT_PATH) #config_distil_rubert CONFIG_DICTIL_RUBERT_PATH
    bert = DistilBertModel(configuration)
    bert_clf = BertClassifier(bert)
    bert_clf.to(device)
    bert_clf.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device))
    return device, tokenizer, bert_clf


def get_df_after_transformer(df,
                             have_dataset=False,
                             dataset=None,
                             is_train=True):
    device, tokenizer, bert_clf = get_model()
    if not have_dataset:
        tokenized_text = TextsDataset(df, tokenizer, have_label=is_train)
    else:
        tokenized_text = dataset
    evaluate_data = TextsSubset(tokenized_text)
    evaluate_loader = DataLoader(evaluate_data,
                                 batch_sampler=TextsSampler(evaluate_data),
                                 collate_fn=func(is_train=is_train).collate_fn)
    indices, y_pred = model_predict(bert_clf, evaluate_loader, device, is_train=is_train)
    return indices, y_pred

# if __name__ == '__main__':
#     val_df = pd.read_csv(VAL_DATA_PATH)
#     #val_df = val_df.drop(['is_bad'], axis=1)
#     print(get_df_after_transformer(val_df[:10], is_train=False))