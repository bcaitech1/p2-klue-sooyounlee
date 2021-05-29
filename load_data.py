import pickle as pickle
import os
import pandas as pd
import torch
from pororo import Pororo
from tqdm import tqdm

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    label = []

    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame(
        {'sentence': dataset[1], 'entity_01': dataset[2], 'entity_01_spos': dataset[3], 'entity_01_epos': dataset[4],
         'entity_02': dataset[5], 'entity_02_spos': dataset[6], 'entity_02_epos': dataset[7], 'label': label, })
    return out_dataset


# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset


# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
#    ner = Pororo(task="ner", lang="ko")

    fixed_sents = []

    for sent, ent01, start1, end1, ent02, start2, end2 in zip(dataset['sentence'], dataset['entity_01'], dataset['entity_01_spos'], dataset['entity_01_epos'], dataset['entity_02'], dataset['entity_02_spos'], dataset['entity_02_epos']):
#        ner_01 = ' | ' + ner(ent01)[0][1].lower() +' | '
#        ner_02 = ' ^ ' + ner(ent02)[0][1].lower() + ' ^ '

        if start1 < start2:
            sent = sent[:start1] + '@' + sent[start1:end1 + 1] + ' @ ' + sent[end1 + 1: start2] + '#' + sent[start2: end2 + 1] + ' # ' + sent[end2 + 1:]
        else:
            sent = sent[:start2] + '#' + sent[start2:end2 + 1] + ' # ' + sent[end2 + 1: start1] + '@' + sent[start1: end1 + 1] + ' @ ' + sent[end1 + 1:]

        # sent += " 앞의 문장에서 " + ent01 + ', ' + ent02 + "의 관계는?"

        fixed_sents.append(sent)

    concat_entity = []

    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '</s>' + e02
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        # list(dataset['sentence']),
        fixed_sents,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )
    return tokenized_sentences
