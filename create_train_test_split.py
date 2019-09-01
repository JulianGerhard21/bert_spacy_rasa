import json

import pandas as pd
from sklearn.model_selection import train_test_split


def create_rasa_training_set(df_train_set):
    label_samples = {}
    with open('train.md', 'w', encoding='utf-8') as file:
        for index, entry in df_train_set.iterrows():
            if entry['label'] not in label_samples:
                label_samples[entry['label']] = []
                label_samples[entry['label']].append(entry['article'])
            else:
                label_samples[entry['label']].append(entry['article'])
        for label, articles in label_samples.items():
            file.write('## intent:' + label + '\n')
            for article in articles:
                file.write('- ' + article + '\n')


def create_rasa_test_set(df_test_set):
    label_samples = {}
    with open('test.md', 'w', encoding='utf-8') as file:
        for index, entry in df_test_set.iterrows():
            if entry['label'] not in label_samples:
                label_samples[entry['label']] = []
                label_samples[entry['label']].append(entry['article'])
            else:
                label_samples[entry['label']].append(entry['article'])
        for label, articles in label_samples.items():
            file.write('## intent:' + label + '\n')
            for article in articles:
                file.write('- ' + article + '\n')


df_train_set = pd.read_excel('mennekes_full_v2.xlsx')
train_label_list = df_train_set['label'].unique().tolist()
# do a stratified train test split and persist the result
train_dataframe, eval_dataframe = train_test_split(df_train_set, test_size=0.10, stratify=df_train_set['label'])
train_data = list(train_dataframe.itertuples(index=False, name=None))
test_data = list(eval_dataframe.itertuples(index=False, name=None))

create_rasa_training_set(train_dataframe)
create_rasa_test_set(eval_dataframe)

with open('train.json', 'w') as handle:
    json.dump(train_data, handle)

with open('test.json', 'w') as handle:
    json.dump(test_data, handle)
