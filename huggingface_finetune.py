import os
import json
import numpy as np
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForSequenceClassification, TFBertModel
from sklearn.preprocessing import LabelEncoder
from rasa.nlu.training_data.formats import MarkdownReader

# disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def read_inputs_json(input_path):
    texts = []
    cats = []
    with open(input_path, mode="r") as file_:
        data = json.load(file_)['rasa_nlu_data']['common_examples']
        for d in data:
            texts.append(d['text'])
            cats.append(d['intent'])

    return texts, cats


def read_inputs_md(input_path):
    reader = MarkdownReader()
    reader.read(input_path, language='de', fformat='MARKDOWN')
    texts = []
    cats = []
    for message in reader.training_examples:
        texts.append(message.text)
        cats.append(message.get('intent'))

    return texts, cats

train_texts, train_labels = read_inputs_md(
    os.path.join('train_test_split', 'training_data.md')
)
test_texts, test_labels = read_inputs_json(
    os.path.join('train_test_split', 'test_data.md')
)

le = LabelEncoder().fit(train_labels)
train_labels = le.transform(train_labels)
test_labels = le.transform(test_labels)


config = BertConfig.from_pretrained("dbmdz/bert-base-german-cased",
                                    num_labels=len(le.classes_))
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
model = TFBertForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-german-cased",
    config=config,
    trainable=True,
)

def encode(texts):
    input_ids = []
    attention_mask = []
    for text in texts:
        tokens = tokenizer.encode_plus(text,
                                       max_length=128,
                                       pad_to_max_length=True,
                                       return_token_type_ids=False,
                                       return_attention_mask=True)
        input_ids.append(tokens['input_ids'])
        attention_mask.append(tokens['attention_mask'])
    
    return np.array(input_ids), np.array(attention_mask)


train_input_ids, train_attention_mask = encode(train_texts)
test_input_ids, test_attention_maks = encode(test_texts)
print(train_input_ids.shape, train_attention_mask.shape)

opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

model.compile(loss=loss, optimizer=opt, metrics=[metric])

print('Model compiled; start training...')
history = model.fit([train_input_ids, train_attention_mask],
                    train_labels,
                    epochs=20,
                    batch_size=8,
                    validation_data=([test_input_ids, test_attention_maks], test_labels))

bert = TFBertModel.from_pretrained("dbmdz/bert-base-german-cased")
bert.bert = model.bert
bert.save_pretrained('model/')

tokenizer.save_pretrained('model/')
