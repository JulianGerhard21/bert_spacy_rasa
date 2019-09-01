#!/usr/bin/env python
# coding: utf-8

import json
import logging.config
import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


def get_features(nlp, texts):
    """

    :param nlp:
    :param texts:
    :return:
    """
    features = []
    for doc in nlp.pipe(texts, batch_size=32):
        features.append(doc.vector)
    return np.array(features)


# Load fine-tuned model
model_dir = 'de_pytt_bertbasecased_lg_gnad'
logger.info('Loading fine-tuned model...')
nlp = spacy.load(model_dir)

# Load test data
logger.info('Loading test data...')
with open('data/test.json', 'r', encoding='utf-8') as handle:
    test_data = json.load(handle)

with open('data/train.json', 'r', encoding='utf-8') as handle:
    train_data = json.load(handle)

train_cats, train_texts = zip(*train_data)
test_cats, test_texts = zip(*test_data)


# Get the features of training and test data
logger.info('Get features of training and test data...')
train_feats = get_features(nlp, train_texts)
test_feats = get_features(nlp, test_texts)


# Train a logistic regression model
logger.info('Train a Logistic Regression model...')
clsr = LogisticRegression()
clsr.fit(train_feats, train_cats)
logger.info('Accuracy of Logistic Regression model on test data: {}'.format(clsr.score(test_feats, test_cats)))

# Train a SVM model
logger.info('Train a SVM model...')
svc = SVC(C=1, gamma=0.1, kernel='linear')
svc.fit(train_feats, train_cats)
logger.info('Accuracy of SVM model on test data: {}'.format(svc.score(test_feats, test_cats)))



