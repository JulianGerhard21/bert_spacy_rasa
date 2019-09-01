import json
import spacy
import numpy as np
import logging.config
from rasa.nlu.utils import json_unpickle
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

# Set the paths
classifier_file = '<path_to_model>/nlu/component_3_SklearnIntentClassifier_classifier.pkl'
encoder_file = '<path_to_model>/nlu/component_3_SklearnIntentClassifier_encoder.pkl'

logger.info('Load Rasa classifier model')
classifier = json_unpickle(classifier_file)
classes = json_unpickle(encoder_file)
encoder = LabelEncoder()
encoder.classes_ = classes


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
test_cats, test_texts = zip(*test_data)

# Get the features of test data
logger.info('Get features of test data...')
test_feats = get_features(nlp, test_texts)

# Encode labels
test_labels = encoder.transform(test_cats)

preds = classifier.predict(test_feats)
logger.info('Micro: {}'.format(metrics.precision_score(test_labels, preds, average='micro')))
logger.info('Macro: {}'.format(metrics.precision_score(test_labels, preds, average='macro')))
logger.info('Weighted: {}'.format(metrics.precision_score(test_labels, preds, average='weighted')))
logger.info('Accuracy: {}'.format(classifier.score(test_feats, test_labels)))
