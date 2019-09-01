# Finetune BERT Embeddings with SpaCy and Rasa

This repository describes the process of finetuning the *german pretrained BERT* model of [deepset.ai](https://deepset.ai/german-bert)
on a domain specific dataset, converting it into a [SpaCy](https://spacy.io/) packaged model and loading it in [Rasa](https://rasa.com/) to evaluate its
performance on domain specific **Conversational AI** tasks like *intent detection* and *NER*.


I am going to use the [10kGNAD](https://tblock.github.io/10kGNAD/) dataset for this task but it should be easiy to
modify the files for your specific case.

**Shortterm Roadmap**:

* Sentencize training samples
* Add [RoBERTa](https://arxiv.org/abs/1907.11692) to the current approach
* Compare against other transformers like [GPT-2](https://github.com/openai/gpt-2) or [XLNet](https://arxiv.org/abs/1906.08237)
* Add [BERT distillation](http://www.nlp.town/blog/distilling-bert/)
___
## Installation

### Requirements

The scripts are tested on the following libraries:

* python = 3.6.8
* spacy = 2.1.8
* spacy-pytorch-transformers = 0.3.0
* rasa = 1.2.5

Please keep in mind that some of the dependencies are work in progress and there might be interincompatibilities. 
However, at the time writing this, the libraries can simply be installed by using `pip`.
___
### Getting started

#### Preparing the dataset

The *split* is done by the finetuning script. If you want to have a different setting,
feel free to modify the script.

As suggested, we do a simple but stratified train test split with 15% test and 85% train which results in 8732 training
samples and 1541 evaluation samples. As there are many possibilities left, this is only one
possible approach. While converting the articles.csv into a pandas dataframe, there were some broken lines
wich currently are omitted.
___
#### Loading the pretrained BERT

The script assumes the pretrained BERT to be installed with:

```
python -m spacy download de_pytt_bertbasecased_lg
```

For the sake of interest, I have added the ``bert_config.json`` from Deepset's awesome work
if someone wonders how the ``de_pytt_bertbasecased_lg`` was trained.
___
#### Finetune the pretrained BERT

You can start the finetuning process by using:

```
python bert_finetuner_splitset.py de_pytt_bertbasecased_lg -o finetuning\output
```

Currently I am using a ```softmax_pooler_ouput``` configuration for the ``pytt_textcat``component.
I'd suggest a ``softmax_last_hidden`` as the next approach. The other parameters
were set based on several evaluations and might be modified for your specific case.
___
#### Package the finetuned BERT with SpaCy and install it

You can easily package your newly trained model by using:

```
python -m spacy package finetuning/output /packaged_model
cd /packaged_model/de_pytt_bertbasecased_lg-1.0.0
python setup.py sdist
pip install dist/de_pytt_bertbasecased_lg-1.0.0.tar.gz
```

I recommend **changing the models name** to avoid unnecessary inconveniences
by editting the config file and modifying the ``name`` value of:

```
/finetuning/output/meta.json
```

___
#### Load the SpaCy model as a part of your rasa pipeline (optional)

At the time writing this, BERT outperforms most of the recent state-of-the-art approaches
in NLP/NLU tasks, e.g. document classification. 
Since those techniques are used in several **conversational AI* tasks like **intent detection**
I thought it might be a good idea, to evaluate its performance with **Rasa** - imho one of the
best open source CAI engines currently available.

If someone is interested in building a chatbot with Rasa, it might be a good idea to read the
[Getting started](https://rasa.com/docs/getting-started/).

Assuming, that someone is familiar with Rasa, here is one possible configuration proposal, which
loads the newly added, finetuned BERT model as a part of the training pipeline:

```
language: de
pipeline: 
 - name: SpacyNLP
   case_sensitive: 1
   model: de_pytt_bertbasecased_lg_gnad
 - name: SpacyTokenizer
 - name: SpacyFeaturizer
 - name: SklearnIntentClassifier
```

As you can see, I just specified the model's name, using the SpaCy architecture with
Rasa. This works, even if ``python -m spacy validate`` does **not** show your model.

Assuming that you might want to test the performance with Rasa, you can use the ``test_bot`` folder
which contains the skeletton for a Rasa bot to do so. In advance, use:

```
python rasa_bot_generator.py
cp test.md test_bot/test_data/
cp train.md test_bot/data/
cd test_bot
rasa train --data data/ -c config.yml -d domain.yml --out models/
rasa run -m models/ --enable-api
```

to create a valid ``stories.md`` and a valid ``domain.yml``. Please keep in mind that
this will be a minimal sample from which I don't recommend to use it productively.

If the bot is loaded, you can use the endpoint:

```
http://localhost:5005/model/parse

POST
{
	"text": "<any article you want to get its domain for>"
}

```
___
#### Evaluate different pipelines

To keep things simple, there are two scripts which will do the work for you.

**bert_classify** evaluates the finetuned BERT itsself by training a logistic regression
and a simple SVM.

```
python -m bert_classify.py 
```

**bert_rasa_classify** loads the trained Rasa model and uses the pretrained BERTs features to evaluate the
model's performance on the test data. Keep in mind that Rasa *compresses* your model and you simply
have to unzip/untar it and to modify the path to the nlu model.

```
python -m bert_rasa_classify.py 
```

Please be aware of the fact, that to evaluate the **generalization capabilities** of the model,
it would be better to split the original dataset into three parts such that there is a dataset
completely unknown by the model.
___
#### Productive usage of a large BERT model

TBD
___

A *thank you* goes to all of the **amazing open source workers** out there:

* [Rasa](https://github.com/RasaHQ)
* [SpaCy](https://github.com/explosion/spaCy)
* [SpaCy PTT](https://github.com/explosion/spacy-pytorch-transformers)
* [Deepset](https://deepset.ai/german-bert)
* [HuggingFace](https://github.com/huggingface/pytorch-transformers)



