# Finetune BERT Embeddings with spaCy and Rasa

** Update 13.10.2019 ** 

Repository was updated to be compatible with the latest changes to spacy-pytorch-transformers which were renamed to spacy-transformers. I recommend to uninstall the old library before installing the new one. The training script
was adapted accordingly.

**For whom this repository might be of interest:**

This repository describes the process of finetuning the *german pretrained BERT* model of [deepset.ai](https://deepset.ai/german-bert)
on a domain-specific dataset, converting it into a [spaCy](https://spacy.io/) packaged model and loading it in [Rasa](https://rasa.com/) to evaluate its
performance on domain-specific **Conversational AI** tasks like *intent detection* and *NER*.
If there are questions though, feel free to ask.

This repository is meant for those who want to have a quick dive into the matter. 

I am going to use the [10kGNAD](https://tblock.github.io/10kGNAD/) dataset for this task but it should be easy to
modify the files for your specific use case.

**Short-term Roadmap**:

* Publish evaluation results on various scenarios
* Add CUDA Installation Guide
* Add [RoBERTa](https://arxiv.org/abs/1907.11692) to the current approach
* Compare against other transformers like [GPT-2](https://github.com/openai/gpt-2) or [XLNet](https://arxiv.org/abs/1906.08237)
* Add [BERT distillation](http://www.nlp.town/blog/distilling-bert/)
* Add NER support

___
## Installation

### Requirements

Basically all you need to to is execute:

```
pip install -r requirements.txt
```

The scripts are tested using the following libraries:

* python = 3.6.8
* spacy = 2.2.1
* spacy-transformers = 0.5.0
* rasa = 1.3.9

Please keep in mind that some of the dependencies are work in progress and there might be inter-incompatibilities. 
However, at the time of writing this, the libraries can simply be installed by using `pip`.

I strongly suggest do finetune and test BERT with GPU support since finetuning on even a good CPU
can last several hours per epoch. 
___
### Getting started

#### Preparing the dataset

The *split* is done by the finetuning script. If you want to have a different setting,
feel free to modify the script.

As suggested, we do a simple but stratified train-test split with 15% as the test subset and 85% as the training subset, which results in 8732 training
samples and 1541 evaluation samples. As there are many possibilities left, this is only one
possible approach. While converting the `articles.csv` into a pandas dataframe, there were some broken lines
which currently are omitted.
___
#### Loading the pretrained BERT

The script assumes the pretrained BERT to be installed with:

```
python -m spacy download de_trf_bertbasecased_lg
```

For the sake of interest, I have added the ``bert_config.json`` from Deepset's awesome work
if someone wonders how the ``de_trf_bertbasecased_lg`` was trained.
___
#### Finetune the pretrained BERT

You can start the finetuning process by using:

```
python bert_finetuner_splitset.py de_trf_bertbasecased_lg -o finetuning\output
```

Currently, I am using a ```softmax_pooler_ouput``` configuration for the ``trf_textcat``component.
I'd suggest a ``softmax_last_hidden`` as the next approach. The other parameters
were set based on several evaluations and might be modified for your specific use case.
___
#### Package the finetuned BERT with spaCy and install it

You can easily package your newly trained model by using:

```
python -m spacy package finetuning/output /packaged_model
cd /packaged_model/de_trf_bertbasecased_lg-1.0.0
python setup.py sdist
pip install dist/de_trf_bertbasecased_lg-1.0.0.tar.gz
```

I recommend **changing the model's name** to avoid unnecessary inconveniences
by editting the config file and modifying the ``name`` value of `/finetuning/output/meta.json`.

___
#### Load the spaCy model as part of your Rasa pipeline (optional)

At the time of writing this, BERT outperforms most of the recent state-of-the-art approaches
in NLP/NLU tasks, e.g. document classification. 
Since those techniques are used in several **conversational AI** tasks like **intent detection**, I thought it might be a good idea to evaluate its performance with **Rasa** - IMHO one of the
best open source CAI engines currently available.

If someone is interested in building a chatbot with Rasa, it might be a good idea to read the
[Getting started](https://rasa.com/docs/getting-started/) guide.

Assuming that someone is familiar with Rasa, here is one possible configuration proposal which
loads the newly added finetuned BERT model as a part of the training pipeline:

```
language: de
pipeline: 
 - name: SpacyNLP
   case_sensitive: 1
   model: de_trf_bertbasecased_lg_gnad
 - name: SpacyTokenizer
 - name: SpacyFeaturizer
 - name: SklearnIntentClassifier
```

As you can see, I just specified the model's name, using the spaCy architecture with
Rasa. This works, even if ``python -m spacy validate`` does **not** show your model.

Assuming that you might want to test the performance with Rasa, you can use the ``test_bot`` directory
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

**bert_classify** evaluates the finetuned BERT by training a logistic regression
and a simple SVM classifier.

```
python -m bert_classify.py 
```

**bert_rasa_classify** loads the trained Rasa model and uses the pretrained BERT features to evaluate the
model's performance on the test data. Keep in mind that Rasa *compresses* your model, so you simply
have to unzip/untar it and also modify the path to the NLU model in the script.

```
python -m bert_rasa_classify.py 
```

Please be aware of the fact that to evaluate the **generalization capabilities** of the model,
it would be better to split the original dataset into three parts such that there is a dataset
completely unknown by the model (i.e. train/validation/test split).
___
#### Productive usage of a large BERT model

TBD
___

#### A note on NER (Named Entity Recognition)

As soon as I realized that I won’t be able to use the finetuned BERT-spaCy model in rasa for e.g. extracting entities like PERSON (in fact, duckling is currently not able to do that), I thought about how this would be done in general:

1. Use the SpacyFeaturizer and SpacyEntityExtractor which currently would be recommended but which is not possible due to manual effort on the side of BERT (as mentioned, I am working on that).
2. Finetuning the pretrained BERT that afterwards is converted into a spaCy-compatible model on any NER dataset is absolutely possible and intended. We can finetune the BERT on both tasks alongside. If so, the model contains everything we are going to need to derive entities from it. Currently just not with spaCy directly. Instead we could use a CustomBERTEntityExtractor which loads the model that the pipeline already has loaded and do the work, that spaCy is currently not “able” to do.

3. Since 2 seems to be an overhead at least for the moment, why not do the following:
```
language: de
pipeline: 
 - name: SpacyNLP
   case_sensitive: 1
   model: de_trf_bertbasecased_lg_gnad
 - name: SpacyTokenizer
 - name: SpacyFeaturizer
 - name: SklearnIntentClassifier
 - name: SpacyNLP
   case_sensitive: 1
   model: de_core_news_md
 - name: RegexFeaturizer
 - name: CRFEntityExtractor
 - name: DucklingHTTPExtractor
   dimensions: ['time', 'duration', 'email']
   locale: de_DE
   timezone: Europe/Berlin
   url: http://localhost:8001
 - name: SpacyEntityExtractor
   dimensions: ['PER', 'LOC', 'CARDINAL']
 - name: rasa_mod_regex.RegexEntityExtractor
 - name: EntitySynonymMapper

```
This pipeline will then load and use the features of de_trf_bertbasecased_lg_gnad for SklearnIntentClassifier, and the features of de_core_news_md for SpacyEntityExtractor.

This is not a neat solution and it should only be used until there is a smarter way (1,2) but it works.

It should be mentioned, that of course you are able to even train your own with spaCy.


#### Troubleshooting


##### CUDA Out of Memory

As discussed in a [spacy-trf-issue](https://github.com/explosion/spacy-pytorch-transformers/issues/48) you may run into
memory problems. I have tested the finetuning script on a *GTX 1080 with 8GB VRAM* and even with a batch size of
2 (which is absolutely *not* recommended), I got memory problems.

One way to deal with it is to use the sentencizer which splits larger documents into sentences while keeping their original labels.
Another way is to reduce the batch size by half, to 12. BERT models usually need bigger batches but for the sake of functionality, I tried it.

Currently I am using a *T80 with 12 GB VRAM*, sentencizing and a lowered batch size and that setup worked fine.


##### AttributeError: module 'thinc_gpu_ops' has no attribute 'mean_pool'

As discussed [here](https://github.com/explosion/spacy-pytorch-transformers/issues/27) you might run into the mentioned
error. I was able to resolve it by manually cloning thinc-gpu-ops, running ``pip install -r requirements.txt`` (that actually installed cython) and then running ``pip install`` .

___




A *thank you* goes to all of the **amazing open source workers** out there:

* [Rasa](https://github.com/RasaHQ)
* [spaCy](https://github.com/explosion/spaCy)
* [Deepset](https://deepset.ai/german-bert)
* [HuggingFace](https://github.com/huggingface/pytorch-transformers)
* [MKaze](https://github.com/mkaze/)



