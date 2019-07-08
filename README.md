## context_recommendation
Contextual Recommendation Implementation for Research Purposes.

## Environment
python: python2/python3
tensorflow: 1.6.0
numpy: 1.15.3
cPickle/_pickle: 1.71

### 1. Dataset
Yelp Dataset(The experiment in paper is conducted on Version 7).
Latest dataset (version 9) can be downloaded from: https://www.kaggle.com/yelp-dataset/yelp-dataset.  (There are some minor changes in different versions of dataset, e.g. 'neighbourhood' features, etc.)

Unzip the archive file and you can see: yelp_academic_dataset_review.json, yelp_academic_dataset_business.jsonm, yelp_academic_dataset_user.json

Yelp Dataset(version 7) is no longer on the kaggle website. You can download dataset (version 7) of the paper's experiment and pretrain/train/test .pkl pickle files from the cloud link as below: 
Link: https://pan.baidu.com/s/1EUm05wn88bDru-LTxuCETw, Password:f1ej

``` python
# Generate Pretraining/Train/Test Examples:
python contextual_dataset_yelp.py

# output multiple files 
# Pretrain: ../data/yelp/yelp-dataset/yelp_pretrain_dataset.pkl
# train:    ../data/yelp/yelp-dataset/train/yelp_train_examples_*.pkl   * is the 1-10
# test:     ../data/yelp/yelp-dataset/test/yelp_test_examples_*.pkl     * is the 1-2

You can also download these data files(version 7) since generating dataset is time-consuming.
Link: https://pan.baidu.com/s/1EUm05wn88bDru-LTxuCETw
Password:f1ej

python contextual_dataset_yelp_old.py
```

### 2. Pretraining using MACDAE model on Yelp Dataset
``` python
# Pretrain the recommendation model using multi-heads number as 4
python contextual_macdae_yelp.py 4
```

### 3. Training Recommendation model
``` python
# BASE: Wide&Deep
python recommd_context_pretrain_yelp.py base train

# BASE+MACDAE: Wide&Deep + Pretrain the recommendation model using multi-heads number as 4
python recommd_context_pretrain_yelp.py macdae train
```

### 4. Evaluating Recommendation model
``` python
## BASE(Wide & Deep), input x dimension (339)
# Example Test Run Result: NDCG@5 is 0.3827, NDCG@10 is 0.4348

python recommd_context_pretrain_yelp.py base test

## BASE(Wide & Deep) + MACDAE， input x dimension (403)
# Example Test Run Result: NDCG@5 is 0.4382, NDCG@10 is 0.4854

python recommd_context_pretrain_yelp.py macdae test

```
