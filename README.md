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

# Related
## References
http://www.deepnlp.org/blog/ <br>
http://www.deepnlp.org/equation/ <br>
http://www.deepnlp.org/search/ <br>
http://www.deepnlp.org/workspace/ai_courses/ <br>
http://www.deepnlp.org/workspace/aigc_chart/ <br>
http://www.deepnlp.org/workspace/ai_writer/ <br>
http://www.deepnlp.org/workspace/detail/ <br>
[AI IMAGE GENERATOR](http://www.deepnlp.org/store/image-generator) <br>
[AI Search Engine](http://www.deepnlp.org/store/search-engine)  <br>
[AI Chatbot Assistant](http://www.deepnlp.org/store/chatbot-assistant)  <br>
[AI for ELDERLY](http://www.deepnlp.org/store/elderly)  <br>
[AI for KIDS](http://www.deepnlp.org/store/kids)  <br>
[AI in LAW](http://www.deepnlp.org/store/law) <br>
[AI in FINANCE](http://www.deepnlp.org/store/finance) <br>
[AI in HEALTHCARE](http://www.deepnlp.org/store/healthcare)  <br>
[AI in BUSINESS](http://www.deepnlp.org/store/business)  <br>
[AI in EDUCATION](http://www.deepnlp.org/store/education) <br>
[AI in PRODUCTIVITY TOOL](http://www.deepnlp.org/store/productivity-tool) <br>
[AI in POLITICS](http://www.deepnlp.org/store/politics) <br>
[AI in ENTERTAINMENT](http://www.deepnlp.org/store/entertainment) <br>
[AI in NEWS](http://www.deepnlp.org/store/news) <br>
[AI in ART AND SPORTS](http://www.deepnlp.org/store/art-and-sports) <br>
[AI in LIFESTYLE](http://www.deepnlp.org/store/lifestyle) <br>
[AI in PAYMENT](http://www.deepnlp.org/store/payment) <br>
[AI in SOCIAL](http://www.deepnlp.org/store/social) <br>
[AI in AGRICULTURE](http://www.deepnlp.org/store/agriculture) <br>
[AI in SCIENCE](http://www.deepnlp.org/store/science) <br>
[AI in TECHNOLOGY](http://www.deepnlp.org/store/technology) <br>
[AI in TRAVEL](http://www.deepnlp.org/store/travel) <br>
[AI in TRANSPORTATION](http://www.deepnlp.org/store/transportation) <br>
[AI in CAR](http://www.deepnlp.org/store/car) <br>
[AI in CHARITY](http://www.deepnlp.org/store/charity) <br>
[AI in PUBLIC SERVICE](http://www.deepnlp.org/store/public-service) <br>
[AI in HOUSING](http://www.deepnlp.org/store/housing) <br>
[AI in COMMUNICATION](http://www.deepnlp.org/store/communication) <br>
[AI in FOOD](http://www.deepnlp.org/store/food) <br>



## Related Blog
http://www.deepnlp.org/blog/ <br>
http://www.deepnlp.org/equation/ <br>
http://www.deepnlp.org/search/ <br>
http://www.deepnlp.org/workspace/ai_courses/ <br>
http://www.deepnlp.org/workspace/aigc_chart/ <br>
http://www.deepnlp.org/workspace/ai_writer/ <br>
http://www.deepnlp.org/workspace/detail/ <br>
[Statistics Equation Formula](http://www.deepnlp.org/blog/statistics-equations-latex-code) <br>
[Machine Learning Equation Formula](http://www.deepnlp.org/blog/latex-code-machine-learning-equations) <br>
[AI Courses for Kids](http://www.deepnlp.org/blog/how-to-use-generative-ai-to-draw-paw-patrol-dog-skye) <br>
[AI in Fashion: Tell IWC Schaffhausen Watches Real or Fake](http://www.deepnlp.org/blog/how-to-tell-iwc-schaffhausen-watches-real-or-fake-20-steps) <br>
[AI in Fashion: Tell Fendi bags real or fake](http://www.deepnlp.org/blog/how-to-tell-fendi-bags-real-or-fake-20-steps) <br>
[AI in Fashion: Tell Coach bags real or fake](http://www.deepnlp.org/blog/how-to-tell-coach-bags-real-or-fake-20-steps) <br>
[AI in Fashion: Tell Prada bags real or fake](http://www.deepnlp.org/blog/how-to-tell-prada-bags-real-or-fake-20-steps) <br>
[AI in Fashion: Tell Gucci bags real or fake](http://www.deepnlp.org/blog/20-tricks-to-tell-gucci-bags-real-or-fake) <br>
[AI in Fashion: Tell Dior bags real or fake](http://www.deepnlp.org/blog/tell-dior-bags-real-or-fake-20-steps) <br>
[AI in Fashion: Tell Hermes bags real or fake](http://www.deepnlp.org/blog/20-tricks-to-tell-hermes-bags-real-or-fake) <br>
[AI in Fashion: Tell Chanel bags real or fake](http://www.deepnlp.org/blog/20-tricks-to-tell-chanel-bags-real-or-fake) <br>
[AI in Fashion: Tell Louis Vuitton bags real or fake](http://www.deepnlp.org/blog/20-tricks-to-tell-louis-vuitton-bags-real-or-fake) <br>
[AI in Fashion: Tell Omega Watches real or fake](http://www.deepnlp.org/blog/20-tricks-to-tell-if-omega-watch-is-real-or-fake) <br>
[AI in Fashion: Tell Rolex Watches real or fake](http://www.deepnlp.org/blog/20-tricks-to-tell-if-rolex-watch-is-real-or-fake) <br>
