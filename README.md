# Fighting against Fake News on Newly-Emerging Crisis: A Case Study of COVID-19
- [Migyeong Yang](https://sites.google.com/view/migyeong-yang), Chaewon Park, [Daeun Lee](https://sites.google.com/view/daeun-lee), [Jiwon Kang](https://ji1kang.github.io/), [Daejin Choi](https://daejin-choi.github.io/), [Jinyoung Han](https://sites.google.com/site/jyhantop/), Fighting against Fake News on Newly-Emerging Crisis: A Case Study of COVID-19, ACM Web Conference 2024 (WWW '24 Companion).


# Dataset
This folder includes the whole dataset that we collected. 

### Components
- COVID-19 Fake News Claim
	- [FakeCovid](https://gautamshahi.github.io/FakeCovid/)
	- [CoAID](https://github.com/cuilimeng/CoAID) 
- Non-COVID-19 Fake News Claim 
	- Crawled from [Politifact](https://www.politifact.com/) and [Snopes](https://www.snopes.com/)
- Topic Keyword
  - Extracted through [BERT-NER](https://github.com/kamalkraj/BERT-NER) and [KeyBERT](https://doi.org/10.5281/zenodo.4461265)
- Propagation Data
  - Crawled from YouTube with [GoogleAPI](https://developers.google.com/youtube/)
 
### File Explanation
1. **[train_ex1.pickle]** Data used for training in Experiment 1, containing non-COVID-19 news claims.
2. **[train_ex2.pickle]** Data used for training in Experiment 2, containing non-COVID-19 news claims and COVID-19 news claims published before April 1, 2020.
3. **[test.pickle]** Data used for testing in Experiment 1 and Experiment 2, containing COVID-19 news claims published after April 1, 2020.


# Data
This folder contains only the data necessary for model training.

### File Explanation
1. **[train_ex1.pkl]** A version extracted from './dataset/train_ex1.pickle', including news claims (title), ner_onehot, titles and descriptions of two YouTube videos, and labels.
2. **[train_ex2.pkl]** A version extracted from './dataset/train_ex2.pickle', including news claims (title), ner_onehot, titles and descriptions of two YouTube videos, and labels.
3. **[test.pkl]** A version extracted from './dataset/test.pickle', including news claims (title), ner_onehot, titles and descriptions of two YouTube videos, and labels.
4. **[train_ex1.npy / train_ex2.npy / test.npy]** These are numpy-formatted adjacency matrices required for training and testing.


# Model
This folder consists of Python scripts for the model.

### Get Started
Python 3.7 & Pytorch 1.10.1
```
pip install -r requirements.txt
```

### Run
For Exp. 1 (Without COVID-19 Data)
```
python3 ./model/main-ex1.py
```

For Exp. 2 (With a few COVID-19 Data)
```
python3 ./model/main-ex2.py
```
