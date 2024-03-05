# Fighting against Fake News on Newly-Emerging Crisis: A Case Study of COVID-19
- [Migyeong Yang](https://sites.google.com/view/migyeong-yang), Chaewon Park, [Daeun Lee](https://sites.google.com/view/daeun-lee), [Jiwon Kang](https://ji1kang.github.io/), [Daejin Choi](https://daejin-choi.github.io/), [Jinyoung Han](https://sites.google.com/site/jyhantop/), Fighting against Fake News on Newly-Emerging Crisis: A Case Study of COVID-19, ACM Web Conference 2024 (WWW '24 Companion).


# Dataset
- COVID-19 Fake News Claim
	- [FakeCovid](https://gautamshahi.github.io/FakeCovid/)
	- [CoAID](https://github.com/cuilimeng/CoAID) 
- Non-COVID-19 Fake News Claim 
	- Crawled from [Politifact](https://www.politifact.com/) and [Snopes](https://www.snopes.com/)
- Topic Keyword
  - Extracted through [BERT-NER](https://github.com/kamalkraj/BERT-NER) and [KeyBERT](https://doi.org/10.5281/zenodo.4461265)
- Propagation Data
  - Crawled from YouTube with [GoogleAPI](https://developers.google.com/youtube/)
- To download the dataset, please send email to mgyang@g.skku.edu.

# Code
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
