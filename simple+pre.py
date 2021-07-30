import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential

#%% 데이터 불러오기

train      = pd.read_csv("C:/Users/PC/Desktop/NLP/DACON/train_data.csv")
test       = pd.read_csv("C:/Users/PC/Desktop/NLP/DACON/test_data.csv")
submission = pd.read_csv("C:/Users/PC/Desktop/NLP/DACON/sample_submission.csv")
topic_dict = pd.read_csv("C:/Users/PC/Desktop/NLP/DACON/topic_dict.csv")

# IT과학 0 / 경제 1 / 사회 2 / 생활문화 3 / 세계	4 / 스포츠 5 / 정치 6
 
#%% 데이터 전처리

def clean_text(sent):
  sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
  return sent_clean

train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))

# replace hanja to hangul
train["title"] = train["title"].str.replace('美','미국 ')
test ["title"] = test ["title"].str.replace('美','미국 ')

train["title"] = train["title"].str.replace('韓','한국 ')
test ["title"] = test ["title"].str.replace('韓','한국 ')

train["title"] = train["title"].str.replace('日','일본 ')
test ["title"] = test ["title"].str.replace('日','일본 ')

train["title"] = train["title"].str.replace('獨','독일 ')
test ["title"] = test ["title"].str.replace('獨','독일 ')

train["title"] = train["title"].str.replace('靑','청와대 ')
test ["title"] = test ["title"].str.replace('靑','청와대 ')

train["title"] = train["title"].str.replace('北','북한 ')
test ["title"] = test ["title"].str.replace('北','북한 ')

train["title"] = train["title"].str.replace('英','영국 ')
test ["title"] = test ["title"].str.replace('英','영국 ')

train["title"] = train["title"].str.replace('中','중국 ')
test ["title"] = test ["title"].str.replace('中','중국 ')

train["title"] = train["title"].str.replace('伊','이탈리아 ')
test ["title"] = test ["title"].str.replace('伊','이탈리아 ')

train["title"] = train["title"].str.replace('UAE','아랍에미리트 ')
test ["title"] = test ["title"].str.replace('UAE','아랍에미리트 ')

train["title"] = train["title"].str.replace('EU','유럽 연합 ')
test ["title"] = test ["title"].str.replace('EU','유럽 연합 ')

train["title"] = train["title"].str.replace('與','여당 ')
test ["title"] = test ["title"].str.replace('與','여당 ')

train["title"] = train["title"].str.replace('軍','군대 ')
test ["title"] = test ["title"].str.replace('軍','군대 ')

train["title"] = train["title"].str.replace('軍','군대 ')
test ["title"] = test ["title"].str.replace('軍','군대 ')

train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))



train_text = train["cleaned_title"].tolist()
test_text = test["cleaned_title"].tolist()
train_label = np.asarray(train.topic_idx)

tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=150000, binary=False)

tfidf.fit(train_text)

train_tf_text = tfidf.transform(train_text).astype('float32')
test_tf_text  = tfidf.transform(test_text).astype('float32')

#%% Model

def dnn_model():
  model = Sequential()
  model.add(Dense(128, input_dim = 150000, activation = "elu"))
  model.add(Dropout(0.8))
  model.add(Dense(7, activation = "softmax"))
  return model

model = dnn_model()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.optimizers.Adam(0.001), metrics = ['accuracy'])

history = model.fit(x = train_tf_text[:40000], y = train_label[:40000],
                    validation_data =(train_tf_text[40000:], train_label[40000:]),
                    epochs = 4)

#%% Predict

tmp_pred = model.predict(test_tf_text)
pred = np.argmax(tmp_pred, axis = 1)

submission.topic_idx = pred
# submission.sample(3)

#path = "C:/Users/PC/Desktop/NLP/DACON/simple1.csv"
#submission.to_csv(path, index = False)

import pandas as pd

path = "C:/Users/PC/Desktop/NLP/DACON/simple2_pred.csv"
df = pd.DataFrame(pred)
df.to_csv(path, index = False)
print(df)
    
