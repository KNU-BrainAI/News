import pandas as pd
import re
from konlpy.tag import Okt,Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score
from lightgbm import LGBMClassifier

#%% EDA

train = pd.read_csv('train_data.csv') # 데이터 불러오기
train.tail() # 데이터 확인
train.isnull().sum() # 결측값 확인
train.topic_idx.value_counts() # label 비율 확인
test = pd.read_csv('test_data.csv')

#%% 데이터 전처리

# 형태소 분석기(Okt) 불러오기 
m = Mecab('C:/mecab/mecab-ko-dic') 

# 조사, 어미, 구두점 제거
def func(text):
    clean = []
    for word in m.pos(text): #어간 추출
        if word[1] not in ['J', 'E', 'S']: #조사, 어미, 구두점 제외 
            clean.append(word[0])
    
    
    return " ".join(clean) 

train['title'] = train['title'].apply(lambda x : func(x))

#%% replace hanja to hangul
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




train["title"] = train["title"].apply(lambda x : func(x))
test["title"] = test["title"].apply(lambda x : func(x))

# tf-idf를 이용한 벡터화
def split(text):
    tokens_ko = text.split()
    return tokens_ko

tfidf_vect = TfidfVectorizer(tokenizer=split)
tfidf_vect.fit(train['title'])
tfidf_matrix_train = tfidf_vect.transform(train['title'])

#%%

# train/valid 데이터 셋 나누기.
def split_dataset(tfidf,df):
    X_data = tfidf
    y_data = df['topic_idx']

    # stratify=y_data Stratified 기반 분할, train 데이터의 30%를 평가 데이터 셋으로 사용. (70% 데이터 학습에 사용)
    X_train, X_test, y_train, y_test = \
    train_test_split(X_data, y_data, test_size=0.3, random_state=42, stratify=y_data)

    
    return (X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = split_dataset(tfidf_matrix_train,train)

#%% 모델 학습

lgbm = LGBMClassifier(random_state = 42)
lgbm.fit(X_train,y_train)

#%% 모델 평가

pred = lgbm.predict(X_test)
accuracy = accuracy_score(y_test,pred)

print('정확도', accuracy)

#%% test 데이터 예측

#test = pd.read_csv('test_data.csv')
test['title'] = test['title'].apply(lambda x : func(x)) 
tfidf_matrix_test = tfidf_vect.transform(test['title'])
pred = lgbm.predict(tfidf_matrix_test)

#%% 제출 파일

submission = pd.read_csv('sample_submission.csv')
submission['topic_idx'] = pred
submission.head()
submission.to_csv('mecab_pos+pre.csv',index = False)

#%% answer

answer = pd.read_csv('answer.csv')

train = pd.read_csv('train_data.csv')
result = pd.DataFrame( columns=['title','pred','answer'])
result['title'] = test['title']
result['pred'] = pred
result['answer'] = answer['topic_idx']

wrong_classification=pd.DataFrame( columns=['title','pred','answer'])

for i in range(len(result)):
    if (result.iloc[i]['answer']!=result.iloc[i]['pred']):
        wrong_classification = wrong_classification.append(result.loc[i])

accuracy =round( (len(result)-len(wrong_classification))/len(result) *100,4)

print("accuracy :"+str(accuracy)+"%")