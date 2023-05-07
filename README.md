# Duplicate Question Checker Application

## Overview : 

Quora is a platform where people can ask and answer questions, with around **100 Million** monthly visitors. Many people ask similar questions, which can be time-consuming for followers to answer repeatedly. To solve this, I worked on a project to predict whether two questions are duplicates or not. I started with basic models like **XGBoost** and **Random Forest** and gradually improved their accuracy. Additionally, I trained the data using an **LSTM** model and NLP TRnasformers like **BERT** , **ROBERTa**. This system could potentially help improve efficiency on Quora by detecting duplicate questions. 


## The Dataset : 

- **id** - the id of a training set question pair
- **qid1**, **qid2** - unique ids of each question (only available in train.csv)
- **question1**, **question2** - the full text of each question
- **is_duplicate** - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

![image](https://user-images.githubusercontent.com/103372852/236662206-6fc1a147-1fe4-405e-8564-4c428912bfb8.png)


## Experimentation : 

### 1. Extracting  Basic Features 
- Following Basic feature are extracted and added to orignal dataset for training
   - **freq_qid1** = Frequency of qid1's #ie, number of times question1 occur
   - **freq_qid2** = Frequency of qid2's
   - **q1len** = Length of q1
   - **q2len** = Length of q2
   - **q1_n_words** = Number of words in Question 1
   - **q2_n_words** = Number of words in Question 2
   - **word_Common** = (Number of common unique words in Question 1 and Question 2)
   - **word_Total** =(Total num of words in Question 1 + Total num of words in Question 2)
   - **word_share** = (word_common)/(word_Total)
   - **freq_q1+freq_q2** = sum total of frequency of qid1 and qid2
   - **freq_q1-freq_q2** = absolute difference of frequency of qid1 and qid2
   
   
  #### Results : 
  - We can observe that sentences which have common words are similar to each other .i.e features who have more word share probability  are similar to each other 
<img src="https://user-images.githubusercontent.com/103372852/236662629-db2a75bb-a83a-40f4-a6bf-ab0aa2037652.png" alt="description of image" width="50%" height="30%">
 
 - I have trained **Random Forest** and **XGBOOST** on this basic features  and used ** Bag of Words** , I gotFollowing  results : 
 
| Model         | Accuracy   | ROC-AUC | F1-Score | 
| ----           | ------     | ------  |--------- |
| Random Forest(BOW)  | 82.75 %      | 79.07 %   |  73.60 %    |
| XGBoost   (BOW)     | 83.76 %   | 81.86 %  |   77.29 %     |

- We can see **XGBoost** Model performed well on this data with Accuracy **83.76%** , so i tried to add more advanced  features and applied **Glove Embedding** 

### 2. Extracting Advanced Features :
#### 1. Token Features
- cwc_min: This is the ratio of the number of common words to the length of the smaller question
- cwc_max: This is the ratio of the number of common words to the length of the larger question
- csc_min: This is the ratio of the number of common stop words to the smaller stop word count among the two questions
- csc_max: This is the ratio of the number of common stop words to the larger stop word count among the two questions
- ctc_min: This is the ratio of the number of common tokens to the smaller token count among the two questions
- ctc_max: This is the ratio of the number of common tokens to the larger token count among the two questions
- last_word_eq: 1 if the last word in the two questions is same, 0 otherwise
- first_word_eq: 1 if the first word in the two questions is same, 0 otherwise

#### 2. Length Based Features
- mean_len: Mean of the length of the two questions (number of words)
- abs_len_diff: Absolute difference between the length of the two questions (number of words)

#### 3. Fuzzy Features
- fuzz_ratio: fuzz_ratio score from fuzzywuzzy
- fuzz_partial_ratio: fuzz_partial_ratio from fuzzywuzzy
- token_sort_ratio: token_sort_ratio from fuzzywuzzy
- token_set_ratio: token_set_ratio from fuzzywuzzy

- With  Some More advanced Features i got following results with **Bag Of Words**

| Model         | Accuracy   | ROC-AUC | F1-Score | 
| ----           | ------     | ------  |--------- |
| Random Forest(BOW)  | 82.65 %      | 80.20 %   |  75.33 %    |
| XGBoost   (BOW)     | 83.57 %   | 81.83 %  |   77.44 %     |

- We can see Accuracy is not much increased using Advanced features , Let's try different approach. 


### 3. Training With  LSTM , with and without  Glove : 

- I have first trained data  using **LSTM** Model without **Glove embeddings** , then i applied **Glove Embedding** . I got following results: 

| Model         | Accuracy   | ROC-AUC | F1-Score | 
| ----           | ------     | ------  |--------- |
| LSTM (without Glove)  | 74.03 %      | 7.47 %   |  63.65 %    |
| LSTM (with Glove)    | 76.16 %   | 74.41 %  |   67.66 %     |

- Accuracy is improved by 2.13 %  when i applied  **Glove** to the same architecture of **LSTM**. 
- We can observe that accuracy is more when we applied feature engineering and used  **Machine Learning Models** which was around 83.76%. 
-  Our **XGboost with BOW** model is still better than **LSTM with Glove**

### 4. Using NLP Transformers : 

- I have finetuned  **BERT** Encoder [click here](https://huggingface.co/bert-base-uncased)  which is available on  **HuggingFace-Hub**. **BERT** is pretrained on large English corpus , so it will automatically extract features when we finetune this model on our custome dataset.
- Surprisingly **BERT** gave me **90 %** Accuracy  which is much better than previous models

- Additionally , I finetuned **ROBERTa** Encoder [click here](https://huggingface.co/roberta-base) which also available on **HuggingFace-Hub**. 
- 

- Results of **BERT** and **ROBERTa** 


| Model          | Accuracy | train loss | 
| ----           | ------  | --------------|
| BERT           | 90 %      | 0.1196
| ROBERTa        | 90 %   | 0.20 | 

- **BERT** and **ROBERTa** are performing very well on data with accurcay around 90 %


### 5. How to use this project for  checking duplicate questions  : 

- clone this repo : `$ git clone https://github.com/Vinayakmane47/duplicate_question_checker_NLP.git`
- create new environment : `$ conda create -n qachecker python=3.7 -y`
- install requirements : `pip install -r requirements.txt` 
- run app.py : `python app.py`

### 6. Results : 

![image](https://user-images.githubusercontent.com/103372852/236672512-23baf525-d152-4bb4-a142-2e3ce55864e3.png)

### 7. Application URL : 

[Question Similarity Checker](https://huggingface.co/spaces/VinayakMane47/Check_duplicate_questions)














   




