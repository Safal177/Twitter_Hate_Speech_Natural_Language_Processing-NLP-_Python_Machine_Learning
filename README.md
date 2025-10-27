# Twitter_Hate_Speech_Natural_Language_Processing-NLP-_Python_Machine_Learning

# Project Title:
Help Twitter Combat Hate Speech Using Natural Language Processing (NLP) and Python Machine Learning

# Project Overview:
Users share perspectives and ideas in social media such as twitter. some tweets sadly indicate hate including partiality. There is a critical issue to classifying and handling high-volume content. Working as a data scientist in twitter, the project goal is to programmatically recognize and label tweets using Machine Learning (ML), and Natural Language Processing (NLP). Finally, twitter is safely categorized into hate or not content. 

# Key Tools: 
•	Coding language: Python,
•	Workspace: Jupyter Notebook, 
•	Libraries used: dotenv, NumPy, Pandas, , scikit-learn, NLTK
•	ML Model: Logistic Regression with key parameters (regularization and class weight)
•	NLP tools: dropping stopwords, tokenization, lemmatization, vectorization with tf-idf
•	Model Tuning: stratifiedkfold with cross validation, gridsearchcv with hyperparameter tuning
•	Evaluation Parameters: accuracy, recall, f1-score

# Primary Goal:
Main goal in this project is to make an fine-tuned natural language classification model and then correctly predict hate speech (racist or not) in tweets. I used the following seps to get this goal. 
•	Cleaning original text in tweet using NLP,
•	Normalizing original text in tweet using NLP,
•	Changing text into numerical form with the help of tf-idf
•	Training logistic regression model
•	Tuning logistic regression model
•	Refining logistic regression parameters
•	Quantifying model outcomes with accuracy, recall and f1-score used for hate speech detection.

# Major Contributions:
I built a tweet NLP preprocessing workflow to drop hashtags, urls, and non-alphanumeric symbols, and mentions. The keys are 
•	Engineered tweet cleaning pipeline used to remove urls, mentions, hashtags, and non-textual characters,
•	Leveraged nltk for tokenization,
•	Adopted tf-idf upto 5000 items,
•	Handled uneven class ratios with a parameter (class_weight = ‘balanced’),
•	Carried out GridSearchCV with StratifiedKFold used for configuration parameters to get reliable outcomes, 
•	 Secured reliable model outcomes after strengthened f1-score as well as recall for maximization results with very low number of hate tweets missed. 

# Closing Statement:
In this project, I clearly show that Python machine learning and NLP can solve the real-world business problems such as hate speech classification (racist or not) in Twitter. With the help of raw data preprocessing, feature engineering, optimization of model response pattern, logistic regression model showed unbiased operational success. The pipeline presented here can be used to high-volume datasets as well as cutting-edge deep learning systems such as BERT, LSTM for real-world operationalization. 

# Project file:
Twitter_Hate_Speech_ NLP_ML.ipynb
