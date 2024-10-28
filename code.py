import pickle 
import pandas as pd
import seaborn as sns
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
import nltk.metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)
s_analyzer = SentimentIntensityAnalyzer()

trainingSet = pd.read_csv('./data/train.csv')
testingSet = pd.read_csv('./data/test.csv')
trainingSet['Score'] = trainingSet['Score']

trainingSet['Score'].value_counts().plot(kind='bar', alpha=.5)
plt.show()

features = [
    'Sentiment Score', 
    'Sentiment Summary', 
    'User_Confidence',
    'HelpfulnessNumerator', 
    'HelpfulnessDenominator', 
    'Review Length',
    'Word Count',
    'Summary Word Count',
    'user_mean_score',
    'user_std_score',
    'Helpfulness',
    'Sentiment_Length_Ratio',
    'user_review_count',
    'Help_Score_Ratio',
    'Help_Review_Ratio',
]
    
def add_user_features(df):
    userid_data = df.groupby('UserId').agg({
        'Score': ['mean', 'std', 'count', 'min', 'max', 'median']
    }).reset_index()
    
    userid_data.columns = [
        'UserId', 'user_mean_score', 'user_std_score', 'user_review_count', 
        'user_min_score', 'user_max_score', 'user_median_score'
    ]
    
    userid_data['user_std_score'] = userid_data['user_std_score'].fillna(0)
    
    return df.merge(user_stats, on='UserId', how='left')

def sentiment_analysis(text):
    if isinstance(text, str):
        return s_analyzer.polarity_scores(text)['compound']
    return 0

def add_features_to(df):
    df = df.copy()  
    
    df['Helpfulness'] = df['HelpfulnessNumerator'].div(df['HelpfulnessDenominator']).fillna(0)
    df['Review Length'] = df['Text'].str.len().fillna(0)
    df['Word Count'] = df['Text'].str.split().str.len().fillna(0)
    df['Summary Word Count'] = df['Summary'].str.split().str.len().fillna(0)
    df['Sentiment Score'] = df['Text'].apply(sentiment_analysis)
    df['Sentiment Summary'] = df['Summary'].apply(sentiment_analysis)
    
    df = add_user_features(df)
    
    df['user_mean_score'] = df['user_mean_score'].fillna(df['user_mean_score'].mean())
    df['user_std_score'] = df['user_std_score'].fillna(df['user_std_score'].mean())
    df['user_review_count'] = df['user_review_count'].fillna(1)
    df['Sentiment_Length_Ratio'] = df['Sentiment Score'] / (df['Review Length'] + 1)
    df['Help_Score_Ratio'] = df['Helpfulness'] * df['Sentiment Score']
    df['Help_Review_Ratio'] = df['HelpfulnessNumerator'] / (df['Review Length'] + 1)
    df['User_Confidence'] = df['user_review_count'] / df['user_std_score'].clip(lower=0.1)
    
    return df

if exists('./data/X_train.csv'):
    X_train = pd.read_csv('./data/X_train.csv')
    X_submission = pd.read_csv('./data/X_submission.csv')
else:
    train = add_features_to(trainingSet)
    X_submission = pd.merge(train, testingSet, on='Id', how='right')
    X_submission = X_submission.drop(columns=['Score_x']).rename(columns={'Score_y': 'Score'})
    X_train = train[train['Score'].notnull()]
    
    X_submission.to_csv('./data/X_submission.csv', index=False)
    X_train.to_csv('./data/X_train.csv', index=False)



X = X_train[features]
y = X_train['Score']

X_train_split, X_test_split, Y_train, Y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

X_submission_features = X_submission[features]

X_train_split = X_train_split.fillna(0)
X_test_split = X_test_split.fillna(0)
X_submission_features = X_submission_features.fillna(0)

forestClassification = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(estimator=forestClassification, threshold='mean')

X_train_selected = selector.fit_transform(X_train_split, Y_train)
X_test_selected = selector.transform(X_test_split)
X_submission_selected = selector.transform(X_submission_features)

selected_features_mask = selector.get_support()
selected_features = [f for f, selected in zip(features, selected_features_mask) if selected]


model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)

model.fit(X_train_selected, Y_train)

Y_test_pred = model.predict(X_test_selected)

print('Accuracy:', accuracy_score(Y_test, Y_test_pred))
print('\nClassification Report:')
print(classification_report(Y_test, Y_test_pred))

plt.figure(figsize=(10,8))
conf_matrix = confusion_matrix(Y_test, Y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


submission_pred = model.predict(X_submission_selected)
submission = pd.DataFrame({
    'Id': X_submission['Id'],
    'Score': submission_pred 
})
submission.to_csv('./data/submission.csv', index=False)
print("\nSubmission file created: submission.csv")