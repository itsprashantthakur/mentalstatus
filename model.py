# importing libraries that are required
import pandas as pd
import numpy as np
import pickle
data = pd.read_csv('depression.csv')

# change column names so that they are readable
data.columns = ['SEQN', 'lack_interest', 'feel_depressed', 'trouble_sleeping', 'feel_tired',
               'bad_apetite', 'feel_bad', 'lack_focus', 'rate_speech', 'wanna_die', 'difficulty']

# fixing data issues such as missing values, and float values being rounded off
data = data.round()
data.drop(['difficulty'], axis=1, inplace=True)
data.dropna(inplace=True)

# Converting the feel_depressed feature into a binary class feature
def convert(x):
    if x < 3:
        return 0
    else:
        return 1

data['depressed'] = data['feel_depressed'].apply(lambda x: convert(x))
data.drop('feel_depressed', axis=1,inplace=True)

# preparing the data for machine learning
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')

# Applying logistic regression
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X, y)

# Testing the model with random inputs
input = [int(x) for x in "2 2 2 1 2 0 3 1".split(' ')]
final = [np.array(input)]

p = logit.predict_proba(final)
print('prob  is:', p)

# exporting the model for deployment
# Save the model as a pickle in a file
pickle.dump(logit, open('logit_model.pkl','wb'))
model = pickle.load(open('logit_model.pkl','rb'))
