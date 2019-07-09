from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.externals import joblib


# In[172]:

#rf = RandomForestClassifier(n_estimators=100000, n_jobs=-1, max_depth = 10)
#lr = LogisticRegression(random_state=10, solver='liblinear', max_iter=1000000, penalty='l1', C=0.01, n_jobs=-1)
filename = 'insurance_trained_model.joblib'
rf = joblib.load(filename)


result_set = pd.read_csv("result_test.csv")

result_set=result_set.drop('fraud_reported', axis=1)
result_set=result_set.drop('policy_bind_date', axis=1)
result_set=result_set.drop('incident_location', axis=1)
result_set=result_set.drop('policy_number', axis=1)
result_set['policy_csl']=result_set['policy_csl'].map({'250/500': 0.5, '100/300': 0.333333333333, '500/1000':  0.5})
result_set['insured_education_level']=result_set['insured_education_level'].map({'MD' : 6, 'PhD': 7, 'Associate': 4, 'Masters': 5, 'High School': 1, 'College': 2, 'JD': 3})
result_set['incident_severity']=result_set['incident_severity'].map({'Major Damage': 3, 'Minor Damage': 2, 'Total Loss': 4, 'Trivial Damage': 1})
lis=[]
str_lis=[]
for x in result_set:
    if(type(result_set[x][0]) != str):
        lis.append(x)
    else:
        str_lis.append(x)
result_set = result_set[lis][:]


fill_miss = Imputer(missing_values='NaN',strategy='median',copy=True)
result_dummy_data = pd.get_dummies(result_set)
complete_result_dummy_data = fill_miss.fit_transform(result_dummy_data.values)
complete_result_data = pd.DataFrame(complete_result_dummy_data)


results = rf.predict(complete_result_data)
print("this is final result ",results)




