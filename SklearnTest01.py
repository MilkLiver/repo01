# -*- coding: utf-8 -*-
#QAQ
"""
Created on Mon Feb 18 13:55:44 2019

@author: NEIL_YU
"""

from sklearn import preprocessing,ensemble,metrics
import numpy as np

#資料標籤
gender=['male','female']
money_list=[22000,32000,40000,50000,70000]

#將標籤轉換成數值形式
gender_label=preprocessing.LabelEncoder()
gender_label.fit(gender)

#訓練用資料
o_genders=np.random.randint(1,3,5000)
o_age=np.random.randint(15,25,5000)
o_money=np.random.choice(money_list,5000,replace=True)

genders=o_genders[:-10].reshape(1,-1).T
age=o_age[:-10].reshape(1,-1).T
money=o_money[:-10].reshape(1,-1).T

peoples=np.array([])
for info in zip(genders,age):
    peoples=np.append(peoples,[*info])

ageSalaryModel=ensemble.AdaBoostRegressor(n_estimators=20,random_state=10)
ageSalaryModel.fit(peoples.reshape(-1,2),money.ravel())

testOwO=np.array([['male',23],['female',16],['male',20],['female',23],['male',17],['female',20]]).T
testdata=np.array([])

testdata_age=testOwO[1]
testdata_gender=gender_label.transform(testOwO[0])
for i in zip(testdata_gender,testdata_age):
    testdata=np.append(testdata,[*i])

#=============================================================================
print(testdata)
predict_money=ageSalaryModel.predict(testdata.reshape(-1,2)).round(2)
print(predict_money)

ageSalaryModel.fit(testdata.reshape(-1,2),predict_money)
print(ageSalaryModel.predict(testdata.reshape(-1,2)))

#=============================================================================
test_genders=o_genders[40:].reshape(1,-1).T
test_age=o_age[40:].reshape(1,-1).T
test_money=o_money[40:]

test_peoples=np.array([])
for info in zip(test_genders,test_age):
    test_peoples=np.append(test_peoples,[*info])

test_predict=ageSalaryModel.predict(test_peoples.reshape(-1,2)).astype(int)
print(test_money)
print(test_predict)
print(sum((test_predict==test_money)*1))
#print(sum(test_predict==test_money))


