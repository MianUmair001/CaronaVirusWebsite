import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import time
import math
import operator
import sys
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
plt.style.use('seaborn')

confirmed_cases=pd.read_csv('covid19_confirmed_global.csv')


deaths_reported=pd.read_csv('covid19_deaths_global.csv')


recovered_cases=pd.read_csv('covid19_recovered_global.csv')


confirmed_cases.head()

deaths_reported.head()


recovered_cases.head()



cols=confirmed_cases.keys()



confirmed=confirmed_cases.loc[:,cols[4]:cols[-1]]

deaths=deaths_reported.loc[:,cols[4]:cols[-1]]


recoveries=recovered_cases.loc[:,cols[4]:cols[-1]]


confirmed.head()



dates=confirmed.keys()
world_cases=[]
total_deaths=[]
mortality_rate=[]
total_recovered=[]




for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=deaths[i].sum()
    recovered_sum=recoveries[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)

confirmed_sum


death_sum



recovered_sum


world_cases

days_since_1_4=np.array([i for i in range(len(dates))]).reshape(-1,1)


world_cases=np.array(world_cases).reshape(-1,1)


total_deaths=np.array(total_deaths).reshape(-1,1)

total_recovered=np.array(total_recovered).reshape(-1,1)


days_in_future=10
future_forcast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjust_dates=future_forcast[:-10]




start='1/22/2020'
start_date=datetime.datetime.strptime(start,'%m/%d/%Y')
future_forcast_dates=[]
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%Y'))



latest_confirmed=confirmed_cases[dates[-1]]
latest_deaths=deaths_reported[dates[-1]]
latest_recoveries=recovered_cases[dates[-1]]



unique_countries=list(confirmed_cases['Country/Region'].unique())



country_confirmed_cases=[]
no_cases=[]
for i in unique_countries:
    cases=latest_confirmed[confirmed_cases['Country/Region']==i].sum()
    if cases >0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(cases)
for i in no_cases:
    unique_countries.remove(i)
    
unique_countries=[k for k,v in sorted(zip(unique_countries,country_confirmed_cases),key=operator.itemgetter(1),reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i]=latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()




print('Confirmed Cases by Country/Region')
for i in range(len(unique_countries)):
    print(f'{unique_countries[i]}:{country_confirmed_cases[i]} cases')


unique_provinces=list(confirmed_cases['Province/State'].unique())
outliers= ['United Kingdom', 'Denmark', 'France']


province_confirmed_cases=[]
no_cases=[]
for i in unique_provinces:
    cases=latest_confirmed[confirmed_cases['Province/State']==i].sum()
    if cases >0:
        province_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
for i in no_cases:
    unique_provinces.remove(i)


print('Confirmed Cases by Province/State')
for i in range(len(unique_provinces)):
    print(f'{unique_provinces[i]}:{province_confirmed_cases[i]} cases')


nan_indexes=[]
for i in range(len(unique_provinces)):
    if type(unique_provinces[i])==float:
        nan_indexes.append(i)

unique_provinces=list(unique_provinces)
province_confirmed_cases=list(province_confirmed_cases)
for i in nan_indexes:
    unique_provinces.pop(i)
    province_confirmed_cases.pop(i)


plt.figure(figsize=(32, 32))
plt.barh(unique_countries,country_confirmed_cases)
plt.title('Number of COV19 confirmed cases in Countries')
plt.xlabel('NUmber of COV19 confirmed cases')
plt.savefig('static/uploads/gallery_img-01.jpg')


china_confirmed=latest_confirmed[confirmed_cases['Country/Region']=='China'].sum()





outside_mainland_china_confirmed=np.sum(country_confirmed_cases)-china_confirmed




plt.figure(figsize=(16,9))
plt.barh('MainLand China',china_confirmed)
plt.barh('Outside Mainland China',outside_mainland_china_confirmed)
plt.title('NUmber of confirmed COV19 cases')
plt.savefig('static/uploads/gallery_img-02.jpg')




print('Outside MianLand China {} cases'.format(outside_mainland_china_confirmed))
print('MianLand China {} cases'.format(china_confirmed))
print('Total Cases: {}'.format(outside_mainland_china_confirmed+china_confirmed))



visual_unique_countries=[]
visual_confirmed_cases=[]
others=np.sum(country_confirmed_cases[10:])
for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])
    
    
visual_unique_countries.append('Others')
visual_confirmed_cases.append(others)



plt.figure(figsize=(32,18))
plt.barh(visual_unique_countries,visual_confirmed_cases)
plt.title('NUmber of confirmed COV19 cases in Country/Region',size=20)
plt.savefig('static/uploads/gallery_img-03.jpg')



c=random.choices(list(mcolors.CSS4_COLORS.values()),k=len(unique_countries))
plt.figure(figsize=(20,20))
plt.title('COV19 confirmed cases per countries')
plt.pie(visual_confirmed_cases,colors=c)
plt.legend(visual_unique_countries,loc='best')
plt.savefig('static/uploads/gallery_img-04.jpg')



c=random.choices(list(mcolors.CSS4_COLORS.values()),k=len(unique_countries))
plt.figure(figsize=(20,20))
plt.title('COV19 confirmed cases Outside MainLand China')
plt.pie(visual_confirmed_cases[1:],colors=c)
plt.legend(visual_unique_countries[1:],loc='best')
plt.savefig('static/uploads/gallery_img-05.jpg')



kernel=['poly','sigmoid','rbf']
c=[0.01,0.1,1,10]
gamma=[0.01,0.1,1]
epsilon=[0.01,0.1,1]
shrinking=[True,False]
svm_grid={'kernel':kernel,'C':c,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}
svm=SVR()
svm_search=RandomizedSearchCV(svm,svm_grid,scoring='neg_mean_squared_error',cv=3,return_train_score=True,n_jobs=-1,n_iter=40,verbose=1)





X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_4, world_cases, test_size=0.15, shuffle=False)


svm_search.fit(X_train_confirmed,y_train_confirmed)


svm_search.best_params_


svm_confirmed=svm_search.best_estimator_
svm_pred=svm_confirmed.predict(future_forcast)



svm_test_pred=svm_confirmed.predict(X_test_confirmed)


plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)
print("MAE:",mean_absolute_error(svm_test_pred,y_test_confirmed))
print("MSE:",mean_squared_error(svm_test_pred,y_test_confirmed))



plt.figure(figsize=(20,12))
plt.plot(adjust_dates,world_cases)
plt.title("NUmber of COV19 cases over time",size=30)
plt.xlabel("Days since 1/22/2020",size=30)
plt.ylabel("Number of cases",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/uploads/gallery_img-06.jpg')



plt.figure(figsize=(20,12))
plt.plot(adjust_dates,world_cases)
plt.plot(future_forcast,svm_pred,linestyle='dashed',color='purple')
plt.title("NUmber of COV19 cases over time",size=30)
plt.xlabel("Days since 1/22/2020",size=30)
plt.ylabel("Number of cases",size=30)
plt.legend('confirmed_cases','SVM Predicted')
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/uploads/gallery_img-07.jpg')




print('SVM future prediction')
set(zip(future_forcast_dates[-10:],svm_pred[-10:]))



from sklearn.linear_model import LinearRegression
linear_model=LinearRegression(normalize=True,fit_intercept=True)
linear_model.fit(X_train_confirmed,y_train_confirmed)
test_linear_pred=linear_model.predict(X_test_confirmed)
linear_pred=linear_model.predict(future_forcast)



print("MAE:",mean_absolute_error(test_linear_pred,y_test_confirmed))
print("MSE:",mean_squared_error(test_linear_pred,y_test_confirmed))




plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.savefig('static/uploads/gallery_img-08.jpg')



plt.figure(figsize=(20,12))
plt.plot(adjust_dates,world_cases)
plt.plot(future_forcast,linear_pred,linestyle='dashed',color='purple')
plt.title("NUmber of COV19 cases over time",size=30)
plt.xlabel("Days since 1/22/2020",size=30)
plt.ylabel("Number of cases",size=30)
plt.legend('confirmed_cases','SVM Predicted')
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/uploads/gallery_img-09.jpg')



print("Linear Regression Future Predition")
print(linear_pred[-10:])


plt.figure(figsize=(20,12))
plt.plot(adjust_dates,total_deaths,color='red')
plt.title("Number of COV19 deaths over Time",size=30)
plt.xlabel("Time",size=30)
plt.ylabel("Number of Deaths",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/uploads/gallery_img-10.jpg')



mean_mortality_rate=np.mean(mortality_rate)
plt.figure(figsize=(20,12))
plt.plot(adjust_dates,mortality_rate,color='orange')
plt.axhline(y=mean_mortality_rate,linestyle='--',color='black')
plt.title("Mortality Rate of COV19 over Time",size=30)
plt.legend('Mortality Rate','y='+str(mean_mortality_rate))
plt.xlabel("Time",size=30)
plt.ylabel("Mortality Rate",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/uploads/gallery_img-11.jpg')


plt.figure(figsize=(20,12))
plt.plot(adjust_dates,total_recovered,color='green')
plt.title("Number of COV19 Recovered over Time",size=30)
plt.xlabel("Time",size=30)
plt.ylabel("Number of Cases",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/uploads/gallery_img-12.jpg')



plt.figure(figsize=(20,12))
plt.plot(adjust_dates,total_deaths,color='red')
plt.plot(adjust_dates,total_recovered,color='green')
plt.legend(['deaths','recovered'],loc='best',fontsize=20)
plt.title("Number of COV19 Casese",size=30)
plt.xlabel("Time",size=30)
plt.ylabel("Number of Cases",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/uploads/gallery_img-13.jpg')


plt.figure(figsize=(20,12))
plt.plot(total_deaths,total_recovered)
plt.plot(adjust_dates,color='green')
plt.title("COV19 Deaths vs Recoveries",size=30)
plt.xlabel("Total Number of COV19 Deaths",size=30)
plt.ylabel("Total Number of COV19 Recoveries",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig('static/uploads/gallery_img-14.jpg')





plt.figure(figsize=(20,12))
plt.plot(adjust_dates,world_cases)
plt.plot(adjust_dates, world_cases, 'bo')
plt.plot(world_cases)
plt.plot(world_cases, 'r+') 
plt.savefig('static/uploads/gallery_img-15.jpg')


