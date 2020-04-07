from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import string


driver = webdriver.Chrome()

pakistan_confirmed_cases=0
recovered=0
deaths=0
critical=0
cases_24hrs=0
deaths_24hrs=0
Test_24hrs=0
Total_Tests=0
ICT=0
PUNJAB=0
SINDH=0
KP=0
BALOCHISTAN=0
AJK=0
GB=0
driver.get("http://covid.gov.pk")


content = driver.page_source
soup = BeautifulSoup(content)
for a in soup.findAll('div', attrs={'id':'statistics'}):
    Pakistan_Overall=a.find('div', attrs={'class':'row'})
    Provinces_Overall=a.find('div', attrs={'class':'row provinc-stat'})
    pakistan_confirmed_cases=a.find('h1',attrs={'class':"text-muted numbers-main"})

Pakistan_Overall=Pakistan_Overall.text
Provinces_Overall=Provinces_Overall.text
pakistan_confirmed_cases=pakistan_confirmed_cases.text


Pakistan_Overall=Pakistan_Overall.replace('\n',' ')
Provinces_Overall=Provinces_Overall.replace('\n',' ')
Pakistan_Overall=Pakistan_Overall.replace('(24','')
Pakistan_Overall=Pakistan_Overall.replace('HRS)','')
Pakistan_Overall=Pakistan_Overall.replace(',','')
Provinces_Overall=Pakistan_Overall.replace(',','')

Pakistan_Overall=Pakistan_Overall.split()
Provinces_Overall=Provinces_Overall.split()

Pakistan_Overall_text=[]
Pakistan_Overall_Numbers=[]
Provinces_Overall_text=[]
Provinces_Overall_Numbers=[]



for i in range(len(Pakistan_Overall)):
    if(Pakistan_Overall[i].isalpha()):
        Pakistan_Overall_text.append(Pakistan_Overall[i])
        print(i,Pakistan_Overall[i])

for i in range(len(Pakistan_Overall)):
    if(Pakistan_Overall[i].isnumeric()):
        Pakistan_Overall_Numbers.append(Pakistan_Overall[i])
        print(i,Pakistan_Overall[i])

for i in range(len(Provinces_Overall)):
    if(Provinces_Overall[i].isalpha()):
        Provinces_Overall_text.append(Provinces_Overall[i])

for i in range(len(Provinces_Overall)):
    if(Provinces_Overall[i].isnumeric()):
        Provinces_Overall_Numbers.append(Provinces_Overall[i])

pakistan_confirmed_cases=Pakistan_Overall_Numbers[0]
recovered=Pakistan_Overall_Numbers[1]
deaths=Pakistan_Overall_Numbers[2]
critical=Pakistan_Overall_Numbers[3]
cases_24hrs=Pakistan_Overall_Numbers[4]
deaths_24hrs=Pakistan_Overall_Numbers[5]
Test_24hrs=Pakistan_Overall_Numbers[6]
Total_Tests=Pakistan_Overall_Numbers[7]


ICT=Provinces_Overall_Numbers[0]
PUNJAB=Provinces_Overall_Numbers[1]
SINDH=Provinces_Overall_Numbers[2]
KP=Provinces_Overall_Numbers[3]
BALOCHISTAN=Provinces_Overall_Numbers[4]
AJK=Provinces_Overall_Numbers[5]
GB=Provinces_Overall_Numbers[6]


def getData():
    data=[
        pakistan_confirmed_cases,recovered,critical,cases_24hrs,deaths_24hrs,Test_24hrs,Total_Tests,ICT,PUNJAB,SINDH,KP,BALOCHISTAN,AJK,GB   
    ]
    return data


