# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 21:39:53 2019

@author: 
"""
import time
import matplotlib.pyplot as plt
start = time.time()
import pandas as pd
import numpy as np
import sklearn

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet

import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import svm
#load data from DB to pandas
from sqlalchemy import create_engine
#from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection import GridSearchCV
#Config of variables for control of models  
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


training=True #flag of training
start_exam=True #flag of entry exams
#

# data partitional in case of BIG DATA
def part(start,finish):
    finish_=finish
    df1=pd.read_sql_query('SELECT id, subject, student_year_id, score, title, examined_at, expectation FROM results WHERE id > %s and id< %s', cnx, params=(start,finish))
    for i in range(1,6):
       try: 
          print(i)
          start=finish_
          finish=str(int(finish_)+int(finish))
          print(finish)
          df11=pd.read_sql_query('SELECT id, subject, student_year_id, score, title, examined_at, expectation FROM results WHERE id BETWEEN %s and %s', cnx, params= (start,finish))
          df1=pd.concat([df1,df11])
       except:
           print ('error')
    return df1
    
# normalization of marks
def calc_start_exam(score, exam_category):
    
    if exam_category=='CAT4':
          result = (score*100)/180
    elif exam_category=='DRT': 
          result = score*10
    else: 
          print('Not CAT4 or DRT')
    return result 


#load data from postgres
    
#postgres_str='postgres://remote_user:0e766b1810c5cd042000c00fb9ee85c1@46.101.58.30:5432/academic_tracker_production'

#product
    
postgres_str='postgres://ml_limited_user:p418b74d6bf350b2405574e62835d37bbcce86c2ad8c34d5a5b80f01975625d58@ec2-18-203-229-185.eu-west-1.compute.amazonaws.com:5432/d9k69uia8l4iu'

cnx = create_engine(postgres_str)

df_schools = pd.read_sql_query('''SELECT id, name FROM schools''', cnx)
df_schools_=df_schools[(df_schools['name'].str.contains("Test*")==False) & (df_schools['name'].str.contains("TEST*")==False)]
#df_schools_=df_schools_[df_schools_['name'].str.contains("TEST*")==False]

#df2=pd.read_sql_query('''SELECT id, student_id, academical_year, calendar_year FROM student_years''', cnx)

#df1=pd.read_sql_query('''SELECT id, subject, student_year_id, score, title, examined_at,
#       expectation FROM results''', cnx)


###
###df1=part('0', '50000')
###df1=df1.rename(columns = {'id': 'idu'})
#df=pd.read_sql_query('''SELECT results.id, results.subject, results.score, results.expectation, student_years.student_id,  student_years.academical_year, student_years.calendar_year FROM results INNER JOIN student_years ON results.student_year_id = student_years.id ORDER BY results.id''', cnx)
#df3=pd.read_sql_query('''SELECT id, `, gender, year_of_entry, name FROM students''', cnx)

####df8=df3.merge(df_schools_,  how='inner', left_on='school_id', right_on='id')

df4=pd.read_sql_query('''SELECT * FROM intro_results;''', cnx)

#normalization of exam marks 
df4['score_start'] = df4.apply(lambda row: calc_start_exam(row['score'],  row['exam_category']), axis=1)

#mean of exam marks 
df4=df4.groupby(['student_id'])['score_start', 'student_id'].mean()
df4=df4.rename(columns = {'student_id': 'student_id_group'})
#

#df=df2.merge(df1,  how='inner', left_on='id', right_on='student_year_id')
#df=df.merge(df3,  how='inner', left_on='student_id', right_on='id')

#df=pd.read_sql_query('SELECT results.id, results.subject, results.score, results.expectation, results.title, results.examined_at, student_years.student_id, student_years.academical_year, student_years.calendar_year,  students.school_id, students.gender, students.year_of_entry FROM results INNER JOIN student_years sy ON sy.id = results.student_year_id INNER JOIN students ON students.id = sy.student_id WHERE students.school_id = 51;', cnx)
df=pd.read_sql_query('SELECT results.id, results.subject, results.score, results.expectation, results.title, results.examined_at, student_years.student_id, student_years.academical_year, student_years.calendar_year,  students.school_id, students.gender, students.year_of_entry FROM results INNER JOIN student_years ON student_years.id = results.student_year_id INNER JOIN students ON students.id = student_years.student_id', cnx)
id_df=df.id

uniq =df.student_id.unique()
student_id_=[x for x in uniq]

df=df.rename(columns = {'id': 'idu'})
#df0_=df
#df_=df.merge(df4,  how='inner', left_on='student_id', right_on='student_id_group')
#df=df.merge(df4,  how='inner', left_on='student_id', right_on='student_id_group')

df=df.merge(df_schools_,  how='inner', left_on='school_id', right_on='id')

#df0 - data without  entry exams
df0=df

#df - data with entry exams
df=df.merge(df4,  how='inner', left_on='student_id', right_on='student_id_group')
#df2,df3,df4= 0,0,0
df4= 0

#for filter of schools
#if school_inp >=0 : df=df[df.school_id == school_inp]



# filter of exam date < time of current exam
def last(examined_at_t):
    time_= examined_at_t
    date_=[x for x in spisok_t_ if x < time_]
    if len(date_)>0:
          result = date_[-1]
    else: 
          result=0
    return result 


#Model of regression
def model_(data_model):
    #from sklearn.metrics.pairwise import pairwise_kernels
    #print(len(data_model))
    #print(data_model.columns)
    Y = data_model.iloc[:,[2]]
    if start_exam==True:
        X_=data_model.columns[6]
        #print(X_)
        X = data_model.iloc[:,[6,7,8]]
        #print(X.columns)
        filename = [data_model.columns[2]+'_'+ data_model.columns[6]+'X0'+'.sav']
    else:
        X = data_model.iloc[:,[5,6]]
        #print(X.columns)
#        print(X)
        X_=data_model.columns[5]
        filename = [data_model.columns[2]+'_'+ data_model.columns[5]+'.sav']
    
    # Regression
    #print(X)
   
    
    # Ridge Regression
    # GridSearchCV of best parameters
#    regr_rbf = svm.SVR(kernel="rbf")
#    C=[1000, 10, 1]
#    gamma_ = [1]
#    epsilon = [0.1, 0.01]
#    parameters = {"C":C, "gamma":gamma_, "epsilon":epsilon}
#    kr = GridSearchCV(regr_rbf, parameters, scoring="r2")
    #clf.fit(train_data, train_labels)
    
#    etr = ExtraTreesRegressor(n_jobs=2, random_state=0)
#    param_grid = {'n_estimators': [50], 'max_features': [1,2,3]}
#    kr = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=4, cv=None, scoring='r2')
        
#    kr.fit(X, Y)
    
    
    
    
    ###
    kr = GridSearchCV(KernelRidge(gamma=1, coef0=1), cv=None, 
                  param_grid = {"alpha": [1, 1e-1, 1e-2],
              "kernel":['rbf']}, scoring='r2')
    kr.fit(X, Y)
    #y_kr = kr1.predict(X)
    #print(y_kr)
    #plt.scatter(Y, list(y_kr))
    #plt.show()
    #print(filename)
    #
    #Estimate of Ridge Regression model
    #y_kr = kr.predict(X)
    #print(X[0:1])
    #print(kr.predict(X[0:1]))
    
    #print(Y-y_kr)
    
#    print(kr.score(X,Y))
    #plt.scatter(X, Y)
    #plt.show()
#    err_kr = np.sqrt(mean_squared_error(Y,y_kr))
    #err_l=kr.score(X,Y)  
    #print(err_l)
    #save of model
    
    pickle.dump(kr, open(filename[0], 'wb'))
    
    loaded_model = pickle.load(open(str(filename[0]), 'rb'))
    forecast_score = loaded_model.predict(X[0:1])
    #print(forecast_score[0])
    
    model = LinearRegression().fit(X,Y)
    r_sq = model.score(X,Y)
       
    #save model results to table
    model_results.append([data_model.columns[2], X_, model.intercept_, model.coef_, r_sq, kr.score(X,Y), len(data_model), X[0:1], forecast_score, data_model.iloc[0:1,[2]]])  

#Model of Log regression
def model_log(data_model):
        
    Y = data_model.iloc[:,[2]]
    X = data_model.iloc[:,[4]]
    
    model1 = LogisticRegression()
    model1.fit(X, Y)
    print(model1.score(X, Y))
    print(model1.get_params())
    
#Model of Lasso regression

    
# Use regression model for forecasting
    
    
    
    
def dX_(student_data_, Y_name, Y_date):
          #global spisok_r
    try:  
          #print(Y_name)
          #print(Y_date)
          subject_r = student_data_[(student_data_.examined_at< Y_date) & (student_data_.subject== Y_name)]
          spisok_r= subject_r.sort_values(by =['examined_at'], ascending=False)[0:1]
          #print(spisok_r.examined_at[0])
          #print(spisok_r.examined_at.values[0])
          # Do we have previous records for forecasting?
          X_=student_data_[student_data_.examined_at< Y_date]
          
          if len(X_)<1:
              forecast_score=None
              return forecast_score
          else:
#                X_=student_data_.loc[student_data_.examined_at== Y_date]
               # use data of entry exams
#                X_s=X_[0:1]
               # 
#                try:
#                     forecast_score=X_s.score_start.values[0]
                #     print('!!!!forecast_score=X_s.score_start.values[0]')
#                except:
                 #    print('!!!!forecast_score=None')
                      
#                     forecast_score=None
                     #return forecast_score
                #print(spisok_r.examined_at.values[0])
                Subject_X =[x for x in X_.subject if X_.examined_at.values[0]==spisok_r.examined_at.values[0]]
                r=0
          # If we have data then we can get best formula of regression
          
                if    start_exam==False: 
                        df_m_=df_m0
                        #print('!!!')
                else: 
                        df_m_=df_m 
          
                try:
                    for x in Subject_X:
                        formula_=df_m_.loc[(df_m_.Y==Y_name) & (df_m_.X == x)]
                        
                        if formula_.R.values>r:
                            r=formula_.R.values                        
                            formula=formula_
                  
                                         
                #print(formula_)
          
                    X_= X_[(X_.subject==formula.X.values[0]) & (X_.examined_at.values==spisok_r.examined_at.values[0])]
          #
                    if start_exam==True and flag==True:
              #
                          try:
                              #print('regr 1')
                 
                              forecast_score= float(formula.A0.values[0][0])+float(formula.A1.values[0][0][0])*float(X_.score.values[-1]) + float(formula.A1.values[0][0][1])*float(X_.score_start.values[-1]) + float(formula.A1.values[0][0][2])*X_.academical_year.values[-1]
                              #print(forecast_score)
                          except:
                                  forecast_score= None
                    else:
                        #print('regr 2')
                        forecast_score= float(formula.A0.values[0][0])+float(formula.A1.values[0][0][0])*float(X_.score.values[-1]) + float(formula.A1.values[0][0][1])*X_.academical_year.values[-1]
                    #print(X_.score.values[-1])
                    #print(X_.score)
                    #print(forecast_score)
                except:
                    forecast_score= None  
    except:
           forecast_score= None
    return forecast_score

def dX_kr(student_data_, Y_name, Y_date):
    
    #Ridge Regression
    try:  
          subject_r = student_data_[(student_data_.examined_at< Y_date) & (student_data_.subject== Y_name)]
          spisok_r= subject_r.sort_values(by =['examined_at'], ascending=False)[0:1]
          #print(spisok_r)
          #print('!-!')
          X_=student_data_.loc[student_data_.examined_at< Y_date]
          
          
          if len(X_)<1:
               #X_=student_data_.loc[student_data_.examined_at== Y_date]
               # 
#                 print('none')
               # X_s=X_[0:1]
               # 
#                try:
#                     forecast_score=X_s.score_start.values[0]
                #     
#                except:
#                 #   
                 forecast_score=None
                 return forecast_score
          #print('0')  
          #Subject_X =[x for x in X_.subject]
          Subject_X =[x for x in X_.subject if X_.examined_at.values[0]==spisok_r.examined_at.values[0]]
          #print(Subject_X)
          r=0
          #
          if start_exam==False: df_m_=df_m0
            
          else: df_m_=df_m    
          
          #print('1')
          
          for x in Subject_X:
              
              formula_=df_m_.loc[(df_m_.Y==Y_name) & (df_m_.X == x)]
              #print(formula_)
              #print(len(formula_))
              #print(formula_.Err_kr.values[0])
              if len(formula_)>0:
                 #print("r00") 
                 if formula_.Err_kr.values[0]>r:
                     #print("r0")
                     r=formula_.Err_kr.values[0]
                     #print(r)
                     #print("r1")
                     formula=formula_[0:1]
                     #print(formula)
                     #print(formula.X.values[0])
          #print(formula.X.values[0])
          
          X_= X_[(X_.subject==formula.X.values[0]) & (X_.examined_at.values==spisok_r.examined_at.values[0])]
          #print(X_)
          X_=X_.sort_values(by =['examined_at'], ascending=False)
          #
          #print(X_)
          if start_exam==True and flag==True:
             #print('Ridge Regression 1') 
             namefile=Y_name +'_'+ X_.subject.values +'X0'+'.sav' 
             #print(namefile)
             #b=np.array([X_.score.values[-1], X_.score_start.values[-1], X_.academical_year.values[-1]])
             #b=b.reshape(1,-1)
             #
             loaded_model = pickle.load(open(str(namefile[0]), 'rb'))
             b=X_[['score','score_start', 'academical_year']]
             #print(b)
             #print('!')
             #print(b[0:1])
#             forecast_score=loaded_model.predict(b.reshape(1, -1))
             forecast_score_=loaded_model.predict(b[0:1])
             #print(loaded_model.predict([[92, 53, 1]]))
             forecast_score=forecast_score_[0][0]
             #print(forecast_score)
             #print(forecast_score[0][0])
          else:
             #print('Ridge Regression 2')
             namefile=Y_name +'_'+ X_.subject.values+'.sav'
             #print(namefile)
             #print(X_.score)
             #print(X_.examined_at)
             b=[[X_.score.values[-1], X_.academical_year.values[-1]]]
             #print(b)
             #b=np.array([b])
             loaded_model = pickle.load(open(str(namefile[0]), 'rb'))
             #forecast_score=loaded_model.predict(b.reshape(1, -1))
#             forecast_score=loaded_model.predict(b.reshape(1, -1))
             forecast_score=loaded_model.predict(b)
             forecast_score=forecast_score[0][0]
             #forecast_score=forecast_score[0][0]
             #print(forecast_score)
          
                                     
    except:
          forecast_score=None
    return forecast_score


def dX_exp(student_data_, Y_score, Y_date):
    #Model of Smoothing
    try:
        X_=student_data_[student_data_.examined_at< Y_date]
        if len(X_)==0:
                X_=student_data_.loc[student_data_.examined_at== Y_date]
                #print(X_)
                X_s=X_[0:1]
                #print(X_s)
                if start_exam==True or flag==True:
                    forecast_score=X_s.score_start.values[0]
                else:
                    forecast_score=None
                #print(forecast_score)
                return forecast_score
        elif len(X_) > 0:
            F_=X_.score
            forecast= F_.rolling(2, min_periods=1).mean()
            forecast_score=forecast.values[-1]
        else: 
            forecast_score= None
    except:
        forecast_score= None 
    return forecast_score
          
def student_(student_data):
    #Choose of students for calculate expectation 
    df_student=student_data
    #print(df_student)
    df_student['expectation_regr'] = df_student.apply(lambda row: dX_(df_student, row['subject'], row['examined_at']), axis=1)
    df_student['expectation_kr'] = df_student.apply(lambda row: dX_kr(df_student, row['subject'], row['examined_at']), axis=1)
    list_subjects=student_data.subject.unique()
    for i in list_subjects:
        
        df_student=student_data.loc[student_data['subject']==i]
        
        if len(df_student)>0:
                        
            df_student['expectation_exp'] = df_student.apply(lambda row: dX_exp(df_student, row['score'], row['examined_at']), axis=1)
        if i== list_subjects[0]:
            df_allsubjects_student=df_student
        else:
            df_allsubjects_student=pd.concat([df_allsubjects_student,df_student]) 
    return df_allsubjects_student



def create_models(df, start_exam):
    global spisok_t_  
    
    if training == True:
              
   # 
        data_all=[x for x in spisok]

   # 
        for j in range(len(spisok)):
            if start_exam==True:
               m=s[j][['student_id','examined_at', 'score', 'score_start', 'academical_year']]
            else:
               #print('+') 
               m=s0[j][['student_id','examined_at', 'score', 'academical_year']]
   #rename column 'score' on name of subject   
            first=m.rename(columns = {'score': spisok[j]}).sort_values('examined_at')
            #
   
            first=first.rename(columns = {'examined_at': 'examined_at_t'})
            spisok_t_=[x for x in first.examined_at_t.unique()] 
            first['examined_at'] = first.apply(lambda row: last(row['examined_at_t']), axis=1)
            #print(first[first.student_id==7181])
            #print('F')
      # 
            data_all[j]=[]
   
   #merge DF for first data subject and second data subject
            for i in range(len(spisok)):
                if start_exam==True:
                    m=s[i][['student_id','examined_at', 'score', 'score_start', 'academical_year']]
                else:
                   # print('++') 
                    m=s0[i][['student_id','examined_at', 'score', 'academical_year']]
             
         #rename column 'score' on name of subject 
                second=m.rename(columns = {'score': spisok[i]}).sort_values('examined_at')
                #print(first[first.student_id==7181])
                #print('F')
                #print(second[second.student_id==7181])
                #print('S')
         #print(j, spisok[j], spisok[i])
                data_all[j].append(pd.merge(first, second, on=['examined_at','student_id']))
             #
#                if len(data_all[j][i])>6 and (abs(data_all[j][i].corr().iloc[2].filter(regex=data_all[j][i].columns[2])[0])>0.15) and (j!=i):
                if len(data_all[j][i])>10 and (j!=i):
                     #print(data_all[j][i].corr())
                     #print(data_all[j][i][data_all[j][i].student_id==7181])
                     #print('!!!')
                     model_(data_all[j][i])
                          
                else: 
                     print('NOT for '+str(j)+' '+ str(i))

              
#Create df for all model results              
        df_m1=pd.DataFrame(model_results, columns=['Y', 'X', 'A0', 'A1', 'R', 'Err_kr', 'Set', 'Test_X0', 'Test_Y0', 'Y0'])              

    else:
        print('No training') 
    return df_m1
# forecasting for subject on base score X


#********** MAIN 
    
spisok=[x for x in df.subject.unique()]
spisok0=[x for x in df0.subject.unique()]

spisok_t_=[x for x in df.examined_at.unique()]
spisok_t=sorted(spisok_t_)

chars_ = ['A','ABS','B','C','D','E','N/A','NA','NG','#']

s=[]
s0=[]
df=df.sort_values(['subject', 'examined_at'])
df0=df0[-df0.score.isin(chars_)]
df=df[-df.score.isin(chars_)]
for i in spisok:
    #
    s.append(df[(df.subject==i)])
    s0.append(df0[(df0.subject==i)])



if start_exam==True:
     model_results=[]
     df_m0=[]
     start_exam=False     
     df_m0=create_models(df0, start_exam)
     start_exam=True
     model_results=[]
     df_m=[]     
     df_m=create_models(df, start_exam)
     
else: 
     model_results=[]
     df_m0=[] 
     df_m0=create_models(df0, start_exam)


#uniq=df0[df0.name == '68070E'].student_id.unique()
#uniq=df0.student_id.unique()
#student_id_=[x for x in uniq]
#student_id_=uniq

   
for i in range(0,len(student_id_)):
    st=df.loc[df['student_id']==student_id_[i]]
    if len(st)>0:
       #if student_id_[i].item() in set3:
          flag=True #We have data of entry exams
          df_student=student_(st)
    #else: flag=True
    else:
         #if student_id_[i].item() in set3:
          flag=False #We have not  data of entry exams
          st=df0.loc[df0['student_id']==student_id_[i]]
          if len(st)>0: df_student=student_(st)
              
    #else: flag=True
    
    
    if i== 0:
          df_all=df_student
    else:
          df_all=pd.concat([df_all,df_student])
    

import psycopg2
from psycopg2 import sql
#conn = psycopg2.connect(dbname='academic_tracker_production', user='remote_user', password='2050018c0b61c0c971e56084eefb0c0d', host='46.101.58.30')
#conn = psycopg2.connect(dbname='academic_tracker_production', user='remote_user', password='0e766b1810c5cd042000c00fb9ee85c1', host='46.101.58.30')
conn = psycopg2.connect(dbname='d9k69uia8l4iu', user='ml_limited_user', password='p418b74d6bf350b2405574e62835d37bbcce86c2ad8c34d5a5b80f01975625d58', host='ec2-18-203-229-185.eu-west-1.compute.amazonaws.com')
       
#postgres_str='postgres://remote_user:0e766b1810c5cd042000c00fb9ee85c1@46.101.58.30:5432/academic_tracker_production'

cursor = conn.cursor()

#for uid in df0.idu:
for uid in id_df:
   
   uexpectation = df_all.loc[df_all['idu']==uid]
   uexpectation_=list(uexpectation.expectation)
   uexpectation_regr=list(uexpectation.expectation_regr)
   uexpectation_exp=list(uexpectation.expectation_exp)
   uexpectation_kr=list(uexpectation.expectation_kr)
   #print (uexpectation)
   #print (uexpectation_exp[0])
   #print (uexpectation_kr[0])
   #print (uexpectation_regr[0])
   if len(uexpectation)!=0:

     try:
        #print('root')
        with conn.cursor() as cursor:
                   conn.autocommit = True
                   cur_= uid
                   #print(cur_)
                   cursor.execute("select expectation from results WHERE id=%s",[cur_])
                   records = cursor.fetchall()
                   #print(records[0][0])
                   if (records[0][0] is None) or (str(records[0][0])=='nan'):
                           #print(cursor.fetchone())
                     #if uexpectation.expectation.values[0] is None: 
                           if (str(uexpectation_kr[0])=='nan') or (uexpectation_kr[0] is None) or (uexpectation_kr[0]<=0):
                               print('branch 1')
                               if uexpectation_exp[0]!= None:
                                   #print (uexpectation_exp)
                                   with conn.cursor() as cursor:
                                        conn.autocommit = True    
                                        cursor.execute("UPDATE results SET expectation=%s  WHERE id=%s", (uexpectation_exp[0], uid))
                           else:
                               print('branch 2')
                               if (uexpectation_regr[0] is None) or (str(uexpectation_regr[0])=='nan') or (uexpectation_regr[0]<=0):
                                    with conn.cursor() as cursor:
                                         conn.autocommit = True    
                                         cursor.execute("UPDATE results SET expectation=%s  WHERE id=%s", (uexpectation_kr[0], uid))

                               else:
                                   print('branch 3')
                                   if abs(uexpectation_regr[0] - uexpectation_exp[0])<abs(uexpectation_regr[0] - uexpectation_kr[0]) and abs(uexpectation_regr[0] - uexpectation_exp[0])<abs(uexpectation_kr[0] - uexpectation_exp[0]):
                                       #print('!')
                                       with conn.cursor() as cursor:
                                           conn.autocommit = True    
                                           cursor.execute("UPDATE results SET expectation=%s  WHERE  id=%s", (uexpectation_regr[0], uid))
                                   else:
                                      print('branch 4') 
                                      with conn.cursor() as cursor:
                                           conn.autocommit = True    
                                           cursor.execute("UPDATE results SET expectation=%s  WHERE id=%s", (uexpectation_kr[0], uid))
                   else:
                        print ('not empty')
     except:
         print('error '+str(uid))

# Clear data expectation
def updateTable():
    try:
        #conn = psycopg2.connect(dbname='academic_tracker_production', user='remote_user', password='0e766b1810c5cd042000c00fb9ee85c1', host='46.101.58.30')
        conn = psycopg2.connect(dbname='d9k69uia8l4iu', user='ml_limited_user', password='p418b74d6bf350b2405574e62835d37bbcce86c2ad8c34d5a5b80f01975625d58', host='ec2-18-203-229-185.eu-west-1.compute.amazonaws.com')
        
        #postgres_str='postgres://ml_limited_user:p418b74d6bf350b2405574e62835d37bbcce86c2ad8c34d5a5b80f01975625d58@ec2-18-203-229-185.eu-west-1.compute.amazonaws.com:5432/d9k69uia8l4iu'

        cursor = conn.cursor()
        
        print("Table Before updating record ")
        sql_select_query = """select * from results"""
        cursor.execute(sql_select_query)
        record = cursor.fetchone()
        print(record)
        # Update single record now
        sql_update_query = """UPDATE results SET expectation = NULL"""
        cursor.execute(sql_update_query)
        conn.commit()
        count = cursor.rowcount
        print(count, "Record Updated successfully ")
        print("Table After updating record ")
        sql_select_query = """select * from results"""
        cursor.execute(sql_select_query)
        record = cursor.fetchone()
        print(record)
    except (Exception, psycopg2.Error) as error:
        print("Error in update operation", error)
    finally:
        # closing database connection.
        if (conn):
            cursor.close()
            conn.close()
            print("PostgreSQL connection is closed")
             

try:
    cursor.close()
    conn.close()
except:   
    print('error connection')
    
    
end = time.time()
print(end - start)
           
#

