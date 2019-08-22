# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 21:39:53 2019

@author: 
"""
import time
start = time.time()
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
import pickle

#load data from DB to pandas
from sqlalchemy import create_engine

#Config of variables for control of models  

training=True #flag of training
start_exam=True #flag of entry exams
#

def part(start,finish):
    finish_=finish
    df1=pd.read_sql_query('SELECT id, subject, student_year_id, score, title, examined_at, expectation FROM results WHERE id > %s and id< %s', cnx, params=(start,finish))
    for i in range(1,350):
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
    
postgres_str='postgres://remote_user:0e766b1810c5cd042000c00fb9ee85c1@46.101.58.30:5432/academic_tracker_production'

cnx = create_engine(postgres_str)

df_schools = pd.read_sql_query('''SELECT id, name FROM schools''', cnx)
df_schools_=df_schools[df_schools['name'].str.contains("Test*")==False]
df_schools_=df_schools_[df_schools_['name'].str.contains("TEST*")==False]

df2=pd.read_sql_query('''SELECT id, student_id, academical_year, calendar_year FROM student_years''', cnx)

df1=pd.read_sql_query('''SELECT id, subject, student_year_id, score, title, examined_at,
       expectation FROM results''', cnx)
df1=df1.rename(columns = {'id': 'idu'})
#df=pd.read_sql_query('''SELECT results.id, results.subject, results.score, results.expectation, student_years.student_id,  student_years.academical_year, student_years.calendar_year FROM results INNER JOIN student_years ON results.student_year_id = student_years.id ORDER BY results.id''', cnx)
df3=pd.read_sql_query('''SELECT id, school_id, gender, year_of_entry, name FROM students''', cnx)

df8=df3.merge(df_schools_,  how='inner', left_on='school_id', right_on='id')

df4=pd.read_sql_query('''SELECT * FROM intro_results;''', cnx)

#normalization of exam marks 
df4['score_start'] = df4.apply(lambda row: calc_start_exam(row['score'],  row['exam_category']), axis=1)

#mean of exam marks 
df4=df4.groupby(['student_id'])['score_start', 'student_id'].mean()
df4=df4.rename(columns = {'student_id': 'student_id_group'})
#

df=df2.merge(df1,  how='inner', left_on='id', right_on='student_year_id')
df=df.merge(df3,  how='inner', left_on='student_id', right_on='id')

#df0 - data without  entry exams
df0=df

#df - data with entry exams
df=df.merge(df4,  how='inner', left_on='student_id', right_on='student_id')
df2,df3,df4= 0,0,0

#for filter of schools
#if school_inp >=0 : df=df[df.school_id == school_inp]



# filter of exam date < time of current exam
def last(examined_at_t):
    time_= examined_at_t
    date_=[x for x in spisok_t if x < time_]
    if len(date_)>0:
          result = date_[-1]
    else: 
          result=0
    return result 


#Model of regression
def model_(data_model):
    #from sklearn.metrics.pairwise import pairwise_kernels
  
    Y = data_model.iloc[:,[2]]
    if start_exam==True:
        X_=data_model.columns[5]
        #print(X_)
        X = data_model.iloc[:,[5,6]]
        filename = [data_model.columns[2]+'_'+ data_model.columns[5]+'X0'+'.sav']
    else:
        X = data_model.iloc[:,[4]]
        X_=data_model.columns[4]
        filename = [data_model.columns[2]+'_'+ data_model.columns[4]+'.sav']
    
    # Regression
    
    model = LinearRegression().fit(X,Y)
    r_sq = model.score(X,Y)
    
    # Ridge Regression
    # GridSearchCV of best parameters
    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1, 0.1, 0.01, 0.001, 0.0001],
                              "gamma": np.logspace(-2, 2, 5)})
    kr.fit(X, Y)
    
    #
    #Estimate of Ridge Regression model
    y_kr = kr.predict(X)
    err_kr = np.sqrt(mean_squared_error(Y,y_kr))
        
    #save of model
    pickle.dump(kr, open(filename[0], 'wb'))
    #save model results to table
    model_results.append([data_model.columns[2], X_, model.intercept_, model.coef_, r_sq, err_kr])  

#Model of Log regression
def model_log(data_model):
        
    Y = data_model.iloc[:,[2]]
    X = data_model.iloc[:,[4]]
    
    model1 = LogisticRegression()
    model1.fit(X, Y)
    print(model1.score(X, Y))
    print(model1.get_params())
    
#Model of Lasso regression
def model_Las (data_model):
    
    Y = data_model.iloc[:,[2]]
    #print(spisok[j], spisok[i])
    if start_exam==True:
        X_=data_model.columns[5]
        #print(X_)
        X = data_model.iloc[:,[5,6]]
        X_=data_model.columns[5]
        filename = [data_model.columns[2]+'-'+ data_model.columns[5]+'X0'+'.sav']
    else:
        X = data_model.iloc[:,[4]]
        X_=data_model.columns[4]
        filename = [data_model.columns[2]+'-'+ data_model.columns[4]+'.sav']
        
    seed = 3
    
    model = ElasticNet()
    
    result=model.fit(X, Y)
   
    err_las = np.sqrt(mean_squared_error(Y,result.predict(X)))
    
    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1, 0.1, 0.01, 0.001, 0.0001],
                              "gamma": np.logspace(-2, 2, 5)})
    parameters = {'kernel':['rbf', 'sigmoid'], 'C':np.logspace(np.log10(0.001), np.log10(200), num=20), 'gamma':np.logspace(np.log10(0.00001), np.log10(2), num=30)}
    
    kr.fit(X, Y)
    
    err_kr = np.sqrt(mean_squared_error(Y, kr.predict(X)))
          
        
    model_lin = LinearRegression().fit(X,Y)
    
    err_lin = np.sqrt(mean_squared_error(Y, model_lin.predict(X))) 
    
    pickle.dump(result, open(filename[0], 'wb'))
    model_results.append([data_model.columns[2],X_, err_kr, err_las, err_lin])
    
    
# Use regression model for forecasting
    
def dX_(student_data_, Y_name, Y_date):
    #
    try:  
          # Do we have previous records for forecasting?
          X_=student_data_[student_data_.examined_at< Y_date]
          if len(X_)<1:
                X_=student_data_.loc[student_data_.examined_at== Y_date]
               # use data of entry exams
                X_s=X_[0:1]
               # 
                try:
                     forecast_score=X_s.score_start.values[0]
                #     print('!!!!forecast_score=X_s.score_start.values[0]')
                except:
                 #    print('!!!!forecast_score=None')
                     forecast_score=None
                return forecast_score
          
          Subject_X =[x for x in X_.subject]
          r=0
          # If we have data then we can get best formula of regression
          for x in Subject_X:
              formula_=df_m.loc[(df_m.Y==Y_name) & (df_m.X == x)]
              if formula_.R.values>r:
                 r=formula_.R.values
                 formula=formula_
          
          X_= X_[X_.subject==formula.X.values[0]]
          #
          if start_exam==True and flag==True:
              #
              try:
                  #print('1')
                 
                  forecast_score= float(formula.A0.values[0][0])+float(formula.A1.values[0][0][0])*X_.score.values[-1] + float(formula.A1.values[0][0][1])*X_.score_start.values[0]
                  print(forecast_score)
              except:
                  forecast_score=X_.score_start.values[0]
          else:
              #print('2')
              forecast_score= float(formula.A0.values[0][0])+float(formula.A1.values[0][0][0])*X_.score.values[-1]
              print(forecast_score)
    except:
          forecast_score= None
    return forecast_score

def dX_kr(student_data_, Y_name, Y_date):
    
    #Ridge Regression
    
    #
    try: 
          X_=student_data_.loc[student_data_.examined_at< Y_date]
          #
          if len(X_)<1:
                X_=student_data_.loc[student_data_.examined_at== Y_date]
               # 
              
                X_s=X_[0:1]
               # 
                try:
                     forecast_score=X_s.score_start.values[0]
                #     
                except:
                 #   
                     forecast_score=None
                return forecast_score
            
          Subject_X =[x for x in X_.subject]
          r=100
          #
          for x in Subject_X:
              
              formula_=df_m.loc[(df_m.Y==Y_name) & (df_m.X == x)]
                            
              if len(formula_)>0: 
                 if formula_.Err_kr.values<r:
                     r=formula_.Err_kr.values
                     formula=formula_
                 
              
                  
          print(formula)       
          X_= X_[X_.subject==formula.X.values[0]]
          print(X_)
          
          if start_exam==True and flag==True:
             print('Ridge Regression 1') 
             namefile=Y_name +'_'+ X_.subject.values+'X0'+'.sav' 
             #
             b=[[X_.score.values[-1], X_.score_start.values[0]]]
             #
             loaded_model = pickle.load(open(str(namefile[0]), 'rb'))
             forecast_score=loaded_model.predict(b)
             forecast_score=forecast_score[0][0]
             print(forecast_score)
             #print(forecast_score[0][0])
          else:
             print('Ridge Regression 2')
             namefile=Y_name +'_'+ X_.subject.values+'.sav'
             #
             b=X_.score.values[-1]
             b=np.array([b])
             loaded_model = pickle.load(open(str(namefile[0]), 'rb'))
             forecast_score=loaded_model.predict(b.reshape(1, -1))
             #
             forecast_score=forecast_score[0][0]
             print(forecast_score)
          
                                     
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


#********** MAIN 
    
spisok=[x for x in df.subject.unique()]

spisok_t=[x for x in df.examined_at.unique()]

s=[]
df=df.sort_values(['subject', 'examined_at'])
for i in spisok:
    #
    s.append(df[(df.subject==i)])


#

def create_models(df, start_exam):
        
    if training == True:
              
   # 
        data_all=[x for x in spisok]

   # 
        for j in range(len(spisok)):
            if start_exam==True:
               m=s[j][['student_id','examined_at', 'score', 'score_start']]
            else:
               print('+') 
               m=s[j][['student_id','examined_at', 'score']]
   #rename column 'score' on name of subject   
            first=m.rename(columns = {'score': spisok[j]}).sort_values('examined_at')
            #
   
            first=first.rename(columns = {'examined_at': 'examined_at_t'})
      #
            first['examined_at'] = first.apply(lambda row: last(row['examined_at_t']), axis=1)
      # 
            data_all[j]=[]
   
   #merge DF for first data subject and second data subject
            for i in range(len(spisok)):
                if start_exam==True:
                    m=s[i][['student_id','examined_at', 'score', 'score_start']]
                else:
                   # print('++') 
                    m=s[i][['student_id','examined_at', 'score']]
             
         #rename column 'score' on name of subject 
                second=m.rename(columns = {'score': spisok[i]}).sort_values('examined_at')
         #print(j, spisok[j], spisok[i])
                data_all[j].append(pd.merge(first, second, on=['examined_at','student_id']))
             #
                if len(data_all[j][i])>6 and (abs(data_all[j][i].corr().iloc[2].filter(regex=data_all[j][i].columns[2])[0])>0.1) and (j!=i):
           
                     model_(data_all[j][i])
                          
                else: 
                     print('NOT for '+str(j)+' '+ str(i))

              
#Create df for all model results              
        df_m1=pd.DataFrame(model_results, columns=['Y', 'X', 'A0', 'A1', 'R', 'Err_kr'])              

    else:
        print('No training') 
    return df_m1
# forecasting for subject on base score X

if start_exam==True:
     model_results=[]
     start_exam=False
     df_m=create_models(df0, start_exam)
     start_exam=True
     model_results=[]
     df_m=create_models(df, start_exam)
else:    
     df_m=create_models(df0, start_exam)

uniq=df0.student_id.unique() 
student_id_=[x for x in uniq]

   
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
    


#

# Update data of expectation in results
import psycopg2
from psycopg2 import sql
conn = psycopg2.connect(dbname='academic_tracker_production', user='remote_user', password='2050018c0b61c0c971e56084eefb0c0d', host='46.101.58.30')
cursor = conn.cursor()

for uid in df1.idu:
#    
   
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
                   print(records[0][0])
                   if (records[0][0] is None) or (str(records[0][0])=='nan'):
                           #print(cursor.fetchone())
                     #if uexpectation.expectation.values[0] is None: 
                           if (str(uexpectation_kr[0])=='nan') or (uexpectation_kr[0] is None) or (uexpectation_kr[0]<1):
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


def updateTable():
    try:
        conn = psycopg2.connect(dbname='academic_tracker_production', user='remote_user', password='2050018c0b61c0c971e56084eefb0c0d', host='46.101.58.30')
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
           