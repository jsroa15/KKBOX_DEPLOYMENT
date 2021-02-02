#Import Modules and libraires

import pandas as pd
import streamlit as st
import numpy as np
import pickle


st.write('''
         
         Author: **Juan Sebastian Roa**
         
         GitHub Profile: https://github.com/jsroa15
         
         ''')


st.title('Churn Prediction App')

st.write('''
         
         Data and problem from Kaggle competition: https://www.kaggle.com/c/kkbox-churn-prediction-challenge/overview
         
         ''')

    

st.write("Please fill out the User Input Parameters on the left")

st.write(''' 
         ## **Model Used: Random Forest Classifier**

             
         
| ROC AUC Score | Precision | Recall | F1 Score |
|---------------|-----------|--------|----------|
| 94.05         | 64.06%    | 38.34% | 48.01    |
         
         ''')

#####Capturing Features from App user#####

st.sidebar.header('User Input Parameters')

def user_input_features():
    
    #Scaled feature: I have to use mean and std fit in the StandarScaler
    
    regist_trans=st.sidebar.number_input('Number of Transactions')
    regist_trans=(regist_trans-16.27)/8.36
    
    #Scaled feature: I have to use mean and std fit in the StandarScaler
    
    mst_frq_plan_days=st.sidebar.number_input('Plan Days')
    mst_frq_plan_days=(mst_frq_plan_days-33.26)/31.24
    
    
    mst_frq_pay_met=st.sidebar.selectbox('Payment Method ID',('29', '30', '31', '32', '33', '34', '36', '37', '38', '39', '40','41', 'Other'))
    
    if mst_frq_pay_met=='29':
        
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
        
    elif mst_frq_pay_met=='30':
        mst_frq_pay_met_30=1 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
            
    elif mst_frq_pay_met=='31':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=1
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
        
    elif mst_frq_pay_met=='32':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=1
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
        
    elif mst_frq_pay_met=='33':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=1
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
        
    elif mst_frq_pay_met=='34':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=1
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
    
    elif mst_frq_pay_met=='36':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=1
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
        
    elif mst_frq_pay_met=='37':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=1
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
        
    elif mst_frq_pay_met=='38':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=1
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
        
        
    elif mst_frq_pay_met=='39':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=1
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
        
    elif mst_frq_pay_met=='40':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=1
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=0
        
    elif mst_frq_pay_met=='41':
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=1
        mst_frq_pay_met_other=0
        
    else:
        mst_frq_pay_met_30=0 
        mst_frq_pay_met_31=0
        mst_frq_pay_met_32=0
        mst_frq_pay_met_33=0
        mst_frq_pay_met_34=0
        mst_frq_pay_met_36=0
        mst_frq_pay_met_37=0
        mst_frq_pay_met_38=0
        mst_frq_pay_met_39=0
        mst_frq_pay_met_40=0
        mst_frq_pay_met_41=0
        mst_frq_pay_met_other=1
        
    
    #Scaled feature: I have to use mean and std fit in the StandarScaler
    
    revenue=st.sidebar.number_input('Revenue')
    revenue=(revenue-2252.60)/1239.51
    
    is_auto_renew_1=st.sidebar.selectbox('Has Auto Renew',('Yes','No'))
    if is_auto_renew_1 =='Yes':
        is_auto_renew_1=1
    else:
        is_auto_renew_1=0
        
    #Scaled feature: I have to use mean and std fit in the StandarScaler
    
    regist_cancels=st.sidebar.number_input('Number of Cancelations')
    regist_cancels=(regist_cancels-0.27)/0.52
    
    qtr_trans=st.sidebar.selectbox('Transaction made in quarter',(1,2,3,4))
    
    if  qtr_trans=='1':
        qtr_trans_2=0
        qtr_trans_3=0
        qtr_trans_4=0
    
    elif qtr_trans=='2':
        qtr_trans_2=1
        qtr_trans_3=0
        qtr_trans_4=0
        
    elif qtr_trans=='3':
        qtr_trans_2=0
        qtr_trans_3=1
        qtr_trans_4=0
        
    else:
        qtr_trans_2=0
        qtr_trans_3=0
        qtr_trans_4=1  
        
    
    city=st.sidebar.selectbox('City',(1,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15, 17, 18, 21, 22,'Other'))
    if  city==1:
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
     
    elif city==3:
        
        city_3=1
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==4:
        
        city_3=0
        city_4=1
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==5:
        
        city_3=0
        city_4=0
        city_5=1
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==6:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=1
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==8:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=1
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==9:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=1
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==10:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=1
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==11:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=1
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==12:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=1
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==13:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =1
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==14:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=1
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==15:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =1
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==17:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=1
        city_18=0
        city_21=0
        city_22=0
        city_other=0
        
    elif city==18:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=1
        city_21=0
        city_22=0
        city_other=0
        
    elif city==21:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=1
        city_22=0
        city_other=0
        
    elif city==22:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=1
        city_other=0
        
    else:
        
        city_3=0
        city_4=0
        city_5=0
        city_6=0
        city_8=0
        city_9=0
        city_10=0
        city_11=0
        city_12=0
        city_13 =0
        city_14=0
        city_15 =0
        city_17=0
        city_18=0
        city_21=0
        city_22=0
        city_other=1
        
    gender_male=st.sidebar.selectbox('Gender',('Male','Female','Other'))
    if gender_male=='Female':
        gender_male=0
        gender_other=0
    elif gender_male=='Male':
        gender_male=1
        gender_other=0
    else:
        gender_male=0
        gender_other=1
        
    registered=st.sidebar.selectbox('Registered Via',(3,4,7,9,'Other'))
    if registered==3:
        registered_via_4=0
        registered_via_7=0
        registered_via_9=0
        registered_via_other=0
        
    elif registered==4:
        registered_via_4=1
        registered_via_7=0
        registered_via_9=0
        registered_via_other=0
        
    elif registered==7:
        registered_via_4=0
        registered_via_7=1
        registered_via_9=0
        registered_via_other=0
        
    elif registered==9:
        registered_via_4=0
        registered_via_7=0
        registered_via_9=1
        registered_via_other=0
        
    else:
        registered_via_4=0
        registered_via_7=0
        registered_via_9=0
        registered_via_other=1
            
    #Scaled feature: I have to use mean and std fit in the StandarScaler
    
    age=st.sidebar.number_input('Age')
    bd=(age-13.27)/15.98
    
    num_25=st.sidebar.number_input('# of songs played less than 25% of the song length')
    num_25=np.log1p(num_25)
    
    num_50=st.sidebar.number_input(' # of songs played between 25% to 50% of the song lengt')
    num_50=np.log1p(num_50)
    
    num_75=st.sidebar.number_input('# of songs played between 50% to 75% of of the song lengt')
    num_75=np.log1p(num_75)
    
    num_985=st.sidebar.number_input('# of songs played between 75% to 98.5% of the song length')
    num_985=np.log1p(num_985)
    
    num_100=st.sidebar.number_input('# of songs played over 98.5% of the song length')
    num_100=np.log1p(num_100)
    
    num_unq=st.sidebar.number_input('# of unique songs played')
    num_unq=np.log1p(num_unq)
    
    total_secs=st.sidebar.number_input('Total seconds listened')
    total_secs=np.log1p(total_secs)
    
    #Scaled feature: I have to use mean and std fit in the StandarScaler
    
    tenure=st.sidebar.number_input('Tenure in years')
    tenure=(tenure-3.52)/2.88
    
    data={
        'regist_trans':regist_trans, 
        'mst_frq_plan_days':mst_frq_plan_days,
        'revenue':revenue,
        'regist_cancels':regist_cancels,
        'bd':bd,
        'num_25':num_25,
        'num_50':num_50,
        'num_75':num_75,
        'num_985':num_985,
        'num_100':num_100,
        'num_unq':num_unq,
        'total_secs':total_secs,
        'tenure':tenure,
        'mst_frq_pay_met_30':mst_frq_pay_met_30,
        'mst_frq_pay_met_31':mst_frq_pay_met_31,
        'mst_frq_pay_met_32':mst_frq_pay_met_32,
        'mst_frq_pay_met_33':mst_frq_pay_met_33,
        'mst_frq_pay_met_34':mst_frq_pay_met_34,
        'mst_frq_pay_met_36':mst_frq_pay_met_36,
        'mst_frq_pay_met_37':mst_frq_pay_met_37,
        'mst_frq_pay_met_38':mst_frq_pay_met_38,
        'mst_frq_pay_met_39':mst_frq_pay_met_39,
        'mst_frq_pay_met_40':mst_frq_pay_met_40,
        'mst_frq_pay_met_41':mst_frq_pay_met_41,
        'mst_frq_pay_met_other':mst_frq_pay_met_other,
        'is_auto_renew_1':is_auto_renew_1,
        'qtr_trans_2':qtr_trans_2,
        'qtr_trans_3':qtr_trans_3,
        'qtr_trans_4':qtr_trans_4,
        'city_3.0':city_3,
        'city_4.0':city_4,
        'city_5.0':city_5,
        'city_6.0':city_6,
        'city_8.0':city_8,
        'city_9.0':city_9,
        'city_10.0':city_10,
        'city_11.0':city_11,
        'city_12.0':city_12,
        'city_13.0':city_13,
        'city_14.0':city_14,
        'city_15.0':city_15,
        'city_17.0':city_17,
        'city_18.0':city_18,
        'city_21.0':city_21,
        'city_22.0':city_22,
        'city_other':city_other,
        'gender_male':gender_male,
        'gender_other':gender_other,
        'registered_via_4.0':registered_via_4,
        'registered_via_7.0':registered_via_7,
        'registered_via_9.0':registered_via_9,
        'registered_via_other':registered_via_other
    
        }
    
    features=pd.DataFrame(data,index=[0])
    return features


#Here I call the function to get a dataset with features entered by the app user

f=user_input_features()


#Loading the model into the script. I used pickle to save the model

pickle_in=open('final_model.pickle','rb')
model = pickle.load(pickle_in)

#Here I use sklearn API to use the model and make predictions

prediction=model.predict(f)

#Translating output


st.header('Final Prediction:')


if prediction==0:
    st.header('**User has Retention**')
else:
    st.subheader('**User has Churn!!**')