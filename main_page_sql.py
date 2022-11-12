#Env setup
import numpy as np
import pandas as pd
import streamlit as st
import evalml
import shap
import ckwrap
import pickle
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import openpyxl
from evalml.model_understanding import readable_explanation
from evalml.model_understanding import graph_permutation_importance
from evalml.model_understanding import graph_partial_dependence
from evalml.model_understanding.metrics import graph_confusion_matrix
from evalml.model_understanding.force_plots import graph_force_plot
from evalml.data_checks import DefaultDataChecks
from evalml.automl import AutoMLSearch
from evalml.model_understanding.prediction_explanations import explain_predictions
from imblearn.over_sampling import SMOTENC
from imblearn import over_sampling
from io import BytesIO
from kneed import KneeLocator
from scipy.stats import norm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
#sql localhost connection
import sqlalchemy
import pymysql
engine=sqlalchemy.create_engine('mysql+pymysql://root:miupiu19@localhost:3306/workload')

import mysql.connector

# Initialize connection.
# Uses st.experimental_singleton to only run once.
#@st.experimental_singleton
#def init_connection():
#    return mysql.connector.connect(**st.secrets["mysql"])

#conn = init_connection()

import mysql.connector
cnx = mysql.connector.connect(user='root', password='miupiu19',
                              host='localhost',port="3306"
                              database='workload')
#cnx.close()


warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

def preprocessing_encoding(df):
    # mapping dictionary untuk pendidikan
    education = { 
        'Below College': 1,
        'College (D4)': 2,
        'Bachelor (S1)': 3,
        'Master': 4,
        'Doctor':5
    }
    df.loc[:, "EDUCATION_GRADE"] = df.EDUCATION_GRADE.map(education)
    # mapping dictionary untuk level pekerjaan
    grade_pekerjaan = { 
    'VI': 1,
    'V': 2,
    'IV': 3,
    'III': 4,
    'II':5
    }
    df.loc[:, "JOB_LEVEL"] = df.JOB_LEVEL.map(grade_pekerjaan)
    return df

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Data Model')
    workbook = writer.book
    worksheet = writer.sheets['Data Model']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

   

#streamlit page
st.set_page_config(
    page_title="WLA - WFP predictor",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

def main_page():
    st.markdown("# Main page")
    st.write("""
    Pada aplikasi ini, terdapat 4 fungsi utama yaitu:
    
    1. Modeler
    
        a. Pemodelan WLA (FTE) dengan menggunakan pemodelan classification.
        Di halaman ini Anda dapat mentraining model dengan metode classification. Dan dengan menggunakan data WLA yang Anda sajikan. Hasil akhir yng bisa didapat dari halaman ini adalah excel file berisi table hasil populate data dengan menggunakan trained model dan model itu sendiri.

        b. Pemodelan WLA (FTE) dengan menggunakan pemodelan regression.
        Di halaman ini Anda dapat mentraining model dengan metode regression. Dan dengan menggunakan data WLA yang Anda sajikan. Hasil akhir yng bisa didapat dari halaman ini adalah excel file berisi table hasil populate data dengan menggunakan trained model dan model itu sendiri.
        
        c. Pemodelan WFP menggunakan pemodelan regression. 
        Di halaman ini Anda dapat mentraining model dengan metode regression. Dan dengan menggunakan data WFP yang Anda sajikan. Hasil akhir yng bisa didapat dari halaman ini adalah excel file berisi table hasil populate data dengan menggunakan trained model dan model itu sendiri.
        
    2. Check GAP FTE_demand dan FTE_supply. 
        Di halaman ini Anda dapat mengetahui berapa nilai FYE_demand dari data WFP yang Anda sajikan dan berpa jumlah FTE_supply dari data yang telah dilengkapi oleh machine learning.
        
    3. Predictor
    
        a. WLA Classifier Predictor. 
        Di halaman ini Anda dapat me-load hasil Pemodelan WLA (FTE) dengan menggunakan pemodelan classification, untuk mentraining data baru. Tanpa harus melakukan training ulang. 
        
        b. WLA Regression Predictor.
        Di halaman ini Anda dapat me-load hasil Pemodelan WLA (FTE) dengan menggunakan pemodelan regression, untuk mentraining data baru. Tanpa harus melakukan training ulang. 
        
        c. WFP Regression Predictor.
        Di halaman ini Anda dapat me-load hasil Pemodelan WFP menggunakan pemodelan regression, untuk mentraining data baru. Tanpa harus melakukan training ulang. 
    
    4. Explainer
       Di halaman ini, dimuat beberapa penjelasan bagi masing masing model yang sudah terbuat. Mulai dari feature importance, permutation importance, shap value, dan model itu sendiri.
    
    *PS. Untuk halaman 2 "Modeler". Jika Anda telah melakukan training. Untuk menjalankan fungsi Populate, Anda tidak perlu melakukan Training ulang
             """)

def page2():  
    tab1,tab2,tab3=st.tabs(["WLA CLassifier Modeler","WLA Regression Modeler","WFP Regression Modeler"])
    with tab1:
        st.write("# WLA (FTE) - ML Classification Modeler")
        with st.expander("Introduction"):
            st.header("Pemodelan Work Load Analysis (WLA) menggunakan metode Full Time Equivalent (FTE) dengan pemodelan Machine Learning Multi Class Classification")
            st.markdown(
            """
                Beban kerja (Workload) adalah kumpulan atau sejumlah aktivitas yang harus dilakukan oleh suatu unit dalam organisasi atau pemegang jabatan dalam jangka waktu yang telah ditentukan, biasanya disetahunkan. Ada banyak metode dalam menghitung Workload, yang masing masing memiliki kelebihan dan kekurangan. Dari sekian banyak metode yang dapat digunakan untuk menganalisis Workload, terdapat metode yang disebut dengan metode Full-Time Equivalent (FTE). Hasil perhitungan analisis beban kerja melalui metode FTE akan menghasilkan output apakah sebagian pekerja atau karyawan mengalami beban kerja overload, stretch, fit, atau underload. 

                Di telkom riwayat aktivitas tercatat pada diarium. Namun pengisiannya masih belum lengkap. dan meningkatkan keakurasian pengisian aktivitas di diarium dirasa akan memberikan tambahan pekerjaan pada karyawan telkom. Sedangkan metode kuesioner juga akan sangat menyita waktu. 

                Oleh karena itu kita mencoba menghitung nilai FTE berdasarkan isian diarium. kemudian membuat sebuah pemodelan yang mampu mengkelompokkan karyawan dan memperkirakan nilai fte nya. Model ini nantinya akan digunakan untuk memprediksi karyawan yang fte nya kosong atau anomali.

                Metode yang digunakan adalah metode classification, jadi hasil akhir pemodelan adalah dengan menegelompokkan karyawan berdasarkan feature tertentu untuk masuk ke kategori fte yang sejenis. Dengan asumsi apabila karyawan dengan beberapa fitur terntu masuk ke kategori klas fte. maka karywan dengan fitur yang mirip harusnya masuk dalam klas fte yang sama. Dikarenakan hasil akhir dari pemodelan ini adalah kelas fte. maka nilai fte yang sebenarnya tidak dapat diketahui. Untuk dapat mengetahui nilai fte sebenarnya, pemodelan dengan regresi adalah disarankan. Namun berdasarkan hasil running model dengan data yang tersedia, pemodelan dengan metode klasifikasi memiliki akurasi yang lebih besar daripada metode regresi.

                Dan kesemuanya dilakukan dalam tiga langkah, yaitu: Load Data, Pembuatan Model, dan Menggunakan Model yang tealh dibuat untuk menentukan class FTE nya.
            """
            )
        st.markdown("Langkah")    
        with st.expander("1. Upload file Excel Data WLA.xlsx",expanded=True):
            uploaded_file=pd.read_sql_table("data_model", engine)
            col1,col2=st.columns(2)
            with col1:
                feature_select=st.multiselect("pilih features", uploaded_file.columns, 
                                              ['EMPLOYEE_ID', 'AGE', 'GENDER', 'MARITAL_STATUS', 'EDUCATION_GRADE',
                                               'RECRUITMENT_STATUS', 'EMPLOYMENT_STATUS', 'YEARS_AT_COMPANY','DEPARTMENT',
                                               'JOB_LEVEL', 'JOB_NATURE', 'TASK_TYPE',
                                               'THP_CURRENT', 'thp_increase', 'LAST_MUTATION',
                                               'LAST_PROMOTION','training_duration_2022', 'leadership_score',
                                               'functional_score', 'budaya_score']) 
            with col2:
                target_select=st.multiselect("pilih target", uploaded_file.columns, 'fte')
            if uploaded_file is not None:
                useful_feature=list(feature_select+target_select)
                st.success("Data loaded")
        with st.expander("2. Build Model",expanded=False):
            if uploaded_file is not None:
                # read data
                df = uploaded_file[useful_feature]
                #remove null on target
                df = df.dropna(subset=target_select)
                #ordinal encoding
                df = preprocessing_encoding(df)
                #make corr graph
                corr = df.corr()
                st.write( corr.style.background_gradient(cmap='coolwarm').set_precision(2))
                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                
                #convert ft to class
                df['fte_label'] = np.where(df[str(target_select[0])]>=1.3, 'Overload', np.where(df[str(target_select[0])]>=1.15, 'Stretch',np.where(df[str(target_select[0])]>=0.85, 'Fit', 'Underload')))

                
                st.dataframe(df)
                result = df.groupby('fte_label').agg({str(target_select[0]): ['mean', 'min', 'max']})
                st.dataframe(result)
                #drop unclassed target
                df.drop(target_select,axis=1, inplace=True)
                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                # remove column employee_id
                df=df.loc[:,df.columns != 'EMPLOYEE_ID']
                #define x
                x=df.drop(['fte_label'],axis=1,inplace=False)
                #define y
                y=df['fte_label']
                #SMOTENC
                cat_cols = []
                for col in df.loc[:, df.columns != 'fte_label']:
                    if df[col].dtype == 'object': #or 'category' if that's the case
                        cat_cols.append(True)
                    else:
                        cat_cols.append(False)
                sm = SMOTENC(categorical_features=cat_cols,random_state=72,k_neighbors=1)
                X_res, y_res = sm.fit_resample(x, y)
                #define modeling problem
                problem=evalml.problem_types.detect_problem_type(y_res.squeeze(axis=0))
                st.write(problem)
                #select objectives based on problem
                if str(problem)=="multiclass":
                    objectif = ("MCC Multiclass","Log Loss Multiclass","AUC Weighted","AUC Macro","AUC Micro","Precision Weighted","Precision Macro","Precision Micro","F1 Weighted","F1 Macro","F1 Micro","Balanced Accuracy Multiclass","Accuracy Multiclass")
                elif str(problem)=="binary":
                    objectif = ("MCC Binary","Log Loss Binary","Gini","AUC","Precision","F1","Balanced Accuracy Binary","Accuracy Binary")
                elif str(problem)=="regression":
                    objectif = ("ExpVariance","MaxError","MedianAE","MSE","MAE","R2","Root Mean Squared Error")
                option = st.selectbox('Choose Objective',objectif,key=1)
            clicked = st.button("Build Model")
            if uploaded_file is not None and clicked is True:
                #convert df to array
                y_res=y_res.squeeze()
                #Modeling
                ##splitting
                X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X_res,y_res, problem_type=problem)
                #check data problem
                st.write("CHECKING DATA PROBLEM :")
                data_checks = DefaultDataChecks(problem, option)
                if data_checks.validate(X_train, y_train)==[]:
                    st.info("no problem with data")
                else:
                    st.info(data_checks.validate(X_train, y_train))
                ##find the best pipeline
                st.write("FINDING THE BEST MODEL")
                with st.spinner(text="In progress..."):
                    automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type=problem,objective=option,random_seed=72,ensembling=True,allow_long_running_models=True,verbose=True)
                    automl.search()
                    st.success("Best Model Found:")
                    st.success(automl.best_pipeline.name)
                ##Evaluate on hold out data
                st.markdown("EVALUATE DATA")
                st.json(automl.best_pipeline.score(X_test, y_test, objectives=[option]))
                st.write("WRITING MODEL")
                pipeline_to_pickle = automl.best_pipeline
                with open("pipeline-classification.pkl","wb") as f:
                    pickle.dump(pipeline_to_pickle,f)
                st.success("MODEL BUILT! 🥳")
                pickled_pipeline=None
                with open("pipeline-classification.pkl","rb") as f:
                    #pickled_pipeline=pickle.load(f)
                    st.download_button(label='📥 Download ML Model',data=f ,file_name='pipeline-classification.pkl', key=2)
                    
        with st.expander("3. Model Explanation", expanded=False):
            st.write("# WLA Classification Explainer")
                      
            if uploaded_file is not None: 
                # read data
                df = uploaded_file[useful_feature]
                st.success("Data loaded")
                
                #remove null on target
                df =df[df[str(target_select[0])].notnull()]

                #convert ft to class
                df['fte_label'] = np.where(df[str(target_select[0])]>=1.3, 'Overload', np.where(df[str(target_select[0])]>=1.15, 'Stretch',np.where(df[str(target_select[0])]>=0.85, 'Fit', 'Underload')))
                
                
                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                
                #remove target
                df=df.loc[:,df.columns != str(target_select[0])]
                #remove id
                df=df.loc[:,df.columns != 'EMPLOYEE_ID']
                #ordinal encoding
                df = preprocessing_encoding(df)
                #convert object to category
                df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
                
                #load model
                pickled_pipeline=None
                with open("pipeline-classification.pkl","rb") as f:
                    pipe=pickle.load(f)
                #create Xtest ytest
                kelas=list(dict.fromkeys(df['fte_label']))
                CATEGORICAL_FEATURES = df.select_dtypes('category').columns
                CATEGORICAL_FEATURES=CATEGORICAL_FEATURES[:-1]
                OBJECT_COLS = [col for col in useful_feature if col in (CATEGORICAL_FEATURES)]
                X_test=df.loc[:,df.columns != 'fte_label']
                Y_test=df.loc[:,df.columns == 'fte_label'].squeeze()
                
                #explain pipeline
                st.json(pipe.describe(return_dict=True))
                #feature importance
                st.write(pipe.graph_feature_importance())
                #permutation importance         
                objectif = ("MCC Multiclass","Log Loss Multiclass","AUC Weighted","AUC Macro","AUC Micro","Precision Weighted","Precision Macro","Precision Micro","F1 Weighted","F1 Macro","F1 Micro","Balanced Accuracy Multiclass","Accuracy Multiclass")
                option = st.selectbox('Choose Objective',objectif,key=3)
                
                st.write(graph_permutation_importance(pipe, X_test, Y_test, option))
                #partial dependence
                fitur = list(dict.fromkeys(X_test.columns))
                op_fitur = st.selectbox('fitur',options=fitur,key=4)
                st.write(graph_partial_dependence(pipe, X_test, features=op_fitur, grid_resolution=5))
                #shap
                report = explain_predictions(
                    pipeline=pipe,
                    input_features=X_test,
                    y=Y_test,
                    indices_to_explain=[0],
                    top_k_features=len(X_test.columns),
                    include_explainer_values=True,
                    output_format="dataframe",
                )
                test=report.astype(str)
                st.dataframe(test)
                rapot = explain_predictions(
                    pipeline=pipe,
                    input_features=X_test,
                    y=Y_test,
                    indices_to_explain=[0],
                    top_k_features=len(X_test.columns),
                    include_explainer_values=True,
                    output_format="text",
                )
                st.write(rapot)
                
        with st.expander("4. Populate FTE",expanded=False):
                populate = st.button("Populate")
                if populate is True and uploaded_file is not None:
                    df =uploaded_file[useful_feature]
                    nonnull_feature=list(useful_feature)
                    nonnull_feature.append('fte_label')
                    
                    df['fte_label'] = np.where(df[str(target_select[0])]>=1.3, 'Overload', np.where(df[str(target_select[0])]>=1.15, 'Stretch',np.where(df[str(target_select[0])]>=0.85, 'Fit', 'Underload')))
                    #charting before
                    st.write("BEFORE")
                    fig,ax=plt.subplots()
                    df['fte_label'].value_counts().plot(ax = ax, kind = 'bar', ylabel = 'frequency')
                    st.pyplot(fig)
                    
                    #remove null on target
                    df_notnull=df[df[str(target_select[0])].notnull()]
                    df = df[df[str(target_select[0])].isnull()]
                    #convert ft to class
                    
                    df_notnull['fte_label'] = np.where(df_notnull[str(target_select[0])]>=1.3, 'Overload', np.where(df_notnull[str(target_select[0])]>=1.15, 'Stretch',np.where(df_notnull[str(target_select[0])]>=0.85, 'Fit', 'Underload')))
                    
                    
                    # fill the missing values with mean values
                    df.fillna(df.mean(), inplace=True)
                    #select columns
                    df = df[useful_feature]
                    df = df.loc[:,df.columns != str(target_select[0])]
                    #ordinal encoding
                    df = preprocessing_encoding(df)
                    #convert object to category
                    df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
                    df_notnull[df_notnull.select_dtypes(['object']).columns] = df_notnull.select_dtypes(['object']).apply(lambda x: x.astype('category'))
                    st.subheader('Hasil Prediksi')
                    #load model
                    pickled_pipeline=None
                    with open("pipeline-classification.pkl","rb") as f:
                        pickled_pipeline=pickle.load(f)
                    # populate fte with predicted value
                    predicted_fte=pickled_pipeline.predict(df.loc[:,df.columns != 'EMPLOYEE_ID'])
                    df['fte_label']=predicted_fte
                    #convert back
                    education = { 
                        1:'Below College',
                        2:'College (D4)',
                        3:'Bachelor (S1)',
                        4:'Master',
                        5:'Doctor'
                        }
                    df.loc[:, "EDUCATION_GRADE"] = df.EDUCATION_GRADE.map(education)
                    # mapping dictionary for level pekerjaan
                    grade_pekerjaan = { 
                            1:'VI',
                            2:'V',
                            3:'IV',
                            4:'III',
                            5:'II'
                            }
                    df.loc[:, "JOB_LEVEL"] = df.JOB_LEVEL.map(grade_pekerjaan)
                    #put back non null target
                    df=df.append(df_notnull[nonnull_feature])
                    df=df.loc[:,df.columns != 'fte']
                    df=df.sort_values(by=['EMPLOYEE_ID'])
                    df = df.reset_index(drop=True)
                    st.dataframe(df)
                    st.write("AFTER")
                    fig,ax=plt.subplots()
                    df['fte_label'].value_counts().plot(ax = ax, kind = 'bar', ylabel = 'frequency')
                    st.pyplot(fig)
                    csv = convert_df(df)
                    st.download_button(
                        label="📥 Download data as CSV",
                        data=csv,
                        file_name='predicted_WLA-class_batch.csv',
                        mime='text/csv',
                        key=5)

    

    with tab2:
        st.write("# WLA (FTE) - ML Regression Modeler")
        with st.expander("Introduction"):
            st.header("Pemodelan Work Load Analysis (WLA) menggunakan metode Full Time Equivalent (FTE) dengan pemodelan Machine Learning Regression")
            st.markdown(
            """
                Beban kerja (Workload) adalah kumpulan atau sejumlah aktivitas yang harus dilakukan oleh suatu unit dalam organisasi atau pemegang jabatan dalam jangka waktu yang telah ditentukan, biasanya disetahunkan. Ada banyak metode dalam menghitung Workload, yang masing masing memiliki kelebihan dan kekurangan. Dari sekian banyak metode yang dapat digunakan untuk menganalisis Workload, terdapat metode yang disebut dengan metode Full-Time Equivalent (FTE). Hasil perhitungan analisis beban kerja melalui metode FTE akan menghasilkan output apakah sebagian pekerja atau karyawan mengalami beban kerja overload, stretch, fit, atau underload. 

                Di telkom riwayat aktivitas tercatat pada diarium. Namun pengisiannya masih belum lengkap. dan meningkatkan keakurasian pengisian aktivitas di diarium dirasa akan memberikan tambahan pekerjaan pada karyawan telkom. Sedangkan metode kuesioner juga akan sangat menyita waktu. 

                Oleh karena itu kita mencoba menghitung nilai FTE berdasarkan isian diarium. kemudian membuat sebuah pemodelan yang mampu mengkelompokkan karyawan dan memperkirakan nilai fte nya. Model ini nantinya akan digunakan untuk memprediksi karyawan yang fte nya kosong atau anomali.

                Metode yang digunakan adalah metode regression, jadi model akan di training dengan menggunakan data yang ada untuk kemudian membuat model yang mampu mengenali dengan fitur fiture tertentu, maka nilai fte yang seharusnya dieproleh karyawan dengan fitur fitur tersebut. Sehingga hasil akhir dari pemodelan ini adalah nilai fte, dan bukan sekdar kelas dari fte karyawan. Nilai fte ini berguna nantinya dalam menentukan Gap antara fte demand dan fte supply. fte demand adalah kebutuhan fte suatu unit untuk mencapai target tertentu. fte supply adalah fte yang dimiliki unit tersebut pada saat tertentu. Berdasarkan hasil running dengan data yang tersedia, metode regression ini dirasa kurang mampu menangkap pola yang ada antara fitur dengan target. Hal ini dibuktikan dengan nilai r2 yang rendah, bahakan negatif. pemodelan yang bagus akan menghasilkan nilai r2 mendekati 1. Oleh karena itu, kami sangat menyarankan agar future work akan melibatkan fitur yang lebih banyak, data yang lebih akurat, sehingga nilai r2 dari model yang digenerate meningkat mendekati nilai 1. Untuk mengakomodasi future work tersebut, kami telah memberikan keleluasaan bagi user untuk menambah kolom sebanyak apapun pada template excel yang tersedia, dan memberikan user kebebesan untuk menentukan mana yang perlu masuk sebagai futur atau tidak dengan memberikan baris pada template excel untuk memberikan keterangan boolean dari kolom yang ada di excel.

                Dan kesemuanya dilakukan dalam tiga langkah, yaitu: Load Data, Pembuatan Model, dan Menggunakan Model yang tealh dibuat untuk menentukan nilai FTE nya.
            """
            )
        st.markdown("Langkah")    
        with st.expander("1. Upload file Excel Data WLA Telkom.xlsx",expanded=True):
            uploaded_file=pd.read_sql_table("data_model", engine)
            col1,col2=st.columns(2)
            with col1:
                feature_select=st.multiselect("pilih features", uploaded_file.columns, ['EMPLOYEE_ID', 'AGE', 'GENDER','MARITAL_STATUS','EDUCATION_GRADE','RECRUITMENT_STATUS', 'EMPLOYMENT_STATUS', 'YEARS_AT_COMPANY','DEPARTMENT','JOB_LEVEL', 'JOB_NATURE', 'TASK_TYPE','THP_CURRENT', 'thp_increase', 'LAST_MUTATION','LAST_PROMOTION','training_duration_2022', 'leadership_score','functional_score', 'budaya_score'], key=6) 
            with col2:
                target_select=st.multiselect("pilih target", uploaded_file.columns, 'fte', key=7)
               
            if uploaded_file is not None:
                useful_feature=list(feature_select+target_select)
                st.success("Data loaded")
                
        with st.expander("2. Build Model",expanded=False):
            if uploaded_file is not None:     
                 # read data
                df = uploaded_file[useful_feature]
                #remove null on target
                df = df.dropna(subset=target_select)
                #ordinal encoding
                df = preprocessing_encoding(df)
                #make corr graph
                corr = df.corr()
                st.write( corr.style.background_gradient(cmap='coolwarm').set_precision(2))
                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                # remove column employee_id
                df=df.loc[:,df.columns != 'EMPLOYEE_ID']
                #define x
                x=df.drop([str(target_select[0])],axis=1,inplace=False)
                #define y
                y=df[str(target_select[0])]
                #define modeling problem
                problem=evalml.problem_types.detect_problem_type(y.squeeze(axis=0))
                #select objectives based on problem
                if str(problem)=="multiclass":
                    objectif = ("MCC Multiclass","Log Loss Multiclass","AUC Weighted","AUC Macro","AUC Micro","Precision Weighted","Precision Macro","Precision Micro","F1 Weighted","F1 Macro","F1 Micro","Balanced Accuracy Multiclass","Accuracy Multiclass")
                elif str(problem)=="binary":
                    objectif = ("MCC Binary","Log Loss Binary","Gini","AUC","Precision","F1","Balanced Accuracy Binary","Accuracy Binary")
                elif str(problem)=="regression":
                    objectif = ("ExpVariance","MaxError","MedianAE","MSE","MAE","R2","Root Mean Squared Error")
                option = st.selectbox('Choose Objective',objectif,key=8)
                clicked = st.button("Build Model",key=9)
            if uploaded_file is not None and clicked is True:
                #convert df to array
                y_res=y.squeeze()
                #Modeling
                ##splitting
                X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(x,y_res, problem_type=problem)
                #check data problem
                st.write("CHECKING DATA PROBLEM :")
                data_checks = DefaultDataChecks(problem, objective=option)
                if data_checks.validate(X_train, y_train)==[]:
                    st.info(" - no problem with data")
                else:
                    st.info(data_checks.validate(X_train, y_train))
                ##find the best pipeline
                st.write("FINDING THE BEST MODEL")
                with st.spinner(text="In progress..."):
                    automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type=problem,objective=option,random_seed=72,ensembling=True,allow_long_running_models=True,verbose=True)
                    automl.search()
                    st.success("Best Model Found:")
                    st.success(automl.best_pipeline.name)
                ##Evaluate on hold out data
                st.markdown("EVALUATE DATA")
                st.json(automl.best_pipeline.score(X_test, y_test, objectives=[option]))
                st.write("WRITING MODEL")
                pipeline_to_pickle = automl.best_pipeline
                with open("pipeline-reg.pkl","wb") as f:
                    pickle.dump(pipeline_to_pickle,f)
                st.success("MODEL BUILT! 🥳")
                pickled_pipeline=None
                with open("pipeline-reg.pkl","rb") as f:
                    pickled_pipeline=pickle.load(f)
                    st.download_button(label='📥 Download pipeline-reg.pkl',data=f ,
                                                file_name= 'pipeline-reg.pkl', key=10)
        with st.expander("3. Model explanation", expanded=False):
            if uploaded_file is not None:
                # read data
                df = uploaded_file[useful_feature]
                st.success("Data loaded")
                #remove null on target
                df =df[df[str(target_select[0])].notnull()]
                #ordinal encoding
                df = preprocessing_encoding(df)
                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                # remove column employee_id
                df=df.loc[:,df.columns != 'EMPLOYEE_ID']
                
                #convert object to category
                df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
                
                #load model
                pickled_pipeline=None
                with open("pipeline-reg.pkl","rb") as f:
                    pipe=pickle.load(f)
                                
                #create Xtest ytest
                #kelas=list(dict.fromkeys(df[str(target_col[0,0])]))
                CATEGORICAL_FEATURES = df.select_dtypes('category').columns
                CATEGORICAL_FEATURES=CATEGORICAL_FEATURES[:-1]
                OBJECT_COLS = [col for col in useful_feature if col in (CATEGORICAL_FEATURES)]
                X_test=df.loc[:,df.columns != str(target_select[0])]
                Y_test=df.loc[:,df.columns == str(target_select[0])].squeeze()
                #explain pipeline
                st.json(pipe.describe(return_dict=True))
                #feature importance
                st.write(pipe.graph_feature_importance())
                objectif = ("ExpVariance","MaxError","MedianAE","MSE","MAE","R2","Root Mean Squared Error")
                option = st.selectbox('Choose Objective',objectif,key=11)
                #permutation importance         
                st.write(graph_permutation_importance(pipe, X_test, Y_test, option))
                #partial dependence
                fitur = list(dict.fromkeys(X_test.columns))
                op_fitur = st.selectbox('fitur',options=fitur,key=12)
                st.write(graph_partial_dependence(pipe, X_test, features=op_fitur, grid_resolution=5))
                report = explain_predictions(
                    pipeline=pipe,
                    input_features=X_test,
                    y=Y_test,
                    indices_to_explain=[0],
                    top_k_features=len(X_test.columns),
                    include_explainer_values=True,
                    output_format="text",
                )
                st.write(report)
                
                
        with st.expander("4. Populate FTE",expanded=False):
                populate = st.button("Populate", key=13)
                st.warning("WARNING make sure there is no nan value on features value. All empty rows will be filled with mean value")
                st.success("Data loaded")
                if populate is True:
                    df =uploaded_file[useful_feature]
                    #charting before
                    st.write("BEFORE")    
                    arr=df[str(target_select[0])]
                    fig,ax=plt.subplots()
                    ax.hist(arr, bins=20)
                    st.pyplot(fig)
                    #remove null on target
                    df_notnull=df[df[str(target_select[0])].notnull()]
                    df = df[df[str(target_select[0])].isnull()]
                    # fill the missing values with mean values
                    df.fillna(df.mean(), inplace=True)

                    #select columns
                    df = df[useful_feature]
                    df = df.loc[:,df.columns != str(target_select[0])]
                    
                    #ordinal encoding
                    df = preprocessing_encoding(df)
                    #convert object to category
                    df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
                    df_notnull[df_notnull.select_dtypes(['object']).columns] = df_notnull.select_dtypes(['object']).apply(lambda x: x.astype('category'))
                    st.subheader('Hasil Prediksi')
                    #load model
                    pickled_pipeline=None
                    with open("pipeline-reg.pkl","rb") as f:
                        pickled_pipeline=pickle.load(f)
                    # populate fte with predicted value
                    predicted_fte=pickled_pipeline.predict(df.loc[:,df.columns != 'EMPLOYEE_ID'])
                    df[str(target_select[0])]=predicted_fte
                    #convert back
                    education = { 
                        1:'Below College',
                        2:'College (D4)',
                        3:'Bachelor (S1)',
                        4:'Master',
                        5:'Doctor'
                        }
                    df.loc[:, "EDUCATION_GRADE"] = df.EDUCATION_GRADE.map(education)
                    # mapping dictionary for level pekerjaan
                    grade_pekerjaan = { 
                            1:'VI',
                            2:'V',
                            3:'IV',
                            4:'III',
                            5:'II'
                            }
                    df.loc[:, "JOB_LEVEL"] = df.JOB_LEVEL.map(grade_pekerjaan)
                    #put back non null target
                    df=df.append(df_notnull[useful_feature])
                    df=df.sort_values(by=['EMPLOYEE_ID'])
                    df = df.reset_index(drop=True)
                    st.dataframe(df)
                    st.write('AFTER')
                    arr=df[str(target_select[0])]
                    fig,ax=plt.subplots()
                    ax.hist(arr, bins=20)
                    st.pyplot(fig)
                    csv = convert_df(df)
                    st.download_button(
                        label="📥 Download data as CSV",
                        data=csv,
                        file_name='predicted_WLA-reg_batch.csv',
                        mime='text/csv',
                        key=14)

                    
    with tab3:
        st.write("# WFP - ML Regression Modeler")
        with st.expander("Introduction"):
            st.header("Pemodelan Work Force Planning (WFP) dengan pemodelan Machine Learning Regression")
            st.markdown(
            """
            Dalam sebuah unit dalam mencapai targetnya, tentunya diperlukan resource yang memadai untuk mencapainya. Dengan target seperti potensi revenue yang ada, revenue yang tercapai, jumlah pelanggan. Dan dengan resource berupa, jumlah produk, jumlah karyawan organik, dengan berbagai kompetensinya. Baik kompetensi soft skill dan hard skill. Diharapkan unit mampu mencapai target tersebut dengan mengoptimalkan resource yang ada. Pengoptimalan resource, terutama pada sumber daya manusianya, tergambar dari jumlah total full Time Equivalent. Semakin tinggi nilai FTE, menggambarkan semakin banyak jam yang digunakan unit tersebut dalam melakukan kegiatan produktif. 

            Perlu diingat, bahwa nilai FTE ini tidak serta merta berbanding lurus dengan jumlah karyawan yang dimiliki unit. Apabila jumlah karyawan banyak. Namun melakukan sedikit pekerjaan produktif, tentunya nilai FTE nya akan rendah. Sebaliknya jika jumlah karyawan sedikit, namun sangat produktif, tentunya nilai FTE nya akan tinggi. 

            Data yang digunakan dalam pemodelan ini, hanyalah dummy data. Data yang dipakai dalam pemodelan ini 100% random. Dan apabila User memiliki data yang lebih akurat sesuai dengan kondisi di lapangan. Hasil dari pemodelan tentunya akan dapat dipertanggungjawabkan. Pemodelan WFP demand ini, mencoba untuk memprediksi jumlah FTE yang dibutuhkan dalam mencapai target. Dengan keterangan data sebagai berikut : 
        Categorical:
        -departemen
        -tahun
        Independent
        -target_revenue 
        -pelanggan
        -potensi_revenue
        -jumlah_produk
        -karyawan_organik
        -pencapaian_revenue
        -functional_1
        -functional_2
        -functional_3
        -functional_4
        -functional_5
        -soft_competency_1
        -soft_competency_2
        -soft_competency_3
        -soft_competency_4
        -soft_competency_5

            Beberapa asumsi yang digunakan tercapainya revenue adalah 100% dikarenakan kontribusi karyawan organik. hanya 5 kompetensi functional dan soft yang dimiliki oleh karyawan. dan beberapa batasan lainnya. Dari hasil running beberapa kali dengan data yang ada dan berbagai metode pemodelan, metode regresi memiliki tingkat akurasi yang rendah. metode regression ini dirasa kurang mampu menangkap pola yang ada antara fitur dengan target. Hal ini dibuktikan dengan nilai r2 yang rendah, bahakan negatif. pemodelan yang bagus akan menghasilkan nilai r2 mendekati 1. Oleh karena itu, kami sangat menyarankan agar future work akan melibatkan fitur yang lebih banyak, data yang lebih akurat, sehingga nilai r2 dari model yang digenerate meningkat mendekati nilai 1.

            Oleh karena itu, kami sangat menyarankan agar future work akan melibatkan fitur yang lebih banyak, data yang lebih akurat, sehingga nilai r2 dari model yang digenerate meningkat mendekati nilai 1. Untuk mengakomodasi future work tersebut, kami telah memberikan keleluasaan bagi user untuk menambah kolom sebanyak apapun pada template excel yang tersedia, dan memberikan user kebebesan untuk menentukan mana yang perlu masuk sebagai futur atau tidak dengan memberikan baris pada template excel untuk memberikan keterangan boolean dari kolom yang ada di excel.
            """
            )
        st.markdown("Langkah")    
        with st.expander("1. Upload file Excel Data WFP.xlsx",expanded=True):
            uploaded=pd.read_sql_table("data_wfp", engine)
            
            col1,col2=st.columns(2)
            with col1:
                feature_select=st.multiselect("pilih features", uploaded.columns, ["departemen","tahun", "target_revenue","pelanggan", "potensi_revenue","jumlah_produk","karyawan_organik", "pencapaian_revenue","functional_1","functional_2","functional_3","functional_4","functional_5","soft_competency_1","soft_competency_2","soft_competency_3","soft_competency_4","soft_competency_5","jumlah_okr","pencapaian_okr","Productivity"], key=15) 
            with col2:
                target_select=st.multiselect("pilih target", uploaded.columns, "fte", key=16)
                useful_feature=list(feature_select+target_select)
            if uploaded is not None:
                df = uploaded[useful_feature]
                st.success("Data loaded")
                
        with st.expander("2. Build Model",expanded=False):
            if uploaded is not None:
                #remove null on target
                df = df[df[str(target_select[0])].notnull()]    
                #make corr graph
                corr = df.corr()
                st.write( corr.style.background_gradient(cmap='coolwarm').set_precision(2))
                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                #define x
                x=df.drop([str(target_select[0])],axis=1,inplace=False)
                #define y
                y=df[str(target_select[0])]
                #define modeling problem
                problem=evalml.problem_types.detect_problem_type(y.squeeze(axis=0))
                #select objectives based on problem
                if str(problem)=="multiclass":
                    objectif = ("MCC Multiclass","Log Loss Multiclass","AUC Weighted","AUC Macro","AUC Micro","Precision Weighted","Precision Macro","Precision Micro","F1 Weighted","F1 Macro","F1 Micro","Balanced Accuracy Multiclass","Accuracy Multiclass")
                elif str(problem)=="binary":
                    objectif = ("MCC Binary","Log Loss Binary","Gini","AUC","Precision","F1","Balanced Accuracy Binary","Accuracy Binary")
                elif str(problem)=="regression":
                    objectif = ("ExpVariance","MaxError","MedianAE","MSE","MAE","R2","Root Mean Squared Error")
                option = st.selectbox('Choose Objective',objectif,key=17)
                clicked = st.button("Build Model",key=18)
            if uploaded is not None and clicked is True:
                #convert df to array
                y_res=y.squeeze()
                #Modeling
                ##splitting
                X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(x,y_res, problem_type=problem)
                #check data problem
                st.write("CHECKING DATA PROBLEM :")
                data_checks = DefaultDataChecks(problem, objective=option)
                if data_checks.validate(X_train, y_train)==[]:
                    st.info("no problem with data")
                else:
                    st.info(data_checks.validate(X_train, y_train))
                ##find the best pipeline
                st.write("FINDING THE BEST MODEL")
                with st.spinner(text="In progress..."):
                    automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type=problem,objective=option,random_seed=72,ensembling=True,allow_long_running_models=True,verbose=True)
                    automl.search()
                    st.success("Best Model Found:")
                    st.success(automl.best_pipeline.name)
                ##Evaluate on hold out data
                st.markdown("EVALUATE DATA")
                st.json(automl.best_pipeline.score(X_test, y_test,objectives=[option]))
                st.write("WRITING MODEL")
                pipeline_to_pickle = automl.best_pipeline
                with open("pipelineWFP-regression.pkl","wb") as f:
                    pickle.dump(pipeline_to_pickle,f)
                st.success("MODEL BUILT! 🥳")
                pickled_pipeline=None
                   
                with open("pipelineWFP-regression.pkl","rb") as f:
                    pickled_pipeline=pickle.load(f)
                    st.download_button(label='📥 Download pipelineWFP-regression.pkl',data=f ,
                                                file_name= 'pipelineWFP-regression.pkl', key=19)

        with st.expander("3. Model Explanation",expanded=False):
            st.write("# WFP Regression Explainer")
            if uploaded is not None:
                # read data
                df = uploaded[useful_feature]
                st.success("Data loaded")
                #remove null on target
                df =df[df[str(target_select[0])].notnull()]
                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                
                #convert object to category
                df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
                
                #load model
                pickled_pipeline=None
                with open("pipelineWFP-regression.pkl","rb") as f:
                    pipe=pickle.load(f)
                                
                #create Xtest ytest
                #kelas=list(dict.fromkeys(df[str(target_col[0,0])]))
                CATEGORICAL_FEATURES = df.select_dtypes('category').columns
                CATEGORICAL_FEATURES=CATEGORICAL_FEATURES[:-1]
                OBJECT_COLS = [col for col in useful_feature if col in (CATEGORICAL_FEATURES)]
                X_test=df.loc[:,df.columns != str(target_select[0])]
                Y_test=df.loc[:,df.columns == str(target_select[0])].squeeze()
                #explain pipeline
                st.json(pipe.describe(return_dict=True))
                #feature importance
                st.write(pipe.graph_feature_importance())
                objectif = ("ExpVariance","MaxError","MedianAE","MSE","MAE","R2","Root Mean Squared Error")
                option = st.selectbox('Choose Objective',objectif,key=20)
                #permutation importance         
                st.write(graph_permutation_importance(pipe, X_test, Y_test, option))
                #partial dependence
                fitur = list(dict.fromkeys(X_test.columns))
                op_fitur = st.selectbox('fitur',options=fitur,key=21)
                st.write(graph_partial_dependence(pipe, X_test, features=op_fitur, grid_resolution=5))
                report = explain_predictions(
                    pipeline=pipe,
                    input_features=X_test,
                    y=Y_test,
                    indices_to_explain=[0],
                    top_k_features=len(X_test.columns),
                    include_explainer_values=True,
                    output_format="text",
                )
                st.write(report)
 
        with st.expander("4. Predict FTE Unit",expanded=False):
                df = uploaded[useful_feature]
                if uploaded is not None: 
                    st.success("Data loaded")
                    populate = st.button("Populate",key=22)
                if uploaded is not None and populate is True:
                    #charting before
                    st.write("BEFORE")
                    arr=df[str(target_select[0])]
                    fig,ax=plt.subplots()
                    ax.hist(arr, bins=20)
                    st.pyplot(fig)
                    # remove nan in target columns
                    df_notnull=df[df[str(target_select[0])].notnull()]
                    df = df[df[str(target_select[0])].isnull()]      
                    df = df.loc[:,df.columns != 'fte']
                    df = df.loc[:,df.columns != 'EMPLOYEE_ID']
                    
                    # fill the missing values with mean values
                    df.fillna(df.mean(), inplace=True)
                    #convert object to category
                    df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
                    df_notnull[df_notnull.select_dtypes(['object']).columns] = df_notnull.select_dtypes(['object']).apply(lambda x: x.astype('category'))
                    st.subheader('Hasil Prediksi')
                    #load model
                    pickled_pipeline=None
                    with open("pipelineWFP-regression.pkl","rb") as f:
                        pickled_pipeline=pickle.load(f)
                    # populate fte with predicted value
                    predicted_fte=pickled_pipeline.predict(df.loc[:,df.columns != str(target_select[0])])
                    df[str(target_select[0])]=predicted_fte
                    df=df.append(df_notnull[useful_feature])
                    st.dataframe(df)
                    st.write("AFTER")
                    arr=df[str(target_select[0])]
                    fig,ax=plt.subplots()
                    ax.hist(arr, bins=20)
                    st.pyplot(fig)
                    csv = convert_df(df)
                    st.download_button(
                        label="📥 Download data as CSV",
                        data=csv,
                        file_name='predicted_WFP-reg_batch.csv',
                        mime='text/csv',
                        key=23)
                    
def page5():
    st.write("# Check WLA Gap per unit")
    with st.expander("DATA WFP", expanded=True):
        #upload wfp
        uploaded = st.file_uploader("Silakan upload file: predicted_WFP-reg_batch.csv",key=24)
        
        if uploaded is not None:
            uploaded_wfp = pd.read_csv(uploaded)
            st.success("Data WFP loaded")
            df_wfp = uploaded_wfp
    with st.expander("DATA WLA", expanded=True):
        #upload wla
        uploaded_file = st.file_uploader("Silakan upload file: predicted_WLA-reg_batch.csv",key=27)
        
        if uploaded_file is not None:
            uploaded_wla = pd.read_csv(uploaded_file)
            st.success("Data WLA loaded")
            df_wla = uploaded_wla   
                   
    with st.expander("DATA SELECTOR",expanded=True):
        if uploaded is not None and uploaded_file is not None:
            col1,col2=st.columns(2)
            departemen = list(dict.fromkeys(df_wfp['departemen']))
            tahun = list(dict.fromkeys(df_wfp['tahun']))
            with col1:
                op_dep = st.selectbox('departemen',departemen,index=1, key=25)
            with col2:
                op_tahun = st.selectbox('tahun',tahun,index=1, key=26)
            if op_dep is not None and op_tahun is not None:
                st.write("Demand FTE : ")
                fte_demand=df_wfp.loc[(df_wfp['departemen']==op_dep) & (df_wfp['tahun'] == op_tahun),"fte"].item()
                fte_demand=str(fte_demand)
                st.write(fte_demand)
            most_right=list(df_wla.columns)[-1]
            fte_supply=df_wla.loc[(df_wla['DEPARTMENT']==op_dep[0]),most_right].sum() 
    with st.expander("GAP FTE DEMAND vs SUPPLY",expanded=True):
        if uploaded is not None and uploaded_file is not None:
            fte_supply=str(fte_supply)
            st.write('FTE DEMAND :',fte_demand)
            st.write('FTE SUPPLY :',fte_supply)
            gap=float(fte_demand)-float(fte_supply)
            st.write('## GAP fte_demand - fte_supply :', gap)

def page6():
    tab1,tab2,tab3=st.tabs(["WLA Classification Predictor", "WLA Regression Predictor", "WFP Regression Predictor"])
    with tab1:
        st.write("# WLA Classification Predictor")
        upl_modelWLAclass = st.file_uploader("Upload file pipeline-classification.pkl",key=28)
        tab4, tab5 = st.tabs(["Batch", "Individual"])
        with tab4:
            uploaded_file=pd.read_sql_table("data_model", engine)
            st.warning("WARNING make sure there is no nan value on features value. All empty rows will be filled with mean value")
            if uploaded_file is not None: 
                # read data
                col1,col2=st.columns(2)
            with col1:
                feature_select=st.multiselect("pilih features", uploaded_file.columns, 
                                              ['EMPLOYEE_ID', 'AGE', 'GENDER', 'MARITAL_STATUS', 'EDUCATION_GRADE',
                                               'RECRUITMENT_STATUS', 'EMPLOYMENT_STATUS', 'YEARS_AT_COMPANY','DEPARTMENT',
                                               'JOB_LEVEL', 'JOB_NATURE', 'TASK_TYPE',
                                               'THP_CURRENT', 'thp_increase', 'LAST_MUTATION',
                                               'LAST_PROMOTION','training_duration_2022', 'leadership_score',
                                               'functional_score', 'budaya_score']) 
            with col2:
                target_select=st.multiselect("pilih target", uploaded_file.columns, 'fte')
            if uploaded_file is not None:
                useful_feature=list(feature_select+target_select)
                nonnull_feature=np.append(useful_feature,'fte_label')
                st.success("Data loaded")
                populate = st.button("Populate", key=29)
            if uploaded_file is not None and populate is True:
                df = uploaded_file[useful_feature]
                #convert ft to class
                df['fte_label'] = np.where(df[str(target_select[0])]>=1.3, 'Overload', np.where(df[str(target_select[0])]>=1.15, 'Stretch',np.where(df[str(target_select[0])]>=0.85, 'Fit', 'Underload')))
             
                #charting before
                st.write("BEFORE")
                fig,ax=plt.subplots()
                df['fte_label'].value_counts().plot(ax = ax, kind = 'bar', ylabel = 'frequency')
                st.pyplot(fig)
                
                #remove null on target
                df_notnull=df[df[str(target_select[0])].notnull()]
                df = df[df[str(target_select[0])].isnull()]
                #convert ft to class
                df_notnull['fte_label'] = np.where(df_notnull[str(target_select[0])]>=1.3, 'Overload', np.where(df_notnull[str(target_select[0])]>=1.15, 'Stretch',np.where(df_notnull[str(target_select[0])]>=0.85, 'Fit', 'Underload')))
             
          

                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                #select columns
                df = df.loc[:,df.columns != str(target_select[0])]
                df = df.loc[:,df.columns != 'fte_label']
                
                #ordinal encoding
                df = preprocessing_encoding(df)
                #convert object to category
                df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
                df_notnull[df_notnull.select_dtypes(['object']).columns] = df_notnull.select_dtypes(['object']).apply(lambda x: x.astype('category'))
                st.subheader('Hasil Prediksi')
                #load model
                pickled_pipeline=pd.read_pickle(upl_modelWLAclass)
                # populate fte with predicted value
                predicted_fte=pickled_pipeline.predict(df.loc[:,df.columns != 'EMPLOYEE_ID'])
                df['fte_label']=predicted_fte
                #convert back
                education = { 
                    1:'Below College',
                    2:'College (D4)',
                    3:'Bachelor (S1)',
                    4:'Master',
                    5:'Doctor'
                    }
                df.loc[:, "EDUCATION_GRADE"] = df.EDUCATION_GRADE.map(education)
                # mapping dictionary for level pekerjaan
                grade_pekerjaan = { 
                        1:'VI',
                        2:'V',
                        3:'IV',
                        4:'III',
                        5:'II'
                        }
                df.loc[:, "JOB_LEVEL"] = df.JOB_LEVEL.map(grade_pekerjaan)
                #put back non null target
                df=df.append(df_notnull[nonnull_feature])
                st.dataframe(df)
                st.write("AFTER")
                fig,ax=plt.subplots()
                df['fte_label'].value_counts().plot(ax = ax, kind = 'bar', ylabel = 'frequency')
                st.pyplot(fig)
                csv = convert_df(df)
                st.download_button(
                    label="📥 Download data as CSV",
                    data=csv,
                    file_name='predicted_WLA-class_batch.csv',
                    mime='text/csv',
                    key=30)
        with tab5:
            # Prediksi Individu
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('**_Profil Pegawai_**')
                Age = st.number_input(label = 'Umur Pegawai', min_value = 0, max_value = 99 ,value = 25, step = 1, key=31)
                Gender = st.selectbox('Gender', ('PRIA','WANITA'), key=32)
                Marital_Status = st.selectbox('Status Perkawinan', ('Single','Nikah','Janda','Duda'), key=33)
                Education_Grade = st.selectbox('Tingkat Pendidikan', ('Below College','College (D4)','Bachelor (S1)','Master','Doctor'), key=34)
            with col2:
                st.write('**_Status Pegawai_**')
                Recruitment_Status = st.selectbox('Status Rekruitment', ('Aktif Normal','Pemangku Jabatan(PJ)','Rehire','Tenaga Profesional'))
                Employment_Status = st.selectbox('Status Pegawai', ('PKWTT','PKWT'))
                Job_Level = st.selectbox('Level Pekerjaan', ('VI','V','IV','III','II'))
                Job_Nature = st.selectbox('Nature Pekerjaan',('Desk', 'Field'))
            with col3:
                st.write('**_Pekerjaan_**')
                Department = st.selectbox('Departemen Kerja',('WITEL JAKBAR', 'WITEL BANTEN', 'DIG TECH & PLATFORM BUSINESS'))
                Task_type = st.selectbox('Tipe Pekerjaan',('Clerical', 'Analytical','Physical'))
                Percent_Salary_Hike = st.number_input(label = 'Percent kenaikan gaji', min_value = 0.0,max_value = 0.5, value = 0.2, step = 0.05, key=35)
            st.markdown ('---')
            col4, col5 = st.columns(2)
            with col4:
                Years_at_Company = st.slider(label = 'Lama Bekerja', min_value = 0,
                                    max_value = 60 ,
                                    value = 10,
                                    step = 1, key=36)
                Last_promotion =  st.slider(label = 'Promosi terakhir', min_value = 0,
                                    max_value = 300 ,
                                    value = 1,
                                    step = 1, key=37)
                Last_mutation =  st.slider(label = 'Mutasi terakhir', min_value = 0,
                                    max_value = 100 ,
                                    value = 1,
                                    step = 1, key=38)
                training_duration_2022= st.slider(label = 'Durasi Training', min_value = 0.0,
                                    max_value = 200.0,
                                    value = 30.0,
                                    step = 1.0, key=39)
                THP_current= st.slider(label = 'Gaji saat ini', min_value = 0.0,
                                    max_value = 3.0 ,
                                    value = 2.5,
                                    step = 0.1, key=40)

            with col5:
                LEADERSHIP_2021 = st.slider(label = 'Score Leadership', min_value = 0.0,
                                    max_value = 1.0 ,
                                    value = 0.8,
                                    step = 0.1, key=41)
                BUDAYA_2021 = st.slider(label = 'Score Budaya', min_value = 0.0,
                                    max_value = 1.0 ,
                                    value = 0.8,
                                    step = 0.1, key=42)
                FUNCTIONAL_2021 = st.slider(label = 'Score Functional', min_value = 0.0,
                                    max_value = 1.0 ,
                                    value = 0.8,
                                    step = 0.1, key=43)
            st.markdown ('---')
            features = {
                        'AGE': Age,
                        'GENDER':Gender,
                        'MARITAL_STATUS': Marital_Status,
                        'EDUCATION_GRADE': Education_Grade,
                        'RECRUITMENT_STATUS':Recruitment_Status,
                        'EMPLOYMENT_STATUS': Employment_Status,
                        'YEARS_AT_COMPANY':Years_at_Company,  
                        'DEPARTMENT':Department,
                        'JOB_LEVEL':Job_Level,
                        'JOB_NATURE':Job_Nature,
                        'TASK_TYPE':Task_type,
                        'THP_CURRENT':THP_current,
                        'thp_increase':Percent_Salary_Hike,
                        'LAST_MUTATION':Last_mutation,
                        'LAST_PROMOTION':Last_promotion,
                        'training_duration_2022':training_duration_2022,
                        'leadership_score':LEADERSHIP_2021,
                        'functional_score':FUNCTIONAL_2021,
                        'budaya_score':BUDAYA_2021,
                        }
            features_df  = pd.DataFrame([features])
            features_df = preprocessing_encoding(features_df)
            features_df[features_df.select_dtypes(['object']).columns] = features_df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
            if upl_modelWLAclass is not None:
                klik = st.button("Predict", key=44)
                if klik:
                    #load model
                    pickled_pipeline=pd.read_pickle(upl_modelWLAclass)
                    predicted_fte=pickled_pipeline.predict(features_df)
                    st.success(predicted_fte.to_string()[2:])
    with tab2:
        st.write("# WLA Regression Predictor")
        upl_modelWLAreg = st.file_uploader("Upload file pipeline-reg.pkl",key=45)
      
        tab6, tab7 = st.tabs(["Batch", "Individual"])
        with tab6:
            uploaded_file=pd.read_sql_table("data_model", engine)
            col1,col2=st.columns(2)
            with col1:
                feature_select=st.multiselect("pilih features", uploaded_file.columns, 
                                              ['EMPLOYEE_ID', 'AGE', 'GENDER', 'MARITAL_STATUS', 'EDUCATION_GRADE',
                                               'RECRUITMENT_STATUS', 'EMPLOYMENT_STATUS', 'YEARS_AT_COMPANY','DEPARTMENT',
                                               'JOB_LEVEL', 'JOB_NATURE', 'TASK_TYPE',
                                               'THP_CURRENT', 'thp_increase', 'LAST_MUTATION',
                                               'LAST_PROMOTION','training_duration_2022', 'leadership_score',
                                               'functional_score', 'budaya_score'], key=46) 
            with col2:
                target_select=st.multiselect("pilih target", uploaded_file.columns, 'fte', key=47)
            if uploaded_file is not None:
                useful_feature=list(feature_select+target_select)

            st.warning("WARNING make sure there is no nan value on features value. All empty rows will be filled with mean value")
            if uploaded_file is not None: 
                # read data
                df = uploaded_file[useful_feature]
                st.success("Data loaded")
                populate = st.button("Populate",key=48)
            if uploaded_file is not None and populate is True:
                #charting before
                st.write("BEFORE")
                arr=df[str(target_select[0])]
                fig,ax=plt.subplots()
                ax.hist(arr, bins=20)
                st.pyplot(fig)
                #remove null on target
                df_notnull=df[df[str(target_select[0])].notnull()]
                df = df[df[str(target_select[0])].isnull()]
                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                #select columns
                df = df[useful_feature]
                df = df.loc[:,df.columns != str(target_select[0])]
                #ordinal encoding
                df = preprocessing_encoding(df)
                #convert object to category
                df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
                df_notnull[df_notnull.select_dtypes(['object']).columns] = df_notnull.select_dtypes(['object']).apply(lambda x: x.astype('category'))
                st.subheader('Hasil Prediksi')
                #load model
                pickled_pipeline=pd.read_pickle(upl_modelWLAreg)
                # populate fte with predicted value
                predicted_fte=pickled_pipeline.predict(df.loc[:,df.columns != 'EMPLOYEE_ID'])
                df[str(target_select[0])]=predicted_fte
                #convert back
                education = { 
                    1:'Below College',
                    2:'College (D4)',
                    3:'Bachelor (S1)',
                    4:'Master',
                    5:'Doctor'
                    }
                df.loc[:, "EDUCATION_GRADE"] = df.EDUCATION_GRADE.map(education)
                # mapping dictionary for level pekerjaan
                grade_pekerjaan = { 
                        1:'VI',
                        2:'V',
                        3:'IV',
                        4:'III',
                        5:'II'
                        }
                df.loc[:, "JOB_LEVEL"] = df.JOB_LEVEL.map(grade_pekerjaan)
                #put back non null target
                df=df.append(df_notnull[useful_feature])
                st.dataframe(df)
                st.write('AFTER')
                arr=df[str(target_select[0])]
                fig,ax=plt.subplots()
                ax.hist(arr, bins=20)
                st.pyplot(fig)
                csv = convert_df(df)
                st.download_button(
                    label="📥 Download data as CSV",
                    data=csv,
                    file_name='predicted_WLA-reg_batch.csv',
                    mime='text/csv',
                    key=49)
        
        with tab7:
            # Prediksi Individu
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('**_Profil Pegawai_**')
                Age = st.number_input(label = 'Umur Pegawai', min_value = 0,
                                    max_value = 99 ,
                                    value = 25,
                                    step = 1, key=50)
                Gender = st.selectbox('Gender', ('PRIA','WANITA'),key=51)
                Marital_Status = st.selectbox('Status Perkawinan', ('Single','Nikah','Janda','Duda'),key=52)
                Education_Grade = st.selectbox('Tingkat Pendidikan', ('Below College','College (D4)','Bachelor (S1)','Master','Doctor'),key=53)
            with col2:
                st.write('**_Status Pegawai_**')
                Recruitment_Status = st.selectbox('Status Rekruitment', ('Aktif Normal','Pemangku Jabatan(PJ)','Rehire','Tenaga Profesional'),key=54)
                Employment_Status = st.selectbox('Status Pegawai', ('PKWTT','PKWT'),key=55)
                Job_Level = st.selectbox('Level Pekerjaan', ('VI','V','IV','III','II'),key=56)
                Job_Nature = st.selectbox('Nature Pekerjaan',('Desk', 'Field'),key=57)
            with col3:
                st.write('**_Pekerjaan_**')
                Department = st.selectbox('Departemen Kerja',('WITEL JAKBAR', 'WITEL BANTEN', 'DIG TECH & PLATFORM BUSINESS'),key=58)
                Task_type = st.selectbox('Tipe Pekerjaan',('Clerical', 'Analytical','Physical'),key=59)
                Percent_Salary_Hike = st.number_input(label = 'Percent kenaikan gaji', min_value = 0.0,
                                    max_value = 0.5 ,
                                    value = 0.2,
                                    step = 0.05,key=60)

            st.markdown ('---')
            col4, col5 = st.columns(2)
            with col4:
                Years_at_Company = st.slider(label = 'Lama Bekerja', min_value = 0,
                                    max_value = 60 ,
                                    value = 10,
                                    step = 1,key=61)
                Last_promotion =  st.slider(label = 'Promosi terakhir', min_value = 0,
                                    max_value = 300 ,
                                    value = 1,
                                    step = 1,key=61)
                Last_mutation =  st.slider(label = 'Mutasi terakhir', min_value = 0,
                                    max_value = 100 ,
                                    value = 1,
                                    step = 1,key=62)
                training_duration_2021= st.slider(label = 'Durasi Training', min_value = 0.0,
                                    max_value = 200.0,
                                    value = 30.0,
                                    step = 1.0,key=63)
                THP_current= st.slider(label = 'Gaji saat ini', min_value = 0.0,
                                    max_value = 3.0 ,
                                    value = 2.5,
                                    step = 0.1,key=64)
            with col5:
                LEADERSHIP_2021 = st.slider(label = 'Score Leadership', min_value = 0.0,
                                    max_value = 1.0 ,
                                    value = 0.8,
                                    step = 0.1,key=65)
                BUDAYA_2021 = st.slider(label = 'Score Budaya', min_value = 0.0,
                                    max_value = 1.0 ,
                                    value = 0.8,
                                    step = 0.1,key=66)
                FUNCTIONAL_2021 = st.slider(label = 'Score Functional', min_value = 0.0,
                                    max_value = 1.0 ,
                                    value = 0.8,
                                    step = 0.1,key=67)
            st.markdown ('---')
            features = {
                        'AGE': Age,
                        'GENDER':Gender,
                        'MARITAL_STATUS': Marital_Status,
                        'EDUCATION_GRADE': Education_Grade,
                        'RECRUITMENT_STATUS':Recruitment_Status,
                        'EMPLOYMENT_STATUS': Employment_Status,
                        'YEARS_AT_COMPANY':Years_at_Company,  
                        'DEPARTMENT':Department,
                        'JOB_LEVEL':Job_Level,
                        'JOB_NATURE':Job_Nature,
                        'TASK_TYPE':Task_type,
                        'THP_CURRENT':THP_current,
                        'thp_increase':Percent_Salary_Hike,
                        'LAST_MUTATION':Last_mutation,
                        'LAST_PROMOTION':Last_promotion,
                        'training_duration_2022':training_duration_2022,
                        'leadership_score':LEADERSHIP_2021,
                        'functional_score':FUNCTIONAL_2021,
                        'budaya_score':BUDAYA_2021,
                        }
            features_df  = pd.DataFrame([features])
            features_df = preprocessing_encoding(features_df)
            features_df[features_df.select_dtypes(['object']).columns] = features_df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
            if upl_modelWLAreg is not None:
                klik = st.button("Predict", key=68)
                if klik:
                    #load model
                    pickled_pipeline=pd.read_pickle(upl_modelWLAreg)
                    predicted_fte=pickled_pipeline.predict(features_df)
                    st.success(predicted_fte.to_string()[2:])
    with tab3:
        st.write("# WFP Regression Predictor")
        upl_modelWFPreg = st.file_uploader("Upload file pipelineWFP-regression.pkl",key=69)
        tab8, tab9 = st.tabs(["Batch", "Individual"])
        with tab8:
            uploaded=pd.read_sql_table("data_wfp", engine)
            
            col1,col2=st.columns(2)
            with col1:
                feature_select=st.multiselect("pilih features", uploaded.columns, ["departemen","tahun", "target_revenue","pelanggan", "potensi_revenue","jumlah_produk","karyawan_organik", "pencapaian_revenue","functional_1","functional_2","functional_3","functional_4","functional_5","soft_competency_1","soft_competency_2","soft_competency_3","soft_competency_4","soft_competency_5","jumlah_okr","pencapaian_okr","Productivity"], key=70) 
            with col2:
                target_select=st.multiselect("pilih target", uploaded.columns, "fte", key=71)
                useful_feature=list(feature_select+target_select)
            if uploaded is not None:

                st.warning("WARNING make sure there is no nan value on features value. All empty rows will be filled with mean value")
            if uploaded is not None: 
                # read data
                df = uploaded[useful_feature]
                st.success("Data loaded")
                populate = st.button("Populate",key=72)
            if uploaded is not None and populate is True:
                #charting before
                arr=df[str(target_select[0])]
                fig,ax=plt.subplots()
                ax.hist(arr, bins=20)
                st.pyplot(fig)
                # remove nan in target columns
                df_notnull=df[df[str(target_select[0])].notnull()]
                df = df[df[str(target_select[0])].isnull()]      
                #select columns
                df = df[useful_feature]
                df = df.loc[:,df.columns != 'fte']
                df = df.loc[:,df.columns != 'EMPLOYEE_ID']
                # fill the missing values with mean values
                df.fillna(df.mean(), inplace=True)
                #convert object to category
                df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
                df_notnull[df_notnull.select_dtypes(['object']).columns] = df_notnull.select_dtypes(['object']).apply(lambda x: x.astype('category'))
                st.subheader('Hasil Prediksi')
                #load model
                pickled_pipeline=pd.read_pickle(upl_modelWFPreg)
                # populate fte with predicted value
                predicted_fte=pickled_pipeline.predict(df.loc[:,df.columns != str(target_select[0])])
                df[str(target_select[0])]=predicted_fte
                df=df.append(df_notnull[useful_feature])
                st.dataframe(df)
                arr=df[str(target_select[0])]
                fig,ax=plt.subplots()
                ax.hist(arr, bins=20)
                st.pyplot(fig)
                csv = convert_df(df)
                st.download_button(
                    label="📥 Download data as CSV",
                    data=csv,
                    file_name='predicted_WFP-reg_batch.csv',
                    mime='text/csv',
                    key=73)
        with tab9:
            col1, col2, col3,col4 = st.columns(4)
            with col1:
                st.markdown('**_Target_**')
                departemen = st.selectbox('departemen', ('WITEL BANTEN','WITEL JAKBAR'))
                tahun = st.number_input(label = 'tahun', min_value = 1965,
                        max_value = 2030,
                        value = 2022,
                        step = 1,key=74)
                target_revenue = st.number_input(label = 'target_revenue', min_value = 900000000,
                        max_value = 970000000000,
                        value = 900000000,
                        step = 1000,key=75)
                pelanggan = st.number_input(label = 'pelanggan', min_value = 25000,
                        max_value = 50000000,
                        value = 25000,
                        step = 1000,key=76)
                potensi_revenue = st.number_input(label = 'target_revenue', min_value = 900000000,
                        max_value = 970000000000,
                        value = 900000000,
                        step = 1000,key=77)
                jumlah_produk = st.number_input(label = 'jumlah_produk', min_value = 1,
                        max_value = 50,
                        value = 20,
                        step = 1,key=78)
                karyawan_organik = st.number_input(label = 'karyawan_organik', min_value = 50,
                        max_value = 1000,
                        value = 100,
                        step = 1,key=79)
                pencapaian_revenue = st.number_input(label = 'pencapaian_revenue', min_value = 1000000000,
                        max_value = 950000000000,
                        value = 1000000000,
                        step = 1000,key=80)            

            with col2:
                st.write('**_functional_**')
                functional_1 = st.slider(label = 'functional_1', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=81)    
                functional_2 = st.slider(label = 'functional_2', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=82) 
                functional_3 = st.slider(label = 'functional_3', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=83) 
                functional_4 = st.slider(label = 'functional_4', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=84) 
                functional_5 = st.slider(label = 'functional_5', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=85) 
            with col3:
                st.write('**_soft_competency_**')
                soft_competency_1 = st.slider(label = 'soft_competency_1', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=86) 
                soft_competency_2 = st.slider(label = 'soft_competency_2', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=87) 
                soft_competency_3 = st.slider(label = 'soft_competency_3', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=88) 
                soft_competency_4 = st.slider(label = 'soft_competency_4', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=89) 
                soft_competency_5 = st.slider(label = 'soft_competency_5', min_value = 0.0,
                        max_value = 1.0,
                        value = 0.5,
                        step = 0.1,key=90) 
            st.markdown ('---')
            with col4:
                st.write('**_performance_**')
                jumlah_okr = st.slider(label = 'jumlah_okr', min_value = 1,
                    max_value = 100,
                    value = 50,
                    step = 1, key=91)   
                pencapaian_okr = st.slider(label = 'pencapaian_okr', min_value = 0.1,
                    max_value = 1.0,
                    value = 0.5,
                    step = 0.1, key=92) 
                Productivity = st.slider(label = 'Productivity', min_value =  2000000,
                    max_value =  5000000000,
                    value = 5000000,
                    step = 10000, key=93) 
                st.markdown ('---')   
            features = {
                        'departemen':departemen,
                        'tahun':tahun,
                        'target_revenue':target_revenue,
                        'pelanggan':pelanggan,
                        'potensi_revenue':potensi_revenue,
                        'jumlah_produk':jumlah_produk,
                        'karyawan_organik':karyawan_organik, 
                        'pencapaian_revenue':pencapaian_revenue,
                        'functional_1':functional_1,
                        'functional_2':functional_2,
                        'functional_3':functional_3,
                        'functional_4':functional_4,
                        'functional_5':functional_5,
                        'soft_competency_1':soft_competency_1,
                        'soft_competency_2':soft_competency_2,
                        'soft_competency_3':soft_competency_3,
                        'soft_competency_4':soft_competency_4,
                        'soft_competency_5':soft_competency_5,
                        'jumlah_okr':jumlah_okr,
                        'pencapaian_okr':pencapaian_okr,
                        'Productivity':Productivity,
                        }
            features_df  = pd.DataFrame([features])
            features_df[features_df.select_dtypes(['object']).columns] = features_df.select_dtypes(['object']).apply(lambda x: x.astype('category')) 
            if upl_modelWFPreg is not None:
                klik = st.button("Predict", key=94)
                if klik:
                    st.subheader('Hasil Prediksi')
                    #load model
                    pickled_pipeline=pd.read_pickle(upl_modelWFPreg)
                    predicted_fte=pickled_pipeline.predict(features_df)
                    st.success(predicted_fte.to_string()[2:])

        
        
def page10():
    tab1,tab2 = st.tabs(["Average Competencies each Departement","Efective Hours for Each Position"])
    with tab1:
        uploaded_cb=pd.read_sql_table("competencies_behavior", engine)
        uploaded_dm=pd.read_sql_table("data_model", engine)
        
        if uploaded_cb is not None and uploaded_dm is not None:
            df_employee = uploaded_dm
            df_competence= uploaded_cb
            #get list of departments
            departments=df_employee["DEPARTMENT"].unique()
            Amanah=df_employee["DEPARTMENT"].unique()
            Kompeten=df_employee["DEPARTMENT"].unique()
            Harmonis=df_employee["DEPARTMENT"].unique()
            Loyal=df_employee["DEPARTMENT"].unique()
            Adaptif=df_employee["DEPARTMENT"].unique()
            Kolaboratif=df_employee["DEPARTMENT"].unique()

            #define temp emp
            emp=pd.DataFrame()
            #collect all employee in each dept
            emp=pd.DataFrame(list(df_employee.groupby('DEPARTMENT')['EMPLOYEE_ID'].apply(list).values)).T
            #filter only for BUDAYA
            df_competence = df_competence[(df_competence["GROUP_COMPETENCY"] == "BUDAYA") ]    
            #filter based on list of each employee in each dept
            for i in range(emp.shape[1]):    
                Amanah[i]=df_competence[df_competence['EMPLOYEE_ID'].isin(emp[i])]
                Kompeten[i]=df_competence[df_competence['EMPLOYEE_ID'].isin(emp[i])]
                Harmonis[i]=df_competence[df_competence['EMPLOYEE_ID'].isin(emp[i])]
                Loyal[i]=df_competence[df_competence['EMPLOYEE_ID'].isin(emp[i])]
                Adaptif[i]=df_competence[df_competence['EMPLOYEE_ID'].isin(emp[i])]
                Kolaboratif[i]=df_competence[df_competence['EMPLOYEE_ID'].isin(emp[i])]
            
            #filter based on AKHLAK (Amanah,Kompeten,Harmonis,Loyal,Adaptif,Kolaboratif)
            #Amanah
            for i in range(len(departments)):
                Amanah[i]=Amanah[i][(Amanah[i]["COMPETENCY"] == "Amanah") ]
                Amanah[i]=Amanah[i]["GAP_SCORE"].mean()
            #Kompeten
                Kompeten[i]=Kompeten[i][(Kompeten[i]["COMPETENCY"] == "Kompeten") ]
                Kompeten[i]=Kompeten[i]["GAP_SCORE"].mean()
            #Harmonis
                Harmonis[i]=Harmonis[i][(Harmonis[i]["COMPETENCY"] == "Harmonis") ]
                Harmonis[i]=Harmonis[i]["GAP_SCORE"].mean() 
            #Loyal
                Loyal[i]=Loyal[i][(Loyal[i]["COMPETENCY"] == "Loyal") ]
                Loyal[i]=Loyal[i]["GAP_SCORE"].mean()
            #Adaptif
                Adaptif[i]=Adaptif[i][(Adaptif[i]["COMPETENCY"] == "Adaptif") ]
                Adaptif[i]=Adaptif[i]["GAP_SCORE"].mean()
            #Kolaboratif
                Kolaboratif[i]=Kolaboratif[i][(Kolaboratif[i]["COMPETENCY"] == "Kolaboratif") ]
                Kolaboratif[i]=Kolaboratif[i]["GAP_SCORE"].mean()  

            #build df
            data=pd.DataFrame([Amanah.T,Kompeten.T, Loyal.T, Adaptif.T, Kolaboratif.T])
            data.columns=departments
            data.index=("Amanah","Kompeten", "Loyal", "Adaptif", "Kolaboratif")
            st.dataframe(data)
                

                                                              
            
            
    with tab2:
        uploaded_file=pd.read_sql_table("data_model", engine)
        diarium= uploaded_file
        if diarium is not None:
            #read data
            df = diarium
            pivot = pd.pivot_table(data=df,index=['position'],values=['jam_efektif'],aggfunc=np.mean)
            st.dataframe(pivot)
            

            
page_names_to_funcs = {
    "Main Page": main_page,
    "Preprocesing Diarium":page10,
    "Modeler": page2,
    "Check WLA Gap per unit": page5,
    "Predictor":page6,
    }

selected_page = st.sidebar.radio("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
