# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from tkinter import messagebox
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk
    
df = pd.read_csv('Data.csv')
df=df.replace('?',np.nan)
X =df.iloc[:, :-1].values
Y = df.iloc[:,-1].values    
symptoms=np.array(df.columns)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
 #Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test,y_pred)
conf = confusion_matrix(y_test,y_pred)

#Input
#description = np.array(input("Describe your symptoms: ").split())
def funct():
    description=[str(menu.get()),str(menu1.get()),str(menu2.get())]
    print(description)
    s=np.zeros(132)
    i=0
    for x in symptoms:
        if x in  description:      
            s[i]=1
        i+=1

    B = s.reshape(1,132)
    messagebox.showinfo("Disease Prediction System","The patient might have "+classifier.predict(B) + " and the accuracy of the Algorithm is "+ str(acc))

                
m = tk.Tk()
m.geometry("400x300")
label1 = tk.Label(text="Symptom 1")
label1.place(x=5,y=10)
menu1= tk.StringVar()
menu1.set("Select the Symptom")
drop1= tk.OptionMenu(m, menu1,'itching','skin_rash', 'nodal_skin_eruptions' ,'continuous_sneezing',
 'shivering' 'chills', 'joint_pain', 'stomach_pain', 'acidity',
 'ulcers_on_tongue' ,'muscle_wasting' ,'vomiting' ,'burning_micturition',
 'spotting_ urination' ,'fatigue', 'weight_gain' ,'anxiety',
 'cold_hands_and_feets' ,'mood_swings' ,'weight_loss', 'restlessness',
 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
 'high_fever' ,'sunken_eyes', 'breathlessness', 'sweating' ,'dehydration',
 'indigestion', 'headache' ,'yellowish_skin', 'dark_urine', 'nausea',
 'loss_of_appetite','pain_behind_the_eyes', 'back_pain', 'constipation',
 'abdominal_pain' ,'diarrhoea', 'mild_fever', 'yellow_urine',
 'yellowing_of_eyes' ,'acute_liver_failure', 'fluid_overload',
 'swelling_of_stomach' ,'swelled_lymph_nodes', 'malaise',
 'blurred_and_distorted_vision', 'phlegm' 'throat_irritation',
 'redness_of_eyes' ,'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
 'weakness_in_limbs' ,'fast_heart_rate', 'pain_during_bowel_movements',
 'pain_in_anal_region', 'bloody_stool' ,'irritation_in_anus', 'neck_pain',
 'dizziness' 'cramps', 'bruising' 'obesity' ,'swollen_legs',
 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
 'brittle_nails' ,'swollen_extremeties', 'excessive_hunger',
 'extra_marital_contacts' ,'drying_and_tingling_lips', 'slurred_speech',
 'knee_pain', 'hip_joint_pain', 'muscle_weakness' ,'stiff_neck',
 'swelling_joints' ,'movement_stiffness', 'spinning_movements',
 'loss_of_balance' ,'unsteadiness' ,'weakness_of_one_body_side',
 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
 'continuous_feel_of_urine', 'passage_of_gases' ,'internal_itching',
 'toxic_look_(typhos)', 'depression' 'irritability', 'muscle_pain',
 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
 'increased_appetite', 'polyuria' ,'family_history' ,'mucoid_sputum',
 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
 'stomach_bleeding', 'distention_of_abdomen',
 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
 'prominent_veins_on_calf', 'palpitations' ,'painful_walking',
 'pus_filled_pimples' ,'blackheads' ,'scurring', 'skin_peeling',
 'silver_like_dusting' ,'small_dents_in_nails', 'inflammatory_nails',
 'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis')
drop1.pack()
drop1.place(x=100,y=10)
label2 = tk.Label(text="Symptom 2")
label2.place(x=5,y=55)
menu= tk.StringVar()
menu.set("Select the Symptom")
drop= tk.OptionMenu(m, menu,'itching','skin_rash', 'nodal_skin_eruptions' ,'continuous_sneezing',
 'shivering' 'chills', 'joint_pain', 'stomach_pain', 'acidity',
 'ulcers_on_tongue' ,'muscle_wasting' ,'vomiting' ,'burning_micturition',
 'spotting_ urination' ,'fatigue', 'weight_gain' ,'anxiety',
 'cold_hands_and_feets' ,'mood_swings' ,'weight_loss', 'restlessness',
 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
 'high_fever' ,'sunken_eyes', 'breathlessness', 'sweating' ,'dehydration',
 'indigestion', 'headache' ,'yellowish_skin', 'dark_urine', 'nausea',
 'loss_of_appetite','pain_behind_the_eyes', 'back_pain', 'constipation',
 'abdominal_pain' ,'diarrhoea', 'mild_fever', 'yellow_urine',
 'yellowing_of_eyes' ,'acute_liver_failure', 'fluid_overload',
 'swelling_of_stomach' ,'swelled_lymph_nodes', 'malaise',
 'blurred_and_distorted_vision', 'phlegm' 'throat_irritation',
 'redness_of_eyes' ,'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
 'weakness_in_limbs' ,'fast_heart_rate', 'pain_during_bowel_movements',
 'pain_in_anal_region', 'bloody_stool' ,'irritation_in_anus', 'neck_pain',
 'dizziness' 'cramps', 'bruising' 'obesity' ,'swollen_legs',
 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
 'brittle_nails' ,'swollen_extremeties', 'excessive_hunger',
 'extra_marital_contacts' ,'drying_and_tingling_lips', 'slurred_speech',
 'knee_pain', 'hip_joint_pain', 'muscle_weakness' ,'stiff_neck',
 'swelling_joints' ,'movement_stiffness', 'spinning_movements',
 'loss_of_balance' ,'unsteadiness' ,'weakness_of_one_body_side',
 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
 'continuous_feel_of_urine', 'passage_of_gases' ,'internal_itching',
 'toxic_look_(typhos)', 'depression' 'irritability', 'muscle_pain',
 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
 'increased_appetite', 'polyuria' ,'family_history' ,'mucoid_sputum',
 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
 'stomach_bleeding', 'distention_of_abdomen',
 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
 'prominent_veins_on_calf', 'palpitations' ,'painful_walking',
 'pus_filled_pimples' ,'blackheads' ,'scurring', 'skin_peeling',
 'silver_like_dusting' ,'small_dents_in_nails', 'inflammatory_nails',
 'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis')
drop.pack()
drop.place(x=100,y=50)
label2 = tk.Label(text="Symptom 3")
label2.place(x=5,y=105)
menu2= tk.StringVar()
menu2.set("Select the Symptom")
drop2= tk.OptionMenu(m, menu2,'itching','skin_rash', 'nodal_skin_eruptions' ,'continuous_sneezing',
 'shivering' 'chills', 'joint_pain', 'stomach_pain', 'acidity',
 'ulcers_on_tongue' ,'muscle_wasting' ,'vomiting' ,'burning_micturition',
 'spotting_ urination' ,'fatigue', 'weight_gain' ,'anxiety',
 'cold_hands_and_feets' ,'mood_swings' ,'weight_loss', 'restlessness',
 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
 'high_fever' ,'sunken_eyes', 'breathlessness', 'sweating' ,'dehydration',
 'indigestion', 'headache' ,'yellowish_skin', 'dark_urine', 'nausea',
 'loss_of_appetite','pain_behind_the_eyes', 'back_pain', 'constipation',
 'abdominal_pain' ,'diarrhoea', 'mild_fever', 'yellow_urine',
 'yellowing_of_eyes' ,'acute_liver_failure', 'fluid_overload',
 'swelling_of_stomach' ,'swelled_lymph_nodes', 'malaise',
 'blurred_and_distorted_vision', 'phlegm' 'throat_irritation',
 'redness_of_eyes' ,'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
 'weakness_in_limbs' ,'fast_heart_rate', 'pain_during_bowel_movements',
 'pain_in_anal_region', 'bloody_stool' ,'irritation_in_anus', 'neck_pain',
 'dizziness' 'cramps', 'bruising' 'obesity' ,'swollen_legs',
 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
 'brittle_nails' ,'swollen_extremeties', 'excessive_hunger',
 'extra_marital_contacts' ,'drying_and_tingling_lips', 'slurred_speech',
 'knee_pain', 'hip_joint_pain', 'muscle_weakness' ,'stiff_neck',
 'swelling_joints' ,'movement_stiffness', 'spinning_movements',
 'loss_of_balance' ,'unsteadiness' ,'weakness_of_one_body_side',
 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
 'continuous_feel_of_urine', 'passage_of_gases' ,'internal_itching',
 'toxic_look_(typhos)', 'depression' 'irritability', 'muscle_pain',
 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
 'increased_appetite', 'polyuria' ,'family_history' ,'mucoid_sputum',
 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
 'stomach_bleeding', 'distention_of_abdomen',
 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
 'prominent_veins_on_calf', 'palpitations' ,'painful_walking',
 'pus_filled_pimples' ,'blackheads' ,'scurring', 'skin_peeling',
 'silver_like_dusting' ,'small_dents_in_nails', 'inflammatory_nails',
 'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis')
drop2.pack()
drop2.place(x=100,y=100)
submit = tk.Button(text="Submit",command=funct)
submit.place(x=80,y=150)


m.title("Disease Prediction")
m.mainloop()       