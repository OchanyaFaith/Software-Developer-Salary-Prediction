import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('lgm_model.pkl', 'rb' ) as file:
        data = pickle.load(file)
    return data

data = load_model()
lgm_loaded = data["model"]
p_encode = data["p_encode"]
r_encode = data["r_encode"]
e_encode= data["e_encode"]
c_encode = data["c_encode"]


def predict_salary():
    st.title("Software Developer Salary Prediction")

    countries= ("United States of America",                                                 
                    "Germany",                                                
                    "United Kingdom of Great Britain and Northern Ireland",
                    "Canada",                                                  
                    "India",                                                    
                    "France",                                                   
                    "Netherlands",                                              
                    "Australia",                                                 
                    "Brazil",                                                    
                    "Spain",                                                     
                    "Sweden",                                                    
                    "Italy",                                                     
                    "Switzerland",                                               
                    "Denmark",                                                   
                    "Norway",                                                    
                    "Israel",
                    "other" ,  ) 
    education  = ("Bachelor’s degree", "Less than a Bachelors", "Master’s degree",
       "Post Grad",)         
   
    profession = ("I am a developer by profession", "I am not primarily a developer,but I write code sometimes as part of my work/studies",)
    remotework = ("Remote", "Hybrid (some remote, some in-person)", "In-person",)


    Country = st.selectbox("Country", countries)
    EducationLevel = st.selectbox("Education Level", education)
    #Profession = st.selectbox("Profession", profession)
    RemoteWork = st.selectbox("RemoteWork", remotework)
    YearsCode = st.slider("Years of Coding Experience", 0, 50, 3)
    YearsCodePro = st.slider("Years of Professional Coding Experience", 0, 50, 2)

    ok = st.button("Predict Salary")
    if ok:
        X = np.array([[Country,EducationLevel,RemoteWork,YearsCode,YearsCodePro]])
        X[:, 0] = c_encode.transform(X[:, 0])
        X[:, 1] = e_encode.transform(X[:, 1])
       # X[:, 2] = p_encode.transform(X[:, 2])
        X[:, 2] = r_encode.transform(X[:, 2])
        X = X.astype(float)
        st.write(X)
        salary = lgm_loaded.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")


predict_salary()

