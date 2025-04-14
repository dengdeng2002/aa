import streamlit as st
import shap
import joblib
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm


def main():
    best_model = joblib.load('./lgbm.pkl')

    class Subject:
        def __init__(self, Race, BMI, Age, Hypertension, Diabetes, Educationallevel, C140, C40, CVD):
            self.Race = Race
            self.BMI = BMI
            self.Age = Age
            self.Hypertension = Hypertension
            self.Diabetes = Diabetes
            self.Educationallevel = Educationallevel
            self.C140 = C140
            self.C40 = C40
            self.CVD = CVD

        def make_predict(self):
            subject_data = {
                "Race": [self.Race],
                "BMI": [self.BMI],
                "Age": [self.Age],
                "Hypertension": [self.Hypertension],
                "Diabetes": [self.Diabetes],
                "Educationallevel": [self.Educationallevel],
                "C140": [self.C140],
                "C40": [self.C40],
                "CVD": [self.CVD]
            }

            # Create a DataFrame
            df_subject = pd.DataFrame(subject_data)

            # Make the prediction
            prediction = best_model.predict_proba(df_subject)[:, 1]
            adjusted_prediction = np.round(prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>The model predicts the prevalence of preserved ratio impaired spirometry is {adjusted_prediction} %</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            explainer = shap.Explainer(best_model)
            shap_values = explainer.shap_values(df_subject)
            # 力图
            shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
            # 瀑布图
            # ex = shap.Explanation(shap_values[1][0, :], explainer.expected_value[1], df_subject.iloc[0, :])
            # shap.waterfall_plot(ex)
            st.pyplot(plt.gcf())

    st.set_page_config(page_title='the prevalence of preserved ratio impaired spirometry')
    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>Predicting the prevalence of preserved ratio impaired spirometry</h1>
                    <p class='intro'></p>
                </div>
                """, unsafe_allow_html=True)
    Race = st.selectbox("Gender (Mexican American = 1, Other Hispanic = 2, Non-Hispanic White = 3, Non-Hispanic Black = 4, Other race = 5)", [1,2,3,4,5], index=3)
    BMI = st.number_input("BMI (kg/m^2)", value=24.51)
    Age = st.number_input("Age (years)", value=48)
    Hypertension = st.selectbox("Hypertension (Yes = 1, No = 2)", [1, 2], index=1)
    Diabetes = st.selectbox("Diabetes (Yes = 1, No = 2, Borderline = 3)", [1, 2, 3], index=2)
    Educationallevel = st.selectbox("Educational level (Less than high school = 1, High school = 2, More than high school = 3)", [1, 2, 3], index=2)
    C140 = st.number_input("dietary C14:0 fatty acid intake (g/day)", value=8.087)
    C40 =st.number_input("dietary C4:0 fatty acid intake (g/day)", value=2.411)
    CVD = st.selectbox("Gender (No = 0, Yes = 1)", [0, 1], index=0)

    if st.button(label="Submit"):
        user = Subject(Race, BMI, Age, Hypertension, Diabetes, Educationallevel, C140, C40, CVD)
        user.make_predict()

main()
