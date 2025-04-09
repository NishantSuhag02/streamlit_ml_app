import streamlit as st
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame( iris.data, columns=iris.feature_names )
    df['species'] = iris.target
    return df, iris.target_names

df,target_name = load_data()

model_choice = st.sidebar.selectbox(
    "Choose a Classification Model",
    ("Random Forest", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)")
)


# Initialize and train model based on user choice
if model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "Support Vector Machine (SVM)":
    model = SVC(probability=True)  
elif model_choice == "K-Nearest Neighbors (KNN)":
    model = KNeighborsClassifier()

x = df.iloc[:, :-1]
y = df['species']

# Training Model
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
test_accuracy = model.score(X_test, y_test)


# Creating a Slider using Streamlit
st.sidebar.title('Input Features')

sepal_length = st.sidebar.slider('Sepal Length', float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider('Sepal Width', float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider('Petal Length', float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider('Petal Width', float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))

# Input Features
input_data=[[sepal_length,sepal_width,petal_length,petal_width]]

# Prediction
prediction = model.predict(input_data)
predicted_species = target_name[prediction[0]]

models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

results = []

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({
        "Model": name,
        "Accuracy": f"{acc*100:.2f}%"
    })

results_df = pd.DataFrame(results)

# Printing Predicted species
st.title('Leaves Prediction Web App \n')
st.write(f" Prediction Model used: **{model_choice}**")
st.write(f'The predicted species is : **{predicted_species.upper()}**')
st.write(f"Test Accuracy: **{test_accuracy*100:.2f}%**")
st.markdown(f"""Entered leave dimensions are :-  
            - **Sepal Length** : {sepal_length}  
            - **Sepal Width** : {sepal_width}  
            - **Petal Length** : {petal_length}  
            - **Petal Width** : {petal_width}""")

results_df.index = results_df.index + 1 
st.subheader("Model Comparison")
st.dataframe(results_df)
