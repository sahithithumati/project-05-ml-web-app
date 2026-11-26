import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

st.title("🌸 Iris Flower Prediction App")
st.write("Enter flower measurements to predict the flower type.")

iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 2.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

if st.button("Predict Flower"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    flower_name = iris.target_names[prediction[0]]
    st.success(f"Predicted Flower: {flower_name}")
