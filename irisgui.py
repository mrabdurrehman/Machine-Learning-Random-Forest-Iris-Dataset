import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")
target_names = iris.target_names
scatter_data = X.copy()
scatter_data["target"] = y

# Sidebar Model Selector
st.sidebar.header("🔧 Model & Features")
model_choice = st.sidebar.selectbox("Choose Classifier Model", ["Random Forest", "Support Vector Machine", "KNN"])
if model_choice == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_choice == "Support Vector Machine":
    model = SVC(probability=True)
elif model_choice == "KNN":
    model = KNeighborsClassifier()

# Train the selected model
model.fit(X, y)

# Input sliders
st.sidebar.subheader("🧪 Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()), float(X["sepal length (cm)"].mean()))
sepal_width  = st.sidebar.slider("Sepal Width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()), float(X["sepal width (cm)"].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()), float(X["petal length (cm)"].mean()))
petal_width  = st.sidebar.slider("Petal Width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()), float(X["petal width (cm)"].mean()))

# Predict user input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

st.title("🌸 Iris Flower ML Classifier")
st.subheader("🌿 Prediction - Assignment # 8 by Abdur Rehman")
st.success(f"Predicted species using **{model_choice}**: **{predicted_species.capitalize()}**")

# Dataset preview
if st.checkbox("📄 Show Dataset Sample"):
    st.dataframe(scatter_data.sample(10))

# Show model performance & confusion matrix
if st.checkbox("📊 Show Model Performance"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    selected_model = model.__class__()  # create fresh instance
    selected_model.fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.write(f"**Accuracy ({model_choice}):** {acc:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# 2D Scatter plot
if st.checkbox("📈 Show 2D Petal Plot"):
    fig2d, ax = plt.subplots()
    sns.scatterplot(data=scatter_data, x="petal length (cm)", y="petal width (cm)", hue="target", palette="Set1", ax=ax)
    ax.scatter(petal_length, petal_width, color="red", s=120, label="Your Input", marker='*')
    ax.set_title("2D: Petal Length vs Width")
    ax.legend()
    st.pyplot(fig2d)

# Pairplot (like Kaggle)
if st.checkbox("📌 Show Pairplot (Full Feature Comparison)"):
    pairplot_fig = sns.pairplot(scatter_data, hue="target", palette="Set1")
    st.pyplot(pairplot_fig)

# 3D Scatter plot
if st.checkbox("🌐 Show 3D Scatter Plot"):
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue']
    for i in range(3):
        subset = scatter_data[scatter_data['target'] == i]
        ax3d.scatter(subset["sepal length (cm)"], subset["petal length (cm)"], subset["petal width (cm)"], color=colors[i], label=target_names[i], alpha=0.6)
    ax3d.scatter(sepal_length, petal_length, petal_width, color='gold', s=200, label="Your Input", marker='*')
    ax3d.set_xlabel("Sepal Length")
    ax3d.set_ylabel("Petal Length")
    ax3d.set_zlabel("Petal Width")
    ax3d.set_title("3D View: Sepal vs Petal")
    ax3d.legend()
    st.pyplot(fig3d)
