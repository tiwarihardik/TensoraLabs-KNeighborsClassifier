import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import *
import matplotlib.pyplot as plt

st.title('TensoraLabs - KNN Classifier')
st.write('Where ideas are built.')

if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'target_categories' not in st.session_state:
    st.session_state.target_categories = None


file = st.file_uploader("Upload a CSV File", type='.csv')
if file:
    df = pd.read_csv(file).dropna()
    st.write("Data Preview:", df.head())

    target = st.selectbox('Select column to predict:', df.columns)
    features = st.multiselect('Select features to use:', df.columns.drop(target))

    if st.button('Train Model') and features:
        X = df[features]
        y = df[target]

        X_enc = pd.get_dummies(X)
        st.session_state.X_columns = X_enc.columns.tolist()
        
        if y.dtype == 'object':
            y_encoded, target_categories = pd.factorize(y)
            y = y_encoded
            st.session_state.target_categories = target_categories
        else:
            st.session_state.target_categories = None

        X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.3, random_state=42)

        param_grid = {'n_neighbors': range(1, 21)}
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid.fit(X_train, y_train)

        best_k = grid.best_params_['n_neighbors']
        model = KNeighborsClassifier(n_neighbors=best_k)
        model.fit(X_train, y_train)

        st.session_state.model = model

        predictions_ = model.predict(X_test)

        acc = f1_score(y_test, predictions_, average='weighted')
        st.success(f"Model Accuracy: {acc:.4f} ")

        if X_enc.shape[1] == 1:
            plt.figure(figsize=(8, 5))
            plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual', alpha=0.6)
            plt.scatter(X_test.iloc[:, 0], model.predict(X_test), color='red', label='Predicted', alpha=0.6)
            plt.xlabel(features[0])
            plt.ylabel(target)
            plt.title("KNN Predictions")
            plt.legend()
            st.pyplot(plt)

if st.session_state.model:
    st.header("🔮 Make Predictions")

    user_input = {}
    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            user_input[feature] = st.number_input(f"{feature}:")
        else:
            user_input[feature] = st.selectbox(f"{feature}:", df[feature].unique())

    if st.button("📍 Predict"):
        input_df = pd.DataFrame([user_input])
        input_enc = pd.get_dummies(input_df).reindex(columns=st.session_state.X_columns, fill_value=0)
        pred = st.session_state.model.predict(input_enc)[0]

        if st.session_state.target_categories is not None:
            label = st.session_state.target_categories[pred]
        else:
            label = pred

        st.success(f"{target}: {label}")
