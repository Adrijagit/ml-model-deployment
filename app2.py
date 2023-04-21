
import streamlit as st
import pandas as pd
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


from sklearn.metrics import precision_score, recall_score

def main():

    st.title("classifier Model Webapp")
    st.sidebar.title("This is the sidebar")
    st.sidebar.markdown("Let’s start with binary classification!!")
if __name__ == '__main__':
    main()\
#@st.cache(persist=True)
data= pd.read_csv('C:/Users/KALYAN KUMAR GUHA/Desktop/ml-deploy/mushrooms.csv')
label= LabelEncoder()
for i in data.columns:
    data[i] = label.fit_transform(data[i])



if st.sidebar.checkbox("Display data", False):

    st.subheader("Show Mushroom dataset")
    st.write(data)


#@st.cache(persist=True)

y = data['class']
x = data.drop(columns='class',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
class_names = ["edible", "poisonous"]
st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)"))
if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
    gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")
metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
if st.sidebar.button("Classify", key="classify"):
    st.subheader("Support Vector Machine (SVM) results")
    #modelsvm = SVC(C='C', kernel='kernel', gamma='gamma')
    modelsvm = SVC(kernel='linear')
    modelsvm.fit(x_train, y_train)
    accuracy = modelsvm.score(x_test, y_test)
    y_pred = modelsvm.predict(x_test)
    st.write("Accuracy: ", accuracy.round(2))
    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #plot_metrics(metrics)

