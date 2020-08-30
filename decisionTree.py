
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt

#satır isimleri
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)

#veri seti okunuyor

pima.head()

#öznıtelikler
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

#öznitelikler x'e atanır
X = pima[feature_cols] # Features
#sınıflandırıcı y'ye atanır
y = pima.label # Target variable

#%70 eğitim %30 test verisi oluşturulur.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#decision tree sınıflandırması yapılır
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#doğruluk oranı hesaplanıp yazdırılır.
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#hata matrisi oluşturulur.
cm = confusion_matrix(y_test, y_pred )
#hata matrisi yazdırılır.
print("Confusion matrix")
print(cm)

#hata matrisi grafiği çizdirilir.
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()






