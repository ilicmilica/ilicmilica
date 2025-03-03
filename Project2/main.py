import pandas as pd
import os
from nltk.stem import PorterStemmer
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


#  path to the folder that contains the CSV files
data_folder = "Project2/dataset"

df = pd.read_csv(os.path.join(data_folder, "hospital.csv"))

# identify the columns that contain missing values and the number of missing values in each column
missing_values = df.isnull().sum()

# drop the rows that contain missing values if there are any
if missing_values.sum() > 0:
    df = df.dropna()
print("Missing values dropped")

# Remove duplicates if there are any
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()
print("Duplicates removed")

# Save the cleaned data to a new CSV file
df.to_csv(os.path.join(data_folder, "hospital_cleaned.csv"), index=False)
print("Cleaned data saved to hospital_cleaned.csv")

########## TEXT preprocessing ##########

# tokenization - split the text into words
df['Feedback'] = df['Feedback'].apply(lambda x: x.split())

# remove punctuation
df['Feedback'] = df['Feedback'].apply(lambda x: [word.translate(str.maketrans('', '', string.punctuation)) for word in x])

# lowercasing - convert all words to lowercase
df['Feedback'] = df['Feedback'].apply(lambda x: [word.lower() for word in x])

# remove stopwords
from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))
#df['Feedback'] = df['Feedback'].apply(lambda x: [word for word in x if word not in stop_words])

# stemming
stemmer = PorterStemmer()
df['Feedback'] = df['Feedback'].apply(lambda x: [stemmer.stem(word) for word in x])

# save the preprocessed text to a new CSV file
df.to_csv(os.path.join(data_folder, "hospital_cleaned_text.csv"), index=False)

print("Text preprocessing completed")
#print(df.head())

# build an LSTM model and train it on the preprocessed text data
# load the preprocessed text data
df = pd.read_csv(os.path.join(data_folder, "hospital_cleaned_text.csv"))

# convert the text data to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Feedback'])
X = tokenizer.texts_to_sequences(df['Feedback'])
X = pad_sequences(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Sentiment Label'], test_size=0.2, random_state=42)

# build the BI-LSTM model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 50))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# save the trained model
model.save(os.path.join(data_folder, "model.h5"))
print("Model saved as model.h5")

# predict the ratings for the feedback
predictions = model.predict(X_test)
predictions = [1 if p > 0.5 else 0 for p in predictions]

# evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# auc score
auc = roc_auc_score(y_test, predictions)
print("AUC:", auc)

# visualize the results
# confusion matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # random baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
