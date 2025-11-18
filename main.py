import pandas as pd
from classifier import build_keyword_dfa, classify_message
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("dataset.csv")

keywords = ["win", "free", "congratulations"]

dfas = [build_keyword_dfa(k) for k in keywords]

df["predicted"] = df["v2"].apply(lambda x: classify_message(dfas, x))

accuracy = accuracy_score(df["v1"], df["predicted"])
precision = precision_score(df["v1"], df["predicted"], pos_label="spam")
recall = recall_score(df["v1"], df["predicted"], pos_label="spam")
f1 = f1_score(df["v1"], df["predicted"], pos_label="spam")

print("------ DFA SPAM DETECTOR RESULTS ------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
