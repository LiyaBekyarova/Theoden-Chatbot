import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
import os

print("=" * 60)
print("CHATBOT INTENT CLASSIFICATION - MODEL COMPARISON")
print("=" * 60)

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X = []
y = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    y.append(tags.index(tag))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Dataset Info:")
print(f"   Total samples: {len(X)}")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Number of intents: {len(tags)}")
print(f"   Vocabulary size: {len(all_words)}")
print(f"   Intents: {', '.join(tags)}")

results = []
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION (Linear Classifier)")
print("=" * 60)
try:
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f" Accuracy: {lr_accuracy * 100:.2f}%")
    results.append(("Logistic Regression", lr_accuracy))
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("DECISION TREE CLASSIFIER")
print("=" * 60)
try:
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    print(f"Accuracy: {dt_accuracy * 100:.2f}%")
    results.append(("Decision Tree", dt_accuracy))
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFIER (Ensemble)")
print("=" * 60)
try:
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f" Accuracy: {rf_accuracy * 100:.2f}%")
    results.append(("Random Forest", rf_accuracy))
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)
try:
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"Accuracy: {svm_accuracy * 100:.2f}%")
    results.append(("SVM (RBF kernel)", svm_accuracy))
except Exception as e:
    print(f" Error: {e}")

print("\n" + "=" * 60)
print("NAIVE BAYES CLASSIFIER")
print("=" * 60)
try:
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"Accuracy: {nb_accuracy * 100:.2f}%")
    results.append(("Naive Bayes", nb_accuracy))
except Exception as e:
    print(f" Error: {e}")

print("\n" + "=" * 60)
print("6 NEURAL NETWORK")
print("=" * 60)

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class TrainDataset(Dataset):
        def __init__(self, X, y):
            self.x_data = torch.from_numpy(X).float()
            self.y_data = torch.from_numpy(y).long()
        def __len__(self): return len(self.x_data)
        def __getitem__(self, idx): return self.x_data[idx], self.y_data[idx]

    train_loader = DataLoader(TrainDataset(X_train, y_train), batch_size=8, shuffle=True)

    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = len(tags)

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2

    print("Training with strong regularization...")
    best_loss = float('inf')
    patience = 30
    wait = 0

    for epoch in range(400):
        model.train()
        epoch_loss = 0
        for words, labels in train_loader:
            words, labels = words.to(device), labels.to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader) 
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f"   EARLY STOPPING at epoch {epoch+1} | Best loss: {best_loss:.6f}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1:3d} | Avg Loss: {epoch_loss:.6f} | Best: {best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(X_test).float().to(device))
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_test, predicted.cpu().numpy())
        nn_pred = predicted.cpu().numpy()
    
    print(f"TEST ACCURACY: {acc * 100:.2f}%")
    results.append(("Neural Network (PyTorch)", acc))

except Exception as e:
    print(f"Error: {e}")
    results.append(("Neural Network (PyTorch)", 0.0))

print("\n" + "=" * 60)
print(" FINAL RESULTS - CLEAN SUMMARY")
print("=" * 60)

results.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<4} {'Model':<28} {'Accuracy'}")
print("-" * 50)
for i, (name, acc) in enumerate(results, 1):
    medal = "1st" if i == 1 else "2nd" if i == 2 else "3rd" if i == 3 else "   "
    print(f"{medal} {i:<3} {name:<28} {acc*100:6.2f}%")

print("\n" + "=" * 60)
print(f" WINNER → {results[0][0]} with {results[0][1]*100:.2f}% accuracy!")
print("=" * 60)

winner_name = results[0][0]
y_pred_winner = {
    "Logistic Regression": lr_pred,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred,
    "SVM (RBF kernel)": svm_pred,
    "Naive Bayes": nn_pred,
    "Neural Network (PyTorch)":  nn_pred
}[winner_name]

print(f"\nClassification Report — {winner_name}:")
print(classification_report(
    y_test, y_pred_winner,
    target_names=tags,
    zero_division=0         
))

predictions = {
    "Logistic Regression": lr_pred.tolist() if 'lr_pred' in locals() else None,
    "Decision Tree": dt_pred.tolist() if 'dt_pred' in locals() else None,
    "Random Forest": rf_pred.tolist() if 'rf_pred' in locals() else None,
    "SVM (RBF kernel)": svm_pred.tolist() if 'svm_pred' in locals() else None,
    "Naive Bayes": nb_pred.tolist() if 'nb_pred' in locals() else None,
    "Neural Network (PyTorch)": nn_pred.tolist() if 'nn_pred' in locals() else None
}

final_results = []
for name, acc in results:
    final_results.append({
        "model": name,
        "accuracy": round(acc * 100, 2)
    })

final_results = sorted(final_results, key=lambda x: x['accuracy'], reverse=True)
output_data = {
    "dataset_info": {
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_intents": len(tags),
        "vocabulary_size": len(all_words),
        "intents": tags
    },
    "model_results": final_results,
    "winner_model": final_results[0]["model"],
    "winner_accuracy": final_results[0]["accuracy"],
    "test_labels": y_test.tolist(),
    "predictions": predictions
}

output_path = "model_comparison_results.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\nРезултатите са записани в: {output_path}")