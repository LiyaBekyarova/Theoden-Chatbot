# compare_models.py
import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import torch
from model import NeuralNet

print("=" * 60)
print("CHATBOT INTENT CLASSIFICATION - MODEL COMPARISON")
print("=" * 60)

# Load and prepare data
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

# Create training data
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

# ==================== MODEL 1: Logistic Regression ====================
print("\n" + "=" * 60)
print("1️⃣  LOGISTIC REGRESSION (Linear Classifier)")
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

# ==================== MODEL 2: Decision Tree ====================
print("\n" + "=" * 60)
print("2️⃣  DECISION TREE CLASSIFIER")
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

# ==================== MODEL 3: Random Forest ====================
print("\n" + "=" * 60)
print("3️⃣  RANDOM FOREST CLASSIFIER (Ensemble)")
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

# ==================== MODEL 4: Support Vector Machine ====================
print("\n" + "=" * 60)
print("4️⃣  SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)
try:
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"Accuracy: {svm_accuracy * 100:.2f}%")
    results.append(("SVM (RBF kernel)", svm_accuracy))
except Exception as e:
    print(f"❌ Error: {e}")

# ==================== MODEL 5: Naive Bayes ====================
print("\n" + "=" * 60)
print("5️⃣  NAIVE BAYES CLASSIFIER")
print("=" * 60)
try:
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"Accuracy: {nb_accuracy * 100:.2f}%")
    results.append(("Naive Bayes", nb_accuracy))
except Exception as e:
    print(f"❌ Error: {e}")

# ==================== MODEL 6: Neural Network (Current) ====================
print("\n" + "=" * 60)
print("6️⃣  NEURAL NETWORK (PyTorch - Current Model)")
print("=" * 60)
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    data = torch.load('data.pth')
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    
    nn_model = NeuralNet(input_size, hidden_size, output_size).to(device)
    nn_model.load_state_dict(data["model_state"])
    nn_model.eval()
    
    # Test on the same test set
    X_test_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        outputs = nn_model(X_test_tensor)
        _, nn_pred = torch.max(outputs, dim=1)
        nn_pred = nn_pred.cpu().numpy()
    
    nn_accuracy = accuracy_score(y_test, nn_pred)
    print(f"Accuracy: {nn_accuracy * 100:.2f}%")
    results.append(("Neural Network (PyTorch)", nn_accuracy))
except Exception as e:
    print(f"❌ Error: {e}")

# ==================== FINAL COMPARISON ====================
print("\n" + "=" * 60)
print(" FINAL RESULTS - MODEL COMPARISON")
print("=" * 60)

results.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Model':<30} {'Accuracy':<15}")
print("-" * 60)
for i, (model_name, accuracy) in enumerate(results, 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{medal} {i:<4} {model_name:<30} {accuracy*100:>6.2f}%")

print("\n" + "=" * 60)
print(f" WINNER: {results[0][0]} with {results[0][1]*100:.2f}% accuracy!")
print("=" * 60)

print(f"\n📊 Detailed Classification Report for {results[0][0]}:")
print("-" * 60)
if results[0][0] == "Logistic Regression":
    print(classification_report(y_test, lr_pred, target_names=tags))
elif results[0][0] == "Decision Tree":
    print(classification_report(y_test, dt_pred, target_names=tags))
elif results[0][0] == "Random Forest":
    print(classification_report(y_test, rf_pred, target_names=tags))
elif results[0][0] == "SVM (RBF kernel)":
    print(classification_report(y_test, svm_pred, target_names=tags))
elif results[0][0] == "Naive Bayes":
    print(classification_report(y_test, nb_pred, target_names=tags))
elif results[0][0] == "Neural Network (PyTorch)":
    print(classification_report(y_test, nn_pred, target_names=tags))