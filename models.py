from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def get_models():
    models = {
        "Decision Tree":      DecisionTreeClassifier(random_state=42), 
        "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression":LogisticRegression(max_iter=1000, C=1.0, random_state=42), 
        "Linear SVC":         LinearSVC(C=1.0, max_iter=2000, random_state=42),     
        "RBF SVM":            SVC(kernel='rbf', C=10, gamma='scale', random_state=42), 
        "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    return models


def train_and_evaluate(models, X_train, y_train, X_test, y_test):

    results = []

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        Xtr =  X_train
        Xte =  X_test
        start = time.time()
        model.fit(Xtr, y_train)
        end = time.time()
        training_time = round(end - start, 2)
        print(f"Trained {name} in {training_time} seconds.")
        y_pred = model.predict(Xte)
        trained_models[name] = (model,y_pred)

        results.append({
            "Model":     name,
            "Precision": round(precision_score(y_test, y_pred, average='weighted'), 2),
            "Recall":    round(recall_score   (y_test, y_pred, average='weighted'), 2),
            "F1-Score":  round(f1_score       (y_test, y_pred, average='weighted'), 2),
            "Accuracy":  round(accuracy_score (y_test, y_pred), 2),
            "Training Time": training_time
        })
    return results, trained_models