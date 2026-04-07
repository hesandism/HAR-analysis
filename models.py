from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier


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