import joblib
import json
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def encode_categorical_features(dt: pd.DataFrame):
    dt.drop("id", axis=1, inplace=True)
    text_columns = dt.select_dtypes(include=object).columns.tolist()
    encoded_df = pd.DataFrame()
    encoders = {} 
    for col in text_columns:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        col_data = dt[[col]]
        enc_col = pd.DataFrame(enc.fit_transform(col_data), columns=enc.categories_[0])
        encoded_df = pd.concat([encoded_df, enc_col], axis=1)
        encoders[col] = enc 
    numeric_df = dt.select_dtypes(include=np.number)
    encoded_df.reset_index(inplace=True, drop=True)
    numeric_df.reset_index(inplace=True, drop=True)
    df_merged = pd.concat([encoded_df, numeric_df], axis=1)
    
    df_merged.columns = [str(col).replace("(", "").replace(")", "").replace("'", "").strip() for col in df_merged.columns]
    
    os.makedirs("manual_svm_model_output", exist_ok=True)
    for col, enc in encoders.items():
        joblib.dump(enc, f"manual_svm_model_output/encoder_{col}.joblib")
    
    return df_merged

def split_data(features, labels):
    """
    Membagi data menjadi set pelatihan dan pengujian.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify = labels)
    return X_train, X_test, y_train, y_test


class ManualSVM:
    def __init__(self, learning_rate=0.001, epochs=1000, C=1.0, gamma=1.0, degree=3, coef0=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.w = None  
        self.b = None  
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.X_train_fit = None 

    def _polynomial_kernel(self, X1, X2):
        return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree

    def fit(self, X, y):
        self.X_train_fit = X 
        y_transformed = np.where(y <= 0, -1, 1) 

        n_samples, n_features = X.shape

        if self.w is None or len(self.w) != n_samples:
            self.w = np.random.randn(n_samples) * 0.01
            self.b = np.random.randn() * 0.01

        kernel_matrix = self._polynomial_kernel(X, X)
        
        for epoch in range(self.epochs):
            scores = np.dot(kernel_matrix, self.w) + self.b
            
            margins = y_transformed * scores
            
            grad_w = np.zeros_like(self.w)
            grad_b = 0

            for i, margin in enumerate(margins):
                if margin < 1: 
                    grad_w += -y_transformed[i] * kernel_matrix[i, :] 
                    grad_b += -y_transformed[i]
            
            grad_w_total = self.w + self.C * (grad_w / n_samples)
            grad_b_total = self.C * (grad_b / n_samples)
            
            self.w -= self.learning_rate * grad_w_total
            self.b -= self.learning_rate * grad_b_total
            
            if epoch % 100 == 0:
                loss = np.mean(np.maximum(0, 1 - margins)) + 0.5 * np.dot(self.w, self.w)
                print(f'Epoch {epoch}: Loss = {loss:.4f}')

    def predict(self, X):
        if self.w is None or self.b is None or self.X_train_fit is None:
            raise Exception("Model has not been trained yet. Call .fit() first.")
        
        kernel_test_train = self._polynomial_kernel(X, self.X_train_fit)
        
        scores = np.dot(kernel_test_train, self.w) + self.b
        return np.where(scores >= 0, 1, 0) # Klasifikasi

def evaluate_model(model, X, y, y_actual_labels):
    y_pred = model.predict(X)
    
    y_pred_int = y_pred.astype(int)
    y_actual_labels_int = y_actual_labels.astype(int)

    model_metrics = {} 
    model_metrics["Accuracy"] = metrics.accuracy_score(y_actual_labels_int, y_pred_int)
    model_metrics["Precision"] = metrics.precision_score(y_actual_labels_int, y_pred_int, zero_division=0)
    model_metrics["Recall"] = metrics.recall_score(y_actual_labels_int, y_pred_int, zero_division=0)
    return model_metrics

def save_model(folder, model, metrics, root_path):
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return
    
    model_data = {
        "alpha": model.w.tolist(), 
        "bias": model.b.item(),
        "X_train_fit": model.X_train_fit.tolist(), 
        "gamma": model.gamma,
        "degree": model.degree,
        "coef0": model.coef0
    }
    
    current_dir = os.getcwd()
    os.chdir(folder) 
    
    with open("manual_svm_params.json", "w") as outfile:
        json.dump(model_data, outfile)
    
    with open("metrics.json", "w") as outfile:
        json.dump(metrics, outfile)
    
    os.chdir(current_dir) 
    print(f"\nModel parameters (alpha dan b) dan metrics disimpan di '{os.path.join(root_path, folder)}'")

if __name__ == "__main__":
    root = os.getcwd() 

    try:
        df = pd.read_pickle("cleaned_data.pkl")
    except FileNotFoundError:
        print("Error: File 'cleaned_data.pkl' tidak ditemukan. Pastikan file ada di direktori yang sama.")
        exit() 
        
    df.columns = [str(col) for col in df.columns]

    df_merged = encode_categorical_features(df.copy())
    
    df_merged.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df_merged.columns]
    
    if 'stroke' in df_merged.columns:
        stroke = df_merged["stroke"].astype(int)
        df_features = df_merged.drop("stroke", axis=1)
    else:
        print("Kolom 'stroke' tidak ditemukan di DataFrame yang digabungkan.")
        raise ValueError("Kolom 'stroke' tidak ditemukan setelah encoding.")
    
    df_features = df_features.select_dtypes(include=np.number).fillna(0)
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(df_features.mean())
    
    X_train, X_test, y_train, y_test = split_data(df_features, stroke)
    
    print(f"\nProporsi kelas y_train sebelum resampling:\n{y_train.value_counts()}")

    counts = y_train.value_counts()
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    n_minority = counts[minority_class]
    n_majority = counts[majority_class]

    over_ratio = 0.5 
    n_minority_after_smote = int(n_majority * over_ratio)

    under_ratio = 0.8
    n_majority_after_undersample = int(n_minority_after_smote / under_ratio)

    n_minority_after_smote = max(n_minority_after_smote, n_minority)
    
    sampling_strategy_smote = {majority_class: n_majority, minority_class: n_minority_after_smote}
    sampling_strategy_undersample = {majority_class: n_majority_after_undersample, minority_class: n_minority_after_smote}

    over = SMOTE(sampling_strategy=sampling_strategy_smote, random_state=42)
    under = RandomUnderSampler(sampling_strategy=sampling_strategy_undersample, random_state=42)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)
    
    print(f"\nProporsi kelas y_train setelah SMOTE Over-sampling dan Random Under-sampling:\n{y_train_resampled.value_counts()}")
    
    pd.DataFrame(y_train_resampled, columns=['stroke']).value_counts().plot(kind='bar', title='Verify Resampling')
    plt.xlabel('Stroke')
    plt.ylabel('Count')
    plt.show()
    
    print("\nMemulai pelatihan Manual SVM dengan HANYA Kernel Polinomial...")
    manual_svm_model = ManualSVM(learning_rate=0.0001, epochs=2000, C=0.1, 
                                 gamma=0.1, degree=2, coef0=1) 
    
    manual_svm_model.fit(X_train_resampled.values, y_train_resampled.values)
    
    print("\n--- Hasil Evaluasi Model ---")
    train_eval = evaluate_model(manual_svm_model, X_train_resampled.values, y_train_resampled.values, y_train_resampled)
    print("\nTrain Metrics (Manual SVM):")
    for metric, value in train_eval.items():
        print(f"{metric}: {value:.4f}")
    
    test_eval = evaluate_model(manual_svm_model, X_test.values, y_test.values, y_test)
    print("\nTest Metrics (Manual SVM):")
    for metric, value in test_eval.items():
        print(f"{metric}: {value:.4f}")
    
    save_model("manual_svm_model_output", manual_svm_model, test_eval, root)