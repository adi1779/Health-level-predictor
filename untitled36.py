# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.base import clone

import warnings
warnings.filterwarnings("ignore")

# קריאת הקובץ
df = pd.read_csv("extended_new_dataset.csv")

# שליפת משתנים
X = df.drop(['Person ID', 'Index'], axis=1)
y = df['Index']
classes = sorted(y.unique())

# נורמליזציה
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# פיצול לסטים
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# בסיס למודל Bagging: עץ פשוט ולא עמוק (למניעת Overfitting)
base_tree_for_bagging = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=10, class_weight='balanced', random_state=42
)

# הגדרת כל המודלים עם פרמטרים נגד Overfitting
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(
        max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(
        learning_rate=0.05, n_estimators=100, max_depth=3, random_state=42),
    'XGBoost': XGBClassifier(
        max_depth=3, subsample=0.8, colsample_bytree=0.8,
        learning_rate=0.05, n_estimators=100,
        eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    'ExtraTrees': ExtraTreesClassifier(
        max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42),
    'Bagging': BaggingClassifier(
        estimator=base_tree_for_bagging,
        n_estimators=50,
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=100,
        learning_rate=0.8,
        random_state=42
    ),
    'DecisionTree': DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42)
}

# אחסון תוצאות
all_reports = []
all_predictions = []
summary_metrics = []
auc_scores = {}

# Binarize labels for ROC
y_test_bin = label_binarize(y_test, classes=classes)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # מטריצת בלבול
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # דוח ביצועים
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['Model'] = name
    all_reports.append(report_df)

    # תחזיות הסתברות
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr')
        auc_scores[name] = roc_auc

        # גרף ROC
        plt.figure()
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f"Class {classes[i]}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title(f"ROC Curve - {name} (AUC = {roc_auc:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()
    else:
        auc_scores[name] = None

    # תקציר ביצועים כללי
    summary_metrics.append({
        'Model': name,
        'Accuracy': report['accuracy'],
        'Macro Avg Precision': report['macro avg']['precision'],
        'Macro Avg Recall': report['macro avg']['recall'],
        'Macro Avg F1-Score': report['macro avg']['f1-score'],
        'Weighted Avg Precision': report['weighted avg']['precision'],
        'Weighted Avg Recall': report['weighted avg']['recall'],
        'Weighted Avg F1-Score': report['weighted avg']['f1-score'],
        'ROC AUC Score': auc_scores[name]
    })

    # חיזויים
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    pred_df['Model'] = name
    all_predictions.append(pred_df)

# איחוד תוצאות וייצוא
final_report = pd.concat(all_reports)
final_report.to_excel("classification_report_all_models.xlsx")

predictions_df = pd.concat(all_predictions)
predictions_df.to_excel("all_model_predictions.xlsx", index=False)

# טבלת סיכום מדדים מרכזיים
summary_df = pd.DataFrame(summary_metrics)
summary_df.to_excel("summary_metrics_per_model.xlsx", index=False)

print("✅ הכל מוכן. נשמרו הקבצים:")
print("📄 classification_report_all_models.xlsx")
print("📄 all_model_predictions.xlsx")
print("📄 summary_metrics_per_model.xlsx")