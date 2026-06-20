import re
import pandas as pd
import pickle
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# ===============================
# FEATURE EXTRACTION (50 FEATURES)
# ===============================

def featureExtraction(url):

    try:
        url = str(url).strip().lower()

        if not url.startswith(("http://","https://")):
            url = "http://" + url

        parsed = urlparse(url)

    except:
        return [0]*50


    features = []

    # basic URL features
    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('-'))
    features.append(url.count('@'))
    features.append(url.count('?'))
    features.append(url.count('&'))
    features.append(url.count('|'))
    features.append(url.count('='))
    features.append(url.count('_'))
    features.append(url.count('%'))
    features.append(url.count('/'))
    features.append(url.count('*'))
    features.append(url.count(':'))
    features.append(url.count(','))
    features.append(url.count(';'))
    features.append(url.count('$'))
    features.append(url.count(' '))

    # keyword indicators
    keywords = ['login','secure','account','update','verify','bank','signin','confirm','password','billing']
    features.append(1 if any(word in url for word in keywords) else 0)

    # trusted domain detection
    trusted_domains = [
        "google.com",
        "drive.google.com",
        "usercontent.google.com",
        "amazon.com",
        "github.com",
        "microsoft.com",
        "apple.com",
        "facebook.com",
        "instagram.com",
        "whatsapp.com"
    ]

    features.append(1 if any(t in parsed.netloc for t in trusted_domains) else 0)

    # domain info
    features.append(len(parsed.netloc))
    features.append(len(parsed.path))
    features.append(len(parsed.query))

    # numeric properties
    digit_count = sum(c.isdigit() for c in url)
    features.append(digit_count)
    features.append(digit_count/len(url))

    # special character count
    special = len(re.findall(r'[^a-zA-Z0-9]', url))
    features.append(special)
    features.append(special/len(url))

    # protocol indicators
    features.append(1 if "https" in url else 0)
    features.append(1 if "http" in parsed.path else 0)

    # shortening services
    features.append(1 if re.search(r'bit\.ly|goo\.gl|tinyurl', url) else 0)

    # subdomain count
    features.append(url.count('.')-1)

    # long URL indicator
    features.append(1 if len(url) > 75 else 0)

    # suspicious patterns
    features.append(1 if "//" in parsed.path else 0)
    features.append(1 if "-" in parsed.netloc else 0)

    # more structural features
    features.append(url.count('www'))
    features.append(url.count('.com'))
    features.append(url.count('='))

    # additional numeric signals
    features.append(len(parsed.netloc.split('.')))
    features.append(len(parsed.netloc.split('.')[0]))

    # repeated characters
    features.append(len(re.findall(r'(.)\1', url)))

    # vowels ratio
    vowels = sum(c in "aeiou" for c in url)
    features.append(vowels/len(url))

    # consonant ratio
    consonants = sum(c.isalpha() for c in url) - vowels
    features.append(consonants/len(url))

    # digit ratio
    features.append(sum(c.isdigit() for c in url)/len(url))

    # uppercase letters
    features.append(sum(c.isupper() for c in url))

    # random character sequences
    features.append(len(re.findall(r'[a-z]{4,}', url)))

    # path segments
    features.append(len(parsed.path.split('/')))

    # query parameters
    features.append(parsed.query.count('='))

    # ensure total features = 50
    while len(features) < 50:
        features.append(0)

    return features


# ===============================
# LOAD DATASET
# ===============================

dataset = pd.read_csv("phishing_site_urls.csv")

dataset["Label"] = dataset["Label"].astype(str).str.lower()

dataset["label"] = dataset["Label"].apply(
    lambda x: 1 if x in ["bad","phishing","1"] else 0
)

print("Dataset size:", len(dataset))


# ===============================
# FEATURE EXTRACTION
# ===============================

X = [featureExtraction(url) for url in dataset["URL"]]
y = dataset["label"]

X = pd.DataFrame(X)
y = pd.Series(y)


# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===============================
# MODELS
# ===============================

models = {

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            class_weight="balanced",
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            eval_metric="logloss",
            use_label_encoder=False
        )
}


# ===============================
# TRAIN MODELS
# ===============================

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name + " Confusion Matrix")
    plt.savefig(name + "_confusion_matrix.png")
    plt.close()


    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
    plt.plot([0,1],[0,1],'--')
    plt.title(name + " ROC Curve")
    plt.legend()
    plt.savefig(name + "_roc_curve.png")
    plt.close()


    # Metrics Graph
    metrics = ["Accuracy","Precision","Recall","F1"]
    values = [accuracy, precision, recall, f1]

    plt.figure()
    plt.bar(metrics, values)
    plt.ylim(0,1)
    plt.title(name + " Metrics")
    plt.savefig(name + "_metrics.png")
    plt.close()


    pickle.dump(model, open(name+"_model.pkl","wb"))


print("\nTraining completed.")