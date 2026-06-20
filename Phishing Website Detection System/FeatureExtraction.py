import re
import pandas as pd
import pickle
from urllib.parse import urlparse


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

    # Basic URL structure
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

    
    # Suspicious keywords
    suspicious_words = [
        'login','secure','account','update','verify',
        'bank','signin','confirm','password','billing'
    ]
    features.append(1 if any(word in url for word in suspicious_words) else 0)

    trusted_domains = [
                    "google.com",
                    "github.com",
                    "amazon.com",
                    "facebook.com",
                    "microsoft.com",
                    "apple.com",
                    "paypal.com",
                    "netflix.com",
                    "instagram.com",
                    "whatsapp.com"
    ]

    features.append(1 if any(domain in parsed.netloc for domain in trusted_domains) else 0)


    # Domain features
    features.append(len(parsed.netloc))
    features.append(len(parsed.path))
    features.append(len(parsed.query))

    # Numeric features
    digit_count = sum(c.isdigit() for c in url)
    features.append(digit_count)
    features.append(digit_count / len(url))

    # Special characters
    special = len(re.findall(r'[^a-zA-Z0-9]', url))
    features.append(special)
    features.append(special / len(url))

    # Protocol features
    features.append(1 if "https" in url else 0)
    features.append(1 if "http" in parsed.path else 0)

    # Short URL services
    features.append(1 if re.search(r'bit\.ly|goo\.gl|tinyurl', url) else 0)

    # Subdomains
    features.append(url.count('.')-1)

    # Long URL indicator
    features.append(1 if len(url) > 75 else 0)

    # Suspicious patterns
    features.append(1 if "//" in parsed.path else 0)
    features.append(1 if "-" in parsed.netloc else 0)

    # Additional structure
    features.append(url.count('www'))
    features.append(url.count('.com'))
    features.append(url.count('='))

    # Domain split
    features.append(len(parsed.netloc.split('.')))
    features.append(len(parsed.netloc.split('.')[0]))

    # Repeated characters
    features.append(len(re.findall(r'(.)\1', url)))

    # Vowel ratio
    vowels = sum(c in "aeiou" for c in url)
    features.append(vowels / len(url))

    # Consonant ratio
    consonants = sum(c.isalpha() for c in url) - vowels
    features.append(consonants / len(url))

    # Digit ratio again
    features.append(sum(c.isdigit() for c in url) / len(url))

    # Uppercase letters
    features.append(sum(c.isupper() for c in url))

    # Random sequences
    features.append(len(re.findall(r'[a-z]{4,}', url)))

    # Path segments
    features.append(len(parsed.path.split('/')))

    # Query parameters
    features.append(parsed.query.count('='))

    # Ensure exactly 50 features
    while len(features) < 50:
        features.append(0)

    return features


# ===============================
# LOAD MODEL
# ===============================

model = pickle.load(open("RandomForest_model.pkl","rb"))


# ===============================
# PREDICTION FUNCTION
# ===============================

def predict(url):

    features = featureExtraction(url)

    print("Number of features:", len(features))

    df = pd.DataFrame([features])

    prob = model.predict_proba(df)[0][1]

    print("Phishing probability:", round(prob,3))

# Trusted domain override
    trusted_domains = [
        "google.com",
        "drive.google.com",
        "github.com",
        "amazon.com",
        "facebook.com",
        "microsoft.com",
        "apple.com",
        "instagram.com",
        "whatsapp.com"
    ]

    if any(t in url for t in trusted_domains):
        return "LEGITIMATE WEBSITE"

    # Normal ML decision
    if prob > 0.80:
        return "HIGH PHISHING RISK"

    elif prob > 0.50:
        return "SUSPICIOUS WEBSITE"

    else:
        return "LEGITIMATE WEBSITE"


# ===============================
# TESTING
# ===============================

if __name__ == "__main__":

    test_url = input("Enter URL to test: ")

    result = predict(test_url)

    if result == "HIGH PHISHING RISK":
        print("\n⚠️ HIGH PHISHING RISK")

    elif result == "SUSPICIOUS WEBSITE":
        print("\n⚠️ Suspicious Website")

    else:
        print("\n✅ Legitimate Website")