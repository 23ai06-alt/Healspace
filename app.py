from flask import Flask, render_template, request, jsonify
import csv
import numpy as np
import pandas as pd
import random
import re
import os

# NLP & ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ----------------- Quiz Functions -----------------
def load_questions():
    questions = []
    with open('questions.csv', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row)
    return questions

@app.route("/")
def home_page():
    return render_template("dashboard.html", username="Guest")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", username="Guest")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/contact")
def contact_page():
    return render_template("contact.html")


@app.route("/quiz")
def quiz_page():
    questions = load_questions()
    return render_template("quiz.html", questions=questions)

@app.route("/result", methods=["POST"])
def result():
    category_scores = {
        "Depression": 0,
        "Anxiety": 0,
        "Bipolar": 0,
        "Schizophrenia": 0,
        "EatingDisorder": 0,
        "Dementia": 0
    }

    # ✅ Question-to-category mapping based on your CSV
    question_to_category = {
        "1": "Depression",
        "2": "Anxiety",
        "3": "Bipolar",
        "4": "Depression",
        "5": "Schizophrenia",
        "6": "Anxiety",
        "7": "Depression",
        "8": "Schizophrenia",
        "9": "Bipolar",
        "10": "Anxiety",
        "11": "EatingDisorder",
        "12": "Dementia",
        "13": "Dementia",
        "14": "EatingDisorder",
        "15": "Schizophrenia",
        "16": "Depression",
        "17": "Anxiety",
        "18": "Bipolar"
    }

    data = request.get_json()
    total_questions = len(data)
    yes_count = 0
    no_count = 0

    # ✅ Count responses properly using the mapping
    for key, value in data.items():
        if value.lower() == "yes":
            yes_count += 1
            category = question_to_category.get(key)
            if category in category_scores:
                category_scores[category] += 1
        elif value.lower() == "no":
            no_count += 1

    # ✅ If all answers are "Yes"
    if yes_count == total_questions:
        return jsonify({
            "diagnosis": "You might be experiencing multiple mental health challenges.",
            "tips": "It’s strongly recommended to consult a psychologist for a proper diagnosis.",
            "video": "https://youtu.be/_bqT6X0Viac?si=38YE10EvmX5K8HLz",
        })

    # ✅ If all answers are "No"
    if no_count == total_questions:
        return jsonify({
            "diagnosis": "You seem perfectly fine!",
            "tips": "Keep maintaining your mental well-being. Continue doing what keeps you happy and stress-free.",
            "video": "https://youtu.be/MMX6S-4gjqQ?si=mW0-wP2ta7sGiE58",
        })

    # ✅ Otherwise, pick the highest scoring category
    diagnosis = max(category_scores, key=category_scores.get)

    recommendations = {
        "Depression": {
            "tips": "Try to get sunlight every morning. Start journaling your thoughts. Engage in regular physical activity.",
            "youtube": "https://youtu.be/gyQX6bU1NIY?si=YNtnsWMD4stRK6Ih",
        },
        "Anxiety": {
            "tips": "Practice deep breathing exercises. Reduce caffeine and get proper sleep. Try meditation or yoga daily.",
            "youtube": "https://youtu.be/whrN7ujh3Yk?si=4flfMG2KDJ7_NvWf",
        },
        "Bipolar": {
            "tips": "Maintain a consistent sleep schedule. Avoid alcohol and substance use. Keep a daily mood journal.",
            "youtube": "https://youtu.be/llOPqKD-s4w?si=t1p8GkyhXl9CkhJP",
        },
        "Schizophrenia": {
            "tips": "Stay connected with supportive people. Follow a consistent daily routine. Avoid isolation and seek therapy.",
            "youtube": "https://youtu.be/57bR35CA3RE?si=3BL9DfLF7Pe2zE4Y",
        },
        "EatingDisorder": {
            "tips": "Eat balanced meals regularly. Avoid strict diets or skipping meals. Talk to a nutritionist or therapist.",
            "youtube": "https://youtu.be/-ZTDy9vrzd0?si=08zfK_3E6I4ye3A5",
        },
        "Dementia": {
            "tips": "Play memory games daily. Stay physically and mentally active. Maintain a balanced diet rich in omega-3s.",
            "youtube": "https://youtu.be/QlGJA-_a3uM?si=AvJ8wySmDvgDEsuD",
        }
    }

    return jsonify({
        "diagnosis": f"{diagnosis}",
        "tips": recommendations[diagnosis]["tips"],
        "video": recommendations[diagnosis]["youtube"],
    })


# ----------------- Chatbot Section -----------------
csv_path = os.path.join('static', 'mental_health_qna.csv')
data = pd.read_csv(csv_path)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

data['question_processed'] = data['question'].apply(preprocess)

# TF-IDF + Naive Bayes setup
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(data['question_processed'])

le = LabelEncoder()
y = le.fit_transform(data['response'])
clf_nb = MultinomialNB()
clf_nb.fit(X_tfidf, y)

@app.route("/get", methods=["POST"])
def get_chatbot_response():
    user_input = request.json['message']
    user_input_processed = preprocess(user_input)

    # 👋 Greeting handling
    greetings = ["hi", "hello", "hey", "heyy", "hii"]
    if any(word in user_input_processed.split() for word in greetings):
        reply = random.choice([
            "Hello there! 😊 How are you feeling today?",
            "Hey! 💚 How’s your day going so far?",
            "Hi! 🌼 How are you doing right now?"
        ])
        return jsonify({"reply": reply})

    # 💬 Step 1: Exact match first (strongest signal)
    exact_matches = data[data['question_processed'] == user_input_processed]
    if not exact_matches.empty:
        reply = exact_matches.iloc[0]['response']
        endings = ["💙", "🤍 You’re safe here.", "💛 Remember, you matter."]
        return jsonify({"reply": reply + " " + random.choice(endings)})

    # 💡 Step 2: TF-IDF + cosine similarity for semantic intent
    user_vec = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vec, X_tfidf).flatten()

    # ✅ Step 3: Boost longer & more specific questions
    weighted_similarity = []
    for i, sim in enumerate(similarity):
        q_len = len(data['question_processed'][i].split())
        if sim > 0:  # only count related entries
            # boost if question has more context words (e.g. "symptoms of loneliness")
            boost = 1 + (q_len / 10)
            weighted_similarity.append(sim * boost)
        else:
            weighted_similarity.append(sim)

    best_index = int(np.argmax(weighted_similarity))
    best_score = weighted_similarity[best_index]

    # 🤖 Step 4: Naive Bayes fallback for unseen input
    nb_pred_index = clf_nb.predict(user_vec)[0]
    nb_pred_response = le.inverse_transform([nb_pred_index])[0]
    nb_confidence = clf_nb.predict_proba(user_vec).max()

    # 🎯 Step 5: Decision logic (choose the best based on confidence)
    if best_score > 0.25:
        reply = data['response'][best_index].strip()
    elif nb_confidence > 0.6:
        reply = nb_pred_response.strip()
    else:
        reply = "I hear you 💛. Can you tell me more about how you're feeling?"

    # 🌈 Step 6: Add supportive tone
    endings = ["💙", "🤍 You’re safe here.", "💛 Remember, you matter."]
    reply = reply + " " + random.choice(endings)
    return jsonify({"reply": reply})



# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(debug=True)
