import openai
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

load_dotenv()

messages = []
user_profiles = {}
app = Flask(__name__, template_folder='templates')

openai.api_key = os.getenv("OPEN_API_KEY")

# Print API key to ensure it is being loaded correctly
print("Loaded OpenAI API key:", openai.api_key)


class Chatbot:
    def __init__(self):
        self.user_levels = {"new": 0, "occasional": 5, "frequent": 10}
        self.user_level = "new"
        self.interactions_count = 0
        self.classifier = self.train_classifier()
        self.messages = []
        self.user_profile = {}

    def train_classifier(self):
        data = [
            {"input": "Greeting", "level": "new"},
            {"input": "Common cold symptoms", "level": "beginner"},
            {"input": "Understanding diabetes", "level": "intermediate"},
            {"input": "Signs of heart disease", "level": "intermediate"},
            {"input": "Preventing respiratory infections", "level": "beginner"},
            {"input": "Symptoms of influenza", "level": "intermediate"},
            {"input": "Risk factors for cancer", "level": "intermediate"},
            {"input": "Managing allergies", "level": "beginner"},
            {"input": "Recognizing mental health disorders", "level": "intermediate"},
            {"input": "Exploring autoimmune diseases", "level": "advanced"},
            {"input": "Understanding digestive disorders", "level": "intermediate"},
            {"input": "Preventing cardiovascular diseases", "level": "intermediate"},
            {"input": "Common skin conditions", "level": "beginner"},
            {"input": "Signs of hormonal imbalances", "level": "intermediate"},
            {"input": "Coping with chronic pain", "level": "advanced"},
            {"input": "Early detection of infectious diseases", "level": "intermediate"},
            {"input": "Importance of vaccinations", "level": "beginner"},
            {"input": "Managing mental health challenges", "level": "intermediate"},
            {"input": "Exploring neurological disorders", "level": "advanced"},
            {"input": "Recognizing symptoms of common infections", "level": "beginner"},
            {"input": "Caring for individuals with chronic conditions", "level": "advanced"},
            {"input": "Understanding genetic disorders", "level": "advanced"},
        ]

        # Split data into training and testing sets
        X = [item["input"] for item in data]
        y = [item["level"] for item in data]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a pipeline with CountVectorizer and RandomForestClassifier
        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', RandomForestClassifier(random_state=42)),
        ])

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Test the classifier accuracy
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Classifier Accuracy: {accuracy}")

        return classifier

    def detect_user_level_ml(self, user_input):
        # Use the trained classifier to predict the user level
        predicted_level = self.classifier.predict([user_input])[0]
        self.user_level = predicted_level
        return predicted_level

    def process_user_input(self, input_text):
        # Process user input here
        self.interactions_count += 1

        # Update user level based on interactions
        self.detect_user_level_ml(input_text)

    def generate_response(self, user_id, input_text, knowledge_level):
        if user_id not in self.user_profile:
            self.user_profile[user_id] = {"knowledge_level": knowledge_level, "interactions": []}
        user_profile = self.user_profile[user_id]

        self.messages = [{"role": "user", "content": input_text},
                         {"role": "user", "content": f"I am a {knowledge_level}."}]
        if knowledge_level == "beginner":
            self.messages.append({"role": "assistant", "content": "I'll provide answers suitable for beginners."})
        elif knowledge_level == "intermediate":
            self.messages.append({"role": "assistant", "content": "I'll provide answers suitable for intermediate knowledge."})
        elif knowledge_level == "advanced":
            self.messages.append({"role": "assistant", "content": "I'll provide advanced-level answers."})

        try:
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                max_tokens=150,
                temperature=0.7
            )
            reply = chat['choices'][0]['message']['content']
        except openai.OpenAIError as e:
            print(f"OpenAI Error: {e}")
            reply = "An error occurred, please try again later."

        self.messages.append({"role": "assistant", "content": reply})
        user_profile["interactions"].append({"input": input_text, "output": reply})

        # Feedback handling logic
        feedback = user_profile.get("feedback", None)
        if feedback is not None:
            if feedback == "too_easy":
                knowledge_level = 'advanced'
            elif feedback == "too_hard":
                knowledge_level = 'beginner'

            if feedback in user_profiles[user_id]:
                del user_profiles[user_id][feedback]

        return reply


chatbot_instance = Chatbot()

def getApiResponse(user_input):
    # This function can be used to get a response from the chatbot
    user_id = 'default_user'  # You can modify this to use actual user IDs if needed
    knowledge_level = 'beginner'  # Default knowledge level, can be adjusted as needed
    return chatbot_instance.generate_response(user_id, user_input, knowledge_level)


@app.route('/')
def home():
    return render_template('index.html', messages=messages)


@app.route('/chatbot', methods=['POST'])
def chat():
    input_text = request.form['message']
    knowledge_level = request.form.get('knowledge_level', 'beginner')
    user_id = request.form.get('user_id')

    response = chatbot_instance.generate_response(user_id, input_text, knowledge_level)
    output = {'response': response, 'feedback_request': False}

    return jsonify(output)


@app.route('/feedback', methods=['POST'])
def feedback():
    user_id = request.form.get('user_id')
    feedback = request.form.get('feedback')

    if user_id in user_profiles and "response" in user_profiles[user_id]:
        response = user_profiles[user_id]["response"]
        user_profiles[user_id]["feedback"] = feedback
        output = {'response': response, 'feedback_request': True}
        return jsonify(output)
    else:
        return jsonify(error='No response found for the user')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
