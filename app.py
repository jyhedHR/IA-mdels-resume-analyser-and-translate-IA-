from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import spacy
import torch

app = Flask(__name__)
CORS(app, origins=["https://trelix-livid.vercel.app, http://localhost:5173,https://trelix-xj5h.onrender.com"], supports_credentials=True)

# ========== Load Translation Model ==========
print("Loading translation model...")
translation_model_name = "facebook/m2m100_418M"
translation_tokenizer = M2M100Tokenizer.from_pretrained(translation_model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)
print("Translation model loaded successfully!")

# ========== Load Spacy NLP Model ==========
print("Loading Spacy NLP model...")
nlp = spacy.load("en_core_web_trf")
print("Spacy model loaded successfully!")

# ========== Routes ==========

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Combined Translation + NLP API is running"}), 200

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON payload"}), 400
        
        text = data.get("text")
        source_lang = data.get("source_lang")
        target_lang = data.get("target_lang")

        if not all([text, source_lang, target_lang]):
            return jsonify({"error": "Fields 'text', 'source_lang', and 'target_lang' are required"}), 400

        translation_tokenizer.src_lang = source_lang
        encoded = translation_tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            generated_tokens = translation_model.generate(
                **encoded,
                forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang)
            )

        translated_text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return jsonify({"translated_text": translated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        text = data.get('cvText', '')

        if not text:
            return jsonify({"error": "Missing 'cvText' in request body"}), 400

        doc = nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        return jsonify({"entities": entities})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== Run App ==========
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host="0.0.0.0", port=port)
