from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import pickle
import numpy as np
import os


# Cargar embeddings desde .pkl
with open("harrypotter_embeddings.pkl", "rb") as f:
    embeddings, texts, metadatas = pickle.load(f)

# Cargar modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Configurar Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ Falta la variable de entorno 'GEMINI_API_KEY'")
genai.configure(api_key=API_KEY)

modelo_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Crear app Flask
app = Flask(__name__)
CORS(app)

# Función para generar respuesta
def generar_respuesta(prompt, contexto):
    full_prompt = (
        "Respondé la siguiente pregunta EXCLUSIVAMENTE con base en el contexto proporcionado. "
        "NO uses conocimiento externo ni inventes. Si no hay información suficiente, decílo.\n\n"
        f"### CONTEXTO:\n{contexto}\n\n"
        f"### PREGUNTA:\n{prompt}\n\n"
        "### RESPUESTA:"
    )
    try:
        respuesta = modelo_gemini.generate_content(full_prompt)
        return respuesta.text.strip()
    except Exception as e:
        print("❌ Error con Gemini:", e)
        return f"⚠️ Error con Google AI: {str(e)}"

# Ruta de consulta
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Falta el campo 'prompt'"}), 400

    query_emb = embedding_model.encode(prompt).reshape(1, -1)
    sim_scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(sim_scores)[-5:][::-1]
    top_chunks = [texts[i] for i in top_indices]

    contexto = "\n\n".join(top_chunks)
    respuesta = generar_respuesta(prompt, contexto)

    return jsonify({
        "context": top_chunks,
        "answer": respuesta
    })

if __name__ == "__main__":
 port = int(os.environ.get("PORT", 5000))
 app.run(host="0.0.0.0", port=port)
# app.run(port=5000, debug=True)