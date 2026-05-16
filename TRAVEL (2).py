from flask import Flask, request, jsonify, Response
import pandas as pd
import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings, ChatOllama

# ----------------- Data & model setup (runs once) -----------------

# CSV must be in same folder: travel.csv
# Columns: name, genre, budget, crowd, hidden_gem, description
df = pd.read_csv("travel.csv")

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

documents = df["description"].tolist()

vectors = []
for text in documents:
    embedding = embeddings_model.embed_query(text)
    vectors.append(embedding)

vectors = np.array(vectors, dtype="float32")

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

llm = ChatOllama(model="phi3")

# ----------------- Flask app -----------------

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Indian Travel AI</title>
  <style>
    * {
      box-sizing: border-box;
    }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;

      /* Attractive background image */
      background-image: url("https://images.pexels.com/photos/753626/pexels-photo-753626.jpeg");
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;

      /* Dark overlay for readability */
      background-color: rgba(0,0,0,0.55);
      background-blend-mode: darken;
      color: #e5e7eb;
    }
    .chat-container {
      width: 90%;
      max-width: 900px;
      height: 90vh;
      background: rgba(2, 6, 23, 0.9);
      border-radius: 16px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.85);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border: 1px solid #1f2937;
      backdrop-filter: blur(12px);
    }
    .chat-header {
      padding: 16px 20px;
      border-bottom: 1px solid #1f2937;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: radial-gradient(circle at top left, #2563eb, #020617);
    }
    .chat-header h1 {
      margin: 0;
      font-size: 1.3rem;
      color: #f9fafb;
    }
    .chat-header span {
      font-size: 0.85rem;
      color: #d1d5db;
    }
    .messages {
      flex: 1;
      padding: 16px;
      overflow-y: auto;
      background: radial-gradient(circle at top, rgba(37, 99, 235, 0.2), transparent);
    }
    .message {
      max-width: 80%;
      margin-bottom: 12px;
      padding: 10px 14px;
      border-radius: 12px;
      line-height: 1.4;
      white-space: pre-wrap;
      font-size: 0.95rem;
    }
    .user {
      margin-left: auto;
      background: #2563eb;
      color: #f9fafb;
    }
    .bot {
      margin-right: auto;
      background: #020617;
      border: 1px solid #1f2937;
    }
    .places {
      margin-top: 8px;
      font-size: 0.8rem;
      color: #9ca3af;
    }
    .input-area {
      display: flex;
      padding: 12px;
      border-top: 1px solid #1f2937;
      background: #020617;
      gap: 8px;
    }
    #queryInput {
      flex: 1;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid #374151;
      background: #020617;
      color: #e5e7eb;
      outline: none;
      font-size: 0.95rem;
    }
    #queryInput::placeholder {
      color: #6b7280;
    }
    #sendBtn {
      padding: 10px 18px;
      border-radius: 999px;
      border: none;
      background: linear-gradient(to right, #22c55e, #16a34a);
      color: #020617;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.95rem;
      white-space: nowrap;
    }
    #sendBtn:disabled {
      background: #4b5563;
      cursor: default;
    }
    @media (max-width: 640px) {
      .chat-container {
        height: 95vh;
      }
      .message {
        max-width: 100%;
      }
    }
  </style>
</head>
<body>
<div class="chat-container">
  <div class="chat-header">
    <div>
      <h1>Indian Travel AI</h1>
      <span>Ask about trips, treks, beaches, hidden gems across India.</span>
    </div>
  </div>

  <div class="messages" id="messages">
    <div class="message bot">
      Hi! I’m your Indian travel assistant. Ask me anything about travel destinations,
      hidden gems, or trip ideas across India!
    </div>
  </div>

  <div class="input-area">
    <input
      type="text"
      id="queryInput"
      placeholder="Type your travel query and press Enter…"
    />
    <button id="sendBtn">Send</button>
  </div>
</div>

<script>
  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("queryInput");
  const sendBtn = document.getElementById("sendBtn");

  function appendMessage(text, type, places) {
    const div = document.createElement("div");
    div.className = "message " + type;
    div.textContent = text;

    if (type === "bot" && places && places.length) {
      const pDiv = document.createElement("div");
      pDiv.className = "places";
      pDiv.textContent =
        "Suggested places: " +
        places.map(p => p.name + " (" + p.genre + ", " + p.budget + ")").join(" | ");
      div.appendChild(pDiv);
    }

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendQuery() {
    const query = inputEl.value.trim();
    if (!query) return;

    appendMessage(query, "user");
    inputEl.value = "";
    inputEl.focus();
    sendBtn.disabled = true;

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query }),
      });

      const data = await res.json();
      if (res.ok) {
        appendMessage(data.answer, "bot", data.places);
      } else {
        appendMessage(data.error || "Something went wrong.", "bot");
      }
    } catch (err) {
      appendMessage("Server error. Check the backend logs.", "bot");
    } finally {
      sendBtn.disabled = false;
    }
  }

  sendBtn.addEventListener("click", sendQuery);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuery();
  });
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Embed query
    query_vector = embeddings_model.embed_query(query)
    query_vector = np.array([query_vector], dtype="float32")

    # FAISS search
    distances, indices = index.search(query_vector, k=2)

    retrieved_places = []
    for idx in indices[0]:
        retrieved_places.append(df.iloc[int(idx)].to_dict())

    # Build context
    context = ""
    for place in retrieved_places:
        context += f"""
Place: {place['name']}
Genre: {place['genre']}
Budget: {place['budget']}
Crowd: {place['crowd']}
Hidden Gem: {place['hidden_gem']}
Description: {place['description']}
"""

    prompt = f"""
You are an Indian travel recommendation AI.

Recommend places based on the user query.

User Query:
{query}

Context:
{context}

Give personalized recommendations.
"""

    response = llm.invoke(prompt)

    return jsonify({
        "answer": response.content,
        "places": retrieved_places
    })


if __name__ == "__main__":
    app.run(debug=True)