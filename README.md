# ⚖️ HakiBot – AI-Powered Legal Assistant for Kenya

**HakiBot** is an intelligent question-answering system that helps Kenyan citizens access quick and accurate legal information. It focuses on **labour rights** and **police conduct**, providing answers derived from the Kenyan Constitution and relevant laws. This tool is designed for awareness, empowerment, and civic education.

---

## 📌 Features

- ✅ 100+ curated legal Q&A based on Kenyan law
- 🤖 AI-powered similarity search using Sentence Transformers & FAISS
- 🧠 Understands user questions and returns the most relevant answer
- 🆓 Built entirely with free, open-source tools
- 🌐 Future-ready for chatbot/Streamlit integration

---

## 🔧 Tech Stack

| Component | Description |
|----------|-------------|
| `Python` | Core language |
| `Pandas` | Data loading and processing |
| `SentenceTransformers` | Embeds legal questions into vector space |
| `FAISS` | Fast, scalable similarity search |
| `Streamlit`|  For chatbot web interface  |

---

## 🚀 How It Works

1. Legal Q&A pairs are loaded from `legal_qa.csv`
2. Questions are converted to vector embeddings
3. A FAISS index is built from the embeddings
4. When a user asks a question, the system:
   - Encodes the input
   - Finds the most similar stored question
   - Returns the corresponding answer

---

## 📁 Project Structure

hakibot/
├── legal_qa.csv # Dataset with 100 Q&A entries
├── app.py # Main script: builds index and answers questions
├── legal_index.index # Saved FAISS index
├── legal_df.pkl # Serialized DataFrame
├── requement.txt
└── README.md # You're reading it!


---

## 🧪 Usage

1. Clone the repo

```bash
git clone https://github.com/yourusername/hakibot.git


✅ Future Improvements

🌐 Integrate with WhatsApp/SMS bots

🏛️ Expand to cover land law, family law, and business law

🗂️ Enable multilingual support (e.g., Kiswahili)

🙏 Acknowledgements
Kenyan Constitution (2010)

Employment Act (2007)

Independent Policing Oversight Authority (IPOA)

Sentence Transformers (by UKP Lab)

FAISS (by Facebook AI Research)

📜 Disclaimer
HakiBot provides general legal information — not legal advice. For specific legal matters, please consult a qualified lawyer or legal aid organization.

🧑‍💻 Author
Magdalene Thuo
Digital Trainer | AI for Software Engineering

LinkedIn | Portfolio | Email



