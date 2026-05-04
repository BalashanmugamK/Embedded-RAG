# 📟 HARDCORE-AI Diagnostic Engine
### **Deterministic Hardware Fault RAG for Embedded Systems**

The **HARDCORE-AI Diagnostic Engine** is a specialized Retrieval-Augmented Generation (RAG) system architected to troubleshoot complex hardware faults in embedded systems. Developed as an original software project, this engine utilizes a **Hybrid Retrieval** strategy and **Manual Multi-Query Expansion** to provide grounded, deterministic diagnostics based on technical reference manuals (e.g., STM32, ESP8266).

---

## 🚀 Key Engineering Features

*   **Hybrid Retrieval Architecture**: Integrates semantic vector search with BM25 keyword scoring to ensure exact hexadecimal addresses are retrieved with mechanical precision.
*   **Manual Multi-Query Expansion**: Implements custom logic to generate technical variations of user queries, maximizing hit rates across complex hardware data tables.
*   **Deterministic Grounding Gate**: A strict protocol that prevents hallucinations; the system explicitly reports `TECHNICAL DATA NOT FOUND IN LOCAL MANUAL` if information is missing from the retrieved context.
*   **Table-Aware Chunking**: Optimized recursive character splitting (1000-char chunks with 250-char overlap) designed to maintain contextual links between register addresses and bit-field descriptions.
*   **Windows-Safe File Handling**: Robust backend management designed to handle OS-level file handles, enabling seamless manual updates during active indexing.

---

## 🛠️ Technical Stack

*   **Frontend**: Streamlit
*   **Orchestration**: LangChain
*   **Vector Database**: ChromaDB (Retrieval-Augmented Generation using ChromaDB)
*   **Embeddings**: `sentence-transformers/all-mpnet-base-v2` (768-dimensional technical embeddings)
*   **LLM**: OpenAI/OpenRouter (Deterministic `temperature=0`)
*   **Retrieval**: Hybrid Vector + BM25 with Multi-Query logic

---

## 📂 Project Structure
```text
.
├── app.py              # Streamlit UI & OS-safe file management
├── engine.py           # Multi-Query expansion & Hybrid Retrieval logic
├── data/               # Local repository for technical manuals (PDF/TXT)
├── chroma_db/          # Persistent vector database storage
├── .env                # API configuration and environment variables
└── requirements.txt    # Project dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/hardcore-ai-diagnostic.git](https://github.com/your-username/hardcore-ai-diagnostic.git)
   cd hardcore-ai-diagnostic


2.  **Set up a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install streamlit langchain langchain-openai langchain-chroma \
    langchain-community langchain-huggingface sentence-transformers \
    pypdf python-dotenv
    ```

4.  **Configure Environment**:
    Create a `.env` file in the root directory and add your credentials:
    ```env
    OPENAI_API_KEY=your_key_here
    OPENAI_API_BASE=https://openrouter.ai/api/v1
    ```

---

## 📟 Usage

1.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

2.  **Upload a Manual**: Use the sidebar to upload a `.pdf` or `.txt` technical reference manual.
3.  **Re-Index**: Execute the **🔥 Re-Index and Reload** trigger to update the persistent vector database.
4.  **Query Examples**:
    *   **Address Lookup**: "Explain `0x60000000`"
    *   **Bit-Level Analysis**: "What is the function of `IACCVIOL`?"
    *   **Diagnostic Logic**: "Identify the root cause of a forced hard fault."

---
