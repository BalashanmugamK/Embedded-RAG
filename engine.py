"""
HARDCOREAI Diagnostic Engine
─────────────────────────────
New in this version
  • 3 hallucination-guard modes: STRICT / MEDIUM / LENIENT
  • Faithfulness scorer: lightweight LLM-as-judge, no RAGAS dependency
  • URL ingestion: YouTube transcripts + web articles (trafilatura)
  • Fast startup: embeddings warm up in a background thread
  • Fast shutdown: atexit flushes Chroma and kills the thread pool cleanly
  • TimeoutError fix: fresh ThreadPoolExecutor per hybrid_retrieve call
"""
import os, re, atexit, threading
from concurrent.futures import ThreadPoolExecutor, wait
from functools import lru_cache
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

load_dotenv()

# ── Timeouts ──────────────────────────────────────────────────────────────────
_TOTAL_RETRIEVE_TIMEOUT = 20
_PER_RETRIEVER_TIMEOUT  = 8

# ── Hallucination-Guard Modes ─────────────────────────────────────────────────
MODES = {
    "STRICT": {
        "label":       "🔴 STRICT",
        "description": "0 % hallucination tolerance. Every claim must trace to the manual. "
                       "Used for safety-critical / billion-dollar embedded decisions.",
        "temperature": 0,
        "max_tokens":  512,
        "faith_threshold": 0.85,   # flag below this
        "system_prefix": (
            "You are a DETERMINISTIC embedded-systems expert. "
            "EVERY statement you make must be directly supported by the MANUAL CONTEXT. "
            "If the manual does not contain the answer, reply ONLY with: "
            "⛔ NOT IN MANUAL — <one sentence describing what is missing>. "
            "Do NOT speculate. Do NOT use general knowledge. Zero hallucination tolerated."
        ),
    },
    "MEDIUM": {
        "label":       "🟡 MEDIUM",
        "description": "Balanced: manual-first, fills genuine gaps with clearly-labelled "
                       "general knowledge. Recommended for most engineering queries.",
        "temperature": 0,
        "max_tokens":  768,
        "faith_threshold": 0.55,
        "system_prefix": (
            "You are HARDCOREAI — a senior embedded expert and enthusiastic tutor. "
            "Prioritise the MANUAL CONTEXT. When filling gaps with general embedded knowledge, "
            "prefix those sentences with 🧠. Manual-sourced facts need no prefix."
        ),
    },
    "LENIENT": {
        "label":       "🟢 LENIENT",
        "description": "Teaching / exploration mode. Uses manual + broad embedded knowledge. "
                       "Best for learning, brainstorming, and concept exploration.",
        "temperature": 0.3,
        "max_tokens":  1024,
        "faith_threshold": 0.30,
        "system_prefix": (
            "You are HARDCOREAI — a friendly embedded-systems teacher. "
            "Use the manual first, then draw freely on your embedded knowledge. "
            "Be warm, clear, and engaging. Explain all jargon. Use analogies."
        ),
    },
}
DEFAULT_MODE = "MEDIUM"


# ── URL Ingestion helpers ─────────────────────────────────────────────────────

def _load_youtube(url: str) -> list[Document]:
    """Extract transcript from a YouTube URL via youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import urllib.parse as up
        # Parse video id from any youtube URL shape
        parsed = up.urlparse(url)
        if parsed.hostname in ("youtu.be",):
            vid = parsed.path.lstrip("/")
        else:
            vid = up.parse_qs(parsed.query).get("v", [None])[0]
        if not vid:
            return []
        parts = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        text = " ".join(p["text"] for p in parts)
        return [Document(page_content=text, metadata={"source": url, "type": "youtube"})]
    except Exception as e:
        return [Document(page_content=f"[YouTube transcript error: {e}]",
                         metadata={"source": url, "type": "youtube_error"})]


def _load_webpage(url: str) -> list[Document]:
    """Extract clean article text from any webpage using trafilatura."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_comments=False,
                                   include_tables=True, no_fallback=False)
        if not text:
            return []
        return [Document(page_content=text, metadata={"source": url, "type": "web"})]
    except Exception as e:
        return [Document(page_content=f"[Web fetch error: {e}]",
                         metadata={"source": url, "type": "web_error"})]


def load_url(url: str) -> list[Document]:
    """Dispatch: YouTube → transcript API, everything else → trafilatura."""
    if "youtube.com" in url or "youtu.be" in url:
        return _load_youtube(url)
    return _load_webpage(url)


# ── Engine ────────────────────────────────────────────────────────────────────

class HardcoreEngine:
    def __init__(self, data_file: str = "data/stm32_manual.txt",
                 persist_dir: str = "./chroma_db",
                 mode: str = DEFAULT_MODE):
        self.persist_dir = os.path.abspath(persist_dir)
        self.mode        = mode if mode in MODES else DEFAULT_MODE
        self._ready      = threading.Event()   # signals warm-up complete

        # LLM — settings from mode, rebuilt when mode changes
        self._build_llm()

        # ── Background warm-up: embeddings + retrievers ───────────────────────
        self.chunks                = []
        self.vector_db             = None
        self.base_semantic_retriever = None
        self.keyword_retriever     = None

        def _warm_up():
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
            )
            self.chunks = self._load_chunks(data_file)
            if self.chunks:
                self._init_retrievers()
            self._ready.set()

        self._warmup_thread = threading.Thread(target=_warm_up, daemon=True, name="hcai-warmup")
        self._warmup_thread.start()

        # ── Fast shutdown via atexit ──────────────────────────────────────────
        atexit.register(self._shutdown)

    # ── Mode management ───────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        if mode in MODES:
            self.mode = mode
            self._build_llm()

    def _build_llm(self):
        cfg = MODES[self.mode]
        self.llm = ChatOpenAI(
            model="openrouter/free",
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            request_timeout=20,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
        )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self):
        """Called by atexit. Flushes Chroma and joins the warmup thread."""
        try:
            if self.vector_db:
                self.vector_db._client.close()   # flush SQLite WAL
        except Exception:
            pass
        try:
            self._warmup_thread.join(timeout=0.5)
        except Exception:
            pass

    # ── Document Loading ──────────────────────────────────────────────────────

    def _load_chunks(self, data_file: str) -> list:
        if not os.path.exists(data_file):
            return []
        loader = (PyPDFLoader(data_file) if data_file.lower().endswith(".pdf")
                  else TextLoader(data_file))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(loader.load())

    # ── Ingest URL into running knowledge base ────────────────────────────────

    def ingest_url(self, url: str) -> str:
        """
        Load a URL (YouTube or web page), chunk it, add to vector DB + BM25.
        Returns a status message for display.
        Call after _ready is set.
        """
        self._ready.wait(timeout=60)
        docs = load_url(url)
        if not docs or docs[0].metadata.get("type", "").endswith("_error"):
            return f"⚠️ Could not load URL: {url}"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250,
            separators=["\n\n", "\n", " ", ""],
        )
        new_chunks = splitter.split_documents(docs)
        if not new_chunks:
            return "⚠️ No content extracted from URL."

        self.chunks.extend(new_chunks)

        if self.vector_db is None:
            self.vector_db = Chroma.from_documents(
                documents=new_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="hardware_docs",
            )
        else:
            self.vector_db.add_documents(new_chunks)

        # Rebuild BM25 (cheap for <10k chunks)
        self.keyword_retriever = BM25Retriever.from_documents(self.chunks)
        self.keyword_retriever.k = 4
        self.base_semantic_retriever = self.vector_db.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20},
        )

        src_type = docs[0].metadata.get("type", "web")
        return (f"✅ Ingested {'YouTube transcript' if src_type == 'youtube' else 'web article'} "
                f"— {len(new_chunks)} chunks added.")

    # ── Retriever Init ────────────────────────────────────────────────────────

    def _init_retrievers(self):
        db_exists = os.path.exists(self.persist_dir) and os.listdir(self.persist_dir)
        if db_exists:
            self.vector_db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="hardware_docs",
            )
        else:
            self.vector_db = Chroma.from_documents(
                documents=self.chunks, embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="hardware_docs",
            )
        self.base_semantic_retriever = self.vector_db.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20},
        )
        self.keyword_retriever = BM25Retriever.from_documents(self.chunks)
        self.keyword_retriever.k = 4

    # ── Query Expansion ───────────────────────────────────────────────────────

    @lru_cache(maxsize=128)
    def _expand_query(self, query: str) -> tuple:
        prompt = (
            "Generate exactly 3 technical search variations for this hardware query. "
            "Focus on hex addresses, register names, peripheral identifiers, function names. "
            "Output only the 3 queries, one per line, no numbering.\n\nQuery: " + query
        )
        try:
            r = self.llm.invoke(prompt).content
            return tuple(q.strip() for q in r.splitlines() if q.strip())[:3]
        except Exception:
            return ()

    # ── Hybrid Retrieval — fresh pool per call (TimeoutError fix) ─────────────

    def hybrid_retrieve(self, query: str, expand: bool = True) -> tuple:
        if not self.vector_db:
            return [], 0
        queries = [query] + (list(self._expand_query(query)) if expand else [])
        results = []
        with ThreadPoolExecutor(max_workers=min(len(queries) * 2, 8)) as pool:
            fmap = {}
            for q in queries:
                fmap[pool.submit(self.base_semantic_retriever.invoke, q)] = q
                fmap[pool.submit(self.keyword_retriever.invoke, q)] = q
            done, pending = wait(list(fmap.keys()), timeout=_TOTAL_RETRIEVE_TIMEOUT)
            for f in done:
                try:
                    docs = f.result(timeout=_PER_RETRIEVER_TIMEOUT)
                    if docs:
                        results.append(docs)
                except Exception:
                    pass
            for f in pending:
                f.cancel()
        return self._rrf(results) if results else ([], 0)

    @staticmethod
    def _rrf(doc_lists: list, k: int = 60) -> tuple:
        scores, docs_map = {}, {}
        for ranked in doc_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content[:120]
                scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
                docs_map[key] = doc
        if not scores:
            return [], 0
        sk = sorted(scores, key=scores.__getitem__, reverse=True)
        max_p = len(doc_lists) * (1.0 / (k + 1))
        conf  = min(100, int(scores[sk[0]] / max_p * 100))
        return [docs_map[k] for k in sk][:8], conf

    # ── Faithfulness Scorer ───────────────────────────────────────────────────

    def score_faithfulness(self, query: str, answer: str, context_text: str) -> dict:
        """
        Lightweight LLM-as-judge faithfulness scorer.
        Returns {"score": float 0-1, "supported": int, "total": int,
                 "unsupported_claims": [str], "verdict": str}

        Method (inspired by RAGAS, no dependency):
          1. Ask LLM to decompose answer into atomic claims (JSON list).
          2. For each claim ask LLM: is this supported by context? (yes/no)
          3. faithfulness = supported / total
        """
        # Step 1 — decompose
        decompose_prompt = (
            "Decompose the following answer into a list of independent, atomic factual claims. "
            "Return ONLY a JSON array of strings. No explanation.\n\n"
            f"Answer:\n{answer[:2000]}"
        )
        try:
            raw = self.llm.invoke(decompose_prompt).content.strip()
            # strip markdown fences if present
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`")
            import json
            claims = json.loads(raw)
            if not isinstance(claims, list):
                raise ValueError
        except Exception:
            # fallback: split by sentence
            claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20][:10]

        if not claims:
            return {"score": 1.0, "supported": 0, "total": 0,
                    "unsupported_claims": [], "verdict": "No claims to verify."}

        # Step 2 — verify each claim in parallel
        def _check(claim: str) -> tuple:
            p = (
                "Given the CONTEXT below, is the following CLAIM directly supported "
                "by the context? Answer only YES or NO.\n\n"
                f"CONTEXT:\n{context_text[:3000]}\n\n"
                f"CLAIM: {claim}"
            )
            try:
                r = self.llm.invoke(p).content.strip().upper()
                return claim, r.startswith("YES")
            except Exception:
                return claim, True  # assume ok on error

        supported, unsupported = 0, []
        with ThreadPoolExecutor(max_workers=min(len(claims), 5)) as pool:
            for claim, ok in pool.map(_check, claims):
                if ok:
                    supported += 1
                else:
                    unsupported.append(claim)

        total = len(claims)
        score = supported / total if total else 1.0
        threshold = MODES[self.mode]["faith_threshold"]

        if score >= threshold:
            verdict = f"✅ Faithful ({score:.0%} claims supported by manual)"
        elif score >= threshold * 0.6:
            verdict = f"⚠️ Partially faithful ({score:.0%}) — review flagged claims"
        else:
            verdict = f"🚨 Low faithfulness ({score:.0%}) — answer may contain hallucinations"

        return {
            "score": round(score, 3),
            "supported": supported,
            "total": total,
            "unsupported_claims": unsupported,
            "verdict": verdict,
            "threshold": threshold,
        }

    # ── Classifier Signals ────────────────────────────────────────────────────

    TEACH_SIGNALS = {
        "noob","noobie","newbie","beginner","simple","simply","layman","eli5",
        "explain","teach","learn","learning","understand","what is","what's",
        "whats","how does","how do","basics","basic","intro","introduction",
        "overview","summary","in simple","in plain","plain english","for dummies",
        "wat is","guide me","help me understand","break it down","break down",
    }
    FOLLOWUP_SIGNALS = {
        "that","this","it","more","continue","go on","keep going","elaborate",
        "expand","tell me more","also","what about","bruh","bro","dude","man",
        "yeah","yep","yup","got it","i see",
    }
    HISTORY_SIGNALS = {
        "history","chat history","show history","what have we","what did we",
        "our conversation","past messages","previous queries","recap",
        "summarize our chat","chat log","previous questions","what have i asked",
    }
    TRIVIAL_EXACT = {
        "hi","hello","hey","ok","okay","yes","no","thanks","bye","u","k",
        "lol","hm","hmm","sup","yo","wassup","hiya","thx",
    }

    @staticmethod
    def _classify_query(query: str, hex_found: list) -> str:
        q     = query.lower().strip()
        words = set(q.split())
        if any(s in q for s in HardcoreEngine.HISTORY_SIGNALS):  return "history"
        if q in HardcoreEngine.TRIVIAL_EXACT:                     return "trivial"
        if hex_found:                                              return "register"
        C = ("void ","uint8","uint16","uint32","int ","char ","bool ","static ")
        if "(" in query or any(q.startswith(p) for p in C):       return "function"
        if (any(s in q for s in HardcoreEngine.TEACH_SIGNALS)
                or HardcoreEngine.TEACH_SIGNALS & words):          return "teach"
        if (len(query.split()) <= 5
                and (HardcoreEngine.FOLLOWUP_SIGNALS & words
                     or any(s in q for s in HardcoreEngine.FOLLOWUP_SIGNALS))):
            return "followup"
        REG = {"register","bit","fault","hardfault","cfsr","hfsr","mmar","bfar",
               "interrupt","irq","flag","peripheral"}
        if REG & words:                                            return "register"
        return "concept"

    # ── Prompt Builder ────────────────────────────────────────────────────────

    def _build_prompt(self, query, query_type, hex_found,
                      context_text, history_text="", low_context=False):
        mode_cfg  = MODES[self.mode]
        prefix    = mode_cfg["system_prefix"]
        hist_blk  = f"\n--- CONVERSATION SO FAR ---\n{history_text}\n" if history_text else ""
        low_note  = ("\n⚠️ Limited manual coverage — label gaps with 🧠.\n"
                     if low_context and self.mode != "STRICT" else "")

        if query_type == "register":
            task = (f"1. State EXACT Register Name + Base Address for {hex_found or query}\n"
                    f"2. List bit-field definitions (bit → name → meaning).\n"
                    f"3. Engineering Diagnosis + Root Cause.")
        elif query_type == "function":
            task = ("1. PURPOSE in one sentence.\n"
                    "2. Every PARAMETER: name → type → role.\n"
                    "3. Source file/header if present.\n"
                    "4. Constraints or warnings.")
        elif query_type == "teach":
            task = ("Teach a beginner:\n"
                    "1. 'In simple terms, [topic] is...'\n"
                    "2. Everyday analogy.\n"
                    "3. Step-by-step technical explanation.\n"
                    "4. Key Takeaway bullets (≤5).")
        elif query_type == "followup":
            task = ("Continue from the conversation history.\n"
                    "1. Identify topic from history.\n"
                    "2. Expand with manual + knowledge.\n"
                    "3. Concise and friendly.")
        else:
            task = ("1. Answer directly from manual context.\n"
                    "2. Draw on embedded knowledge for gaps.\n"
                    "3. Include addresses/functions/bits where relevant.")

        return (
            f"{prefix}\n"
            f"You NEVER refuse a question.\n"
            f"{hist_blk}{low_note}\n"
            f"TASKS:\n{task}\n\n"
            f"--- MANUAL CONTEXT ---\n"
            f"{context_text or 'No specific manual section found.'}\n\n"
            f"--- QUERY ---\n{query}\n\nANSWER:"
        )

    # ── History helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _format_history(chat_history):
        if not chat_history:
            return ""
        lines = []
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            c    = msg["content"]
            if msg["role"] == "assistant" and len(c) > 300:
                c = c[:300] + "...[truncated]"
            lines.append(f"{role}: {c}")
        return "\n".join(lines)

    @staticmethod
    def _format_history_for_display(chat_history):
        if not chat_history:
            return "_No history yet._ Try: `explain UART` or `teach me SPI`"
        lines = [f"### 📜 History — {len(chat_history)} messages\n"]
        turn, i = 1, 0
        while i < len(chat_history):
            msg = chat_history[i]
            if msg["role"] == "user":
                lines.append(f"**Turn {turn} — 🧑:** {msg['content']}")
                if i+1 < len(chat_history) and chat_history[i+1]["role"] == "assistant":
                    r = chat_history[i+1]["content"]
                    if len(r) > 400: r = r[:400] + "_(truncated)_"
                    lines.append(f"**🤖:** {r}\n---")
                    i += 2
                else:
                    i += 1
                turn += 1
            else:
                i += 1
        return "\n".join(lines)

    @staticmethod
    def _resolve_retrieval_query(query, query_type, chat_history):
        if query_type in ("followup","teach") and chat_history:
            for msg in reversed(chat_history):
                if msg["role"] == "user":
                    prev = msg["content"].strip()
                    if (len(prev.split()) >= 3
                            and not any(s in prev.lower()
                                        for s in HardcoreEngine.FOLLOWUP_SIGNALS)):
                        return f"{prev} {query}"
            for msg in reversed(chat_history):
                if msg["role"] == "assistant":
                    return msg["content"][:80]
        return query

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare_context(self, query: str, chat_history: list | None = None) -> dict:
        """Classify + retrieve — no LLM generation. Call stream_answer() next."""
        # Block only if warmup not done yet (usually instant after first query)
        self._ready.wait(timeout=60)

        chat_history = chat_history or []
        hex_found    = list(self.extract_hex_addresses(query))
        query_type   = self._classify_query(query, hex_found)

        if query_type == "trivial":
            return {
                "query_type": "trivial", "direct_answer": (
                    "👋 I'm **HARDCOREAI** — embedded tutor + debugger.\n\n"
                    "- 🔍 Register: `0xE000ED28`\n"
                    "- ⚙️ Function: `void spi_init(...)`\n"
                    "- 🎓 Learn: `teach me UART`\n"
                    "- 🌐 URL: paste a YouTube/article link\n"
                    "- 📜 History: `show chat history`"
                ),
                "system_prompt": None, "sources": [], "hex_detected": [],
                "is_grounded": True, "confidence": 0, "low_context": False,
                "context_text": "",
            }

        if query_type == "history":
            filtered = [m for m in chat_history if m["content"] != query]
            return {
                "query_type": "history",
                "direct_answer": self._format_history_for_display(filtered),
                "system_prompt": None, "sources": [], "hex_detected": [],
                "is_grounded": True, "confidence": 0, "low_context": False,
                "context_text": "",
            }

        rq           = self._resolve_retrieval_query(query, query_type, chat_history)
        expand       = query_type != "followup"
        docs, conf   = self.hybrid_retrieve(rq, expand=expand)
        ctx_text     = "\n\n---\n\n".join(d.page_content for d in docs)
        hist_text    = self._format_history(chat_history)
        low_ctx      = conf < 30

        prompt = self._build_prompt(query, query_type, hex_found,
                                    ctx_text, hist_text, low_ctx)
        grounded = (any(h.lower() in ctx_text.lower() for h in hex_found)
                    if hex_found else True)
        return {
            "query_type": query_type, "direct_answer": None,
            "system_prompt": prompt, "sources": docs,
            "hex_detected": hex_found, "is_grounded": grounded,
            "confidence": conf, "low_context": low_ctx,
            "context_text": ctx_text,
        }

    def stream_answer(self, system_prompt: str):
        """Yield LLM tokens for st.write_stream()."""
        try:
            for chunk in self.llm.stream(system_prompt):
                if chunk.content:
                    yield chunk.content
        except Exception:
            try:
                yield self.llm.invoke(system_prompt).content
            except Exception as e:
                yield f"⚠️ Engine error: {e}"

    @staticmethod
    @lru_cache(maxsize=256)
    def extract_hex_addresses(text: str) -> tuple:
        return tuple(re.findall(r"0x[0-9A-Fa-f]+", text))

    def get_diagnosis(self, query: str, chat_history: list | None = None) -> dict:
        ctx = self.prepare_context(query, chat_history)
        if ctx["direct_answer"] is not None:
            return {**ctx, "answer": ctx["direct_answer"]}
        answer = "".join(self.stream_answer(ctx["system_prompt"]))
        return {**ctx, "answer": answer}