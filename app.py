"""
HARDCOREAI — Streamlit frontend
New in this version:
  • Mode selector: STRICT / MEDIUM / LENIENT  (sidebar)
  • Faithfulness badge + expandable claim audit after every streamed answer
  • Feedback loop: thumbs up/down + free-text correction → stored in session
  • URL ingestion: paste any YouTube or article link in sidebar
  • Startup spinner only while warmup thread is running (fast first render)
  • Clean shutdown: atexit in engine handles Chroma flush
"""
import streamlit as st
import os, shutil, re

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="HARDCORE-AI",
    page_icon="📟",
    layout="wide",
    initial_sidebar_state="expanded",
)

from engine import HardcoreEngine, MODES, DEFAULT_MODE

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "messages":          [],
    "pending_prompt":    None,
    "last_indexed_file": None,
    "mode":              DEFAULT_MODE,
    "feedback":          [],        # list of {turn, verdict, correction}
    "faith_results":     {},        # turn_index → faithfulness dict
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Control")

    # ── Mode selector ─────────────────────────────────────────────────────────
    st.subheader("🎛️ Hallucination Guard Mode")
    mode_options = list(MODES.keys())
    chosen_mode  = st.radio(
        "Select mode",
        mode_options,
        index=mode_options.index(st.session_state.mode),
        format_func=lambda m: MODES[m]["label"],
        help="\n\n".join(f"**{MODES[m]['label']}**: {MODES[m]['description']}"
                         for m in mode_options),
    )
    if chosen_mode != st.session_state.mode:
        st.session_state.mode = chosen_mode
        # invalidate cached engine so it rebuilds with new mode settings
        st.cache_resource.clear()
        st.rerun()

    st.caption(MODES[chosen_mode]["description"])
    st.divider()

    # ── File upload ───────────────────────────────────────────────────────────
    st.subheader("📂 Upload Manual")
    uploaded_file = st.file_uploader("(.txt or .pdf)", type=["txt", "pdf"])
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Saved: {uploaded_file.name}")
        already = st.session_state.last_indexed_file == uploaded_file.name
        if already:
            st.success("✅ Index up to date.")
        else:
            if st.button("🔥 Re-Index and Reload", type="primary", use_container_width=True):
                st.session_state.last_indexed_file = uploaded_file.name
                st.cache_resource.clear()
                if os.path.exists("chroma_db"):
                    try:
                        shutil.rmtree("chroma_db")
                    except PermissionError:
                        st.error("⚠️ DB locked — close other app instances, then retry.")
                        st.session_state.last_indexed_file = None
                        st.stop()
                st.rerun()

    st.divider()

    # ── URL ingestion ─────────────────────────────────────────────────────────
    st.subheader("🌐 Ingest URL")
    st.caption("Paste a YouTube video or article URL to add to the knowledge base.")
    url_input = st.text_input("URL", placeholder="https://youtube.com/watch?v=... or https://...")
    if st.button("⬇️ Load URL", use_container_width=True) and url_input.strip():
        with st.spinner("Fetching and indexing URL..."):
            # engine must be loaded first — call through the cached loader
            def _get_engine():
                return st.session_state.get("_engine_ref")
            _eng = _get_engine()
            if _eng:
                msg = _eng.ingest_url(url_input.strip())
                st.info(msg)
            else:
                st.warning("Engine not ready yet — try again in a moment.")

    st.divider()

    # ── System status ─────────────────────────────────────────────────────────
    st.header("📊 Status")
    c1, c2 = st.columns(2)
    c1.metric("Mode", MODES[st.session_state.mode]["label"])
    c2.metric("Ranking", "RRF")
    st.info("Embedding: all-mpnet-base-v2\nChunks: 1000 / 250 overlap\nTop-K: 8 fused")

    st.divider()
    st.header("📜 Session")
    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    fb_count   = len(st.session_state.feedback)
    ca, cb = st.columns(2)
    ca.metric("Queries", user_turns)
    cb.metric("Feedback", fb_count)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.pending_prompt = None
        st.session_state.faith_results  = {}
        st.rerun()

    if st.session_state.feedback:
        with st.expander(f"📋 Feedback log ({fb_count} items)"):
            for i, fb in enumerate(st.session_state.feedback):
                st.write(f"**Turn {fb['turn']}** — {fb['verdict']}")
                if fb.get("correction"):
                    st.caption(f"Correction: {fb['correction']}")


# ── Engine loading ────────────────────────────────────────────────────────────
def _get_latest_data_file() -> str:
    os.makedirs("data", exist_ok=True)
    files = [f for f in os.listdir("data") if f.endswith((".txt", ".pdf"))]
    if not files:
        return "data/stm32_manual.txt"
    files.sort(key=lambda x: os.path.getmtime(os.path.join("data", x)), reverse=True)
    return os.path.join("data", files[0])


@st.cache_resource(show_spinner=False)   # spinner handled manually below
def load_engine(data_file: str, mode: str) -> HardcoreEngine | None:
    try:
        eng = HardcoreEngine(data_file=data_file, mode=mode)
        return eng
    except Exception as e:
        st.error(f"Engine init failed: {e}")
        return None


current_file = _get_latest_data_file()
engine = load_engine(current_file, st.session_state.mode)

# Store ref for URL ingestion button above
if engine:
    st.session_state["_engine_ref"] = engine

# Sidebar status
with st.sidebar:
    if engine:
        if engine._ready.is_set():
            chunk_count = len(engine.chunks) if engine.chunks else 0
            st.success(f"✅ Ready — {chunk_count} chunks")
        else:
            st.info("⏳ Warming up embeddings...")
    else:
        st.warning("⚠️ No engine — upload a manual.")


# ── Page header ───────────────────────────────────────────────────────────────
st.title("📟 HARDCORE-AI Diagnostic Engine")
st.markdown(
    f"**Mode:** {MODES[st.session_state.mode]['label']} &nbsp;|&nbsp; "
    "Streaming RAG · Hybrid Retrieval · RRF · Faithfulness Scoring"
)

# ── Quick buttons (only on empty chat) ───────────────────────────────────────
EXAMPLES = [
    "Explain register 0xE000ED28",
    "Teach me UART from scratch",
    "void spi_slave_init(uint8 spi_no)",
    "Show chat history",
]
if not st.session_state.messages:
    st.markdown("**Quick queries:**")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state.pending_prompt = ex
            st.rerun()

# ── Replay history ────────────────────────────────────────────────────────────
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show saved faithfulness badge beneath assistant messages
        if message["role"] == "assistant" and i in st.session_state.faith_results:
            fr = st.session_state.faith_results[i]
            _render_faith_badge(fr) if False else None  # rendered below in live block

# ── Chat input ────────────────────────────────────────────────────────────────
typed = st.chat_input("Ask anything — register, function, concept, or paste a URL")
if typed:
    # Auto-detect URL pasted into chat input
    if re.match(r"https?://", typed.strip()):
        if engine and engine._ready.is_set():
            with st.spinner("Fetching and indexing URL..."):
                msg = engine.ingest_url(typed.strip())
            st.success(msg)
        else:
            st.warning("Engine warming up — try again in a moment.")
        st.stop()  # don't treat URL as a query
    st.session_state.pending_prompt = typed


# ── Helper: render faithfulness badge ────────────────────────────────────────
def render_faith_badge(fr: dict):
    score     = fr["score"]
    threshold = fr.get("threshold", 0.5)
    color     = "green" if score >= threshold else ("orange" if score >= threshold * 0.6 else "red")
    st.caption(fr["verdict"])
    if fr["unsupported_claims"]:
        with st.expander(f"🔬 Claim audit — {len(fr['unsupported_claims'])} unsupported claim(s)"):
            st.warning("The following claims could not be verified against the manual:")
            for c in fr["unsupported_claims"]:
                st.markdown(f"- ❌ {c}")
            st.info("Use feedback below to submit the correct answer.")


# ── Unified processing block ──────────────────────────────────────────────────
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not engine:
        with st.chat_message("assistant"):
            st.error("Engine not initialised. Upload a manual in the sidebar.")
    else:
        # Block only while warmup is in progress (usually done by first query)
        if not engine._ready.is_set():
            with st.spinner("⏳ Warming up engine — almost ready..."):
                engine._ready.wait(timeout=90)

        with st.chat_message("assistant"):
            history_for_engine = st.session_state.messages[:-1]
            ctx    = engine.prepare_context(prompt, chat_history=history_for_engine)
            qtype  = ctx["query_type"]

            # ── Instant answers ───────────────────────────────────────────────
            if ctx["direct_answer"] is not None:
                st.markdown(ctx["direct_answer"])
                full_response = ctx["direct_answer"]

            # ── Streaming RAG ─────────────────────────────────────────────────
            else:
                # Badges before streaming
                badge_parts = []
                type_icon = {
                    "register": "🗂️ Register",
                    "function": "⚙️ Function",
                    "concept":  "📖 Concept",
                    "teach":    "🎓 Teaching",
                    "followup": "💬 Follow-up",
                }.get(qtype, "")
                if type_icon:
                    badge_parts.append(type_icon)
                if ctx["hex_detected"]:
                    badge_parts.append("🔎 " + " ".join(f"`{h}`" for h in ctx["hex_detected"]))
                conf = ctx.get("confidence", 0)
                if conf:
                    badge_parts.append(
                        f"{'🟢' if conf>=60 else '🟡' if conf>=30 else '🔴'} {conf}% match"
                    )
                if badge_parts:
                    st.caption("  |  ".join(badge_parts))

                if not ctx["is_grounded"] and ctx["hex_detected"]:
                    st.error(f"🚨 {ctx['hex_detected']} not found in manual.")
                if ctx.get("low_context") and st.session_state.mode != "STRICT":
                    st.info("ℹ️ Limited manual coverage — gaps filled with 🧠 general knowledge.")

                # Stream tokens
                full_response = st.write_stream(engine.stream_answer(ctx["system_prompt"]))

                # ── Faithfulness scoring (async — shown after streaming) ───────
                if ctx["context_text"] and full_response:
                    with st.spinner("🔬 Scoring faithfulness..."):
                        fr = engine.score_faithfulness(
                            prompt, full_response, ctx["context_text"]
                        )
                    # Store against the assistant message index (next append)
                    msg_idx = len(st.session_state.messages)  # index AFTER append below
                    st.session_state.faith_results[msg_idx] = fr
                    render_faith_badge(fr)

                    # ── Feedback widget ───────────────────────────────────────
                    turn = sum(1 for m in st.session_state.messages if m["role"] == "user")
                    threshold = MODES[st.session_state.mode]["faith_threshold"]
                    if fr["score"] < threshold:
                        st.warning(
                            f"**Faithfulness below threshold ({fr['score']:.0%} < {threshold:.0%})**\n\n"
                            "Was this answer correct? Your feedback helps improve future responses."
                        )
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("👍 Correct enough", key=f"fb_ok_{turn}"):
                                st.session_state.feedback.append(
                                    {"turn": turn, "verdict": "accepted_low_faith", "correction": ""}
                                )
                                st.toast("Thanks for the feedback!")
                        with col_no:
                            correction = st.text_area(
                                "👎 Provide the correct answer:",
                                key=f"fb_corr_{turn}", height=80
                            )
                            if st.button("Submit correction", key=f"fb_sub_{turn}") and correction:
                                st.session_state.feedback.append(
                                    {"turn": turn, "verdict": "corrected", "correction": correction}
                                )
                                st.toast("Correction saved — thank you!")

                # ── Source snippets ───────────────────────────────────────────
                if ctx["sources"]:
                    with st.expander(f"🔍 {len(ctx['sources'])} source snippets"):
                        for i, doc in enumerate(ctx["sources"]):
                            meta = doc.metadata or {}
                            label = (f"Snippet {i+1} — {meta.get('source','Manual')}"
                                     + (f", p.{meta['page']}" if meta.get('page') else ""))
                            st.info(f"**{label}**\n\n{doc.page_content}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})