"""
app.py - The simple web UI (Streamlit).

Run with:   streamlit run app.py
"""
import html
import tempfile
from pathlib import Path

import streamlit as st

import agent
import config
import ingest
import prompts
import reasoning
import vector_store

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(f"<style>{(Path(__file__).parent / 'styles.css').read_text()}</style>", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <div class="eyebrow">Neural research interface · online</div>
  <h1>AGENTIC RAG</h1>
  <p>Ask anything about Agentic RAG. The agent decides whether to search, rewrites your question,
  picks a source and searches again if needed.</p>
  <div class="pills"><span class="pill">notes</span><span class="pill">arXiv</span>
  <span class="pill">wikipedia</span><span class="pill">code</span><span class="pill">local ollama</span></div>
</div>
""", unsafe_allow_html=True)



# ---- Top control bar (no sidebar): buttons around the screen ------------------------
def add_knowledge(upload, link, force):
    with st.spinner("Reading, checking for duplicates and adding..."):
        try:
            if upload:
                path = Path(tempfile.gettempdir()) / upload.name
                path.write_bytes(upload.getvalue())
                result = ingest.ingest_any(str(path), force=force)
            else:
                result = ingest.ingest_any(link.strip(), force=force)
            if result["added"]:
                extra = f" (incl. {result['images']} figures)" if result.get("images") else ""
                st.success(f"Added {result['chunks']} chunks{extra}.")
            else:
                st.warning(f"Not added: {result['reason']}")
        except Exception as error:
            st.error(f"Could not add it: {error}")


if "history" not in st.session_state:
    st.session_state.history = []

rail_left, center, rail_right = st.columns([1, 4, 1.15], gap="medium")

# left rail: search controls, stacked vertically
with rail_left:
    st.markdown('<div class="rail-title">Search mode</div>', unsafe_allow_html=True)
    st.session_state.setdefault("search_mode", vector_store.SEARCH_MODES[0])
    for mode in vector_store.SEARCH_MODES:
        if st.button(mode, key=f"mode_{mode}", use_container_width=True,
                     type="primary" if st.session_state.search_mode == mode else "secondary"):
            st.session_state.search_mode = mode
            st.rerun()
    search_mode = st.session_state.search_mode
    st.caption("similarity = nearest · threshold = only relevant · mmr = relevant + diverse")
    k = st.slider("Chunks (k)", 1, 10, config.DEFAULT_K)
    compare = st.toggle("Compare with plain LLM", value=False)

# right rail: add knowledge, always visible
with rail_right:
    st.markdown('<div class="rail-title">Add knowledge</div>', unsafe_allow_html=True)
    upload = st.file_uploader("PDF, image or text file",
                              type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md"])
    link = st.text_input("…or a web / YouTube link")
    force = st.checkbox("Add even if duplicate / off-topic")
    if (upload or link) and st.button("Add to database", use_container_width=True):
        add_knowledge(upload, link, force)
    st.markdown(f'<div class="chip"><span>{vector_store.count_chunks()}</span> chunks</div>',
                unsafe_allow_html=True)
    if st.button("↺ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

with center:
    style = st.pills("Reasoning style", list(prompts.ANSWER_PROMPTS), default=list(prompts.ANSWER_PROMPTS)[0],
                     label_visibility="collapsed") or list(prompts.ANSWER_PROMPTS)[0]


LOADER = """
<div class="loader">
  <div class="orb"><i></i><i></i><i></i><b></b></div>
  <div class="phases"><span>Planning the query</span><span>Searching the sources</span>
  <span>Reading the evidence</span><span>Composing the answer</span></div>
  <div class="bar"></div>
</div>
"""


# ---- Main area: chat -----------------------------------------------------------
def show_answer(result, question=""):
    """Zero/Few-shot: plain text. Chain of Thought: a chain of steps. Tree of Thoughts: a tree."""
    style, answer = result.get("style"), result["answer"]
    esc = html.escape
    if style == "Chain of Thought" and (chain := reasoning.parse_chain(answer)):
        steps = "".join(
            f'<div class="step"><div class="dot">{n:02d}</div><div class="t">{esc(t)}</div>'
            f'<div class="b">{esc(x)}</div></div>'
            for n, (t, x) in enumerate(chain["steps"], start=1))
        st.markdown(f'<div class="chain">{steps}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="final"><div class="lbl">Final answer</div>{esc(chain["final"])}</div>',
                    unsafe_allow_html=True)
    elif style == "Tree of Thoughts" and (tree := reasoning.parse_tree(answer)):
        st.graphviz_chart(reasoning.tree_to_dot(question, tree), use_container_width=True)
        colors = {"strong": "var(--green)", "weak": "var(--amber)", "none": "var(--red)"}
        for letter, b in tree["branches"].items():
            best = letter == tree["best"]
            why = f'<div class="why">{esc(b["why"])}</div>' if b["why"] else ""
            st.markdown(
                f'<div class="branch{" best" if best else ""}" style="--c:{colors.get(b["support"], "var(--muted)")}">'
                f'<div class="h">Branch {letter}<span class="tag">{b["support"]} support</span>'
                f'{"<span class=tag star>★ chosen</span>" if best else ""}</div>'
                f'<div>{esc(b["text"])}</div>{why}</div>', unsafe_allow_html=True)
        if tree["best_why"]:
            st.markdown(f'<div class="verdict"><b>Why branch {tree["best"]}:</b> {esc(tree["best_why"])}</div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div class="final"><div class="lbl">Final answer</div>{esc(tree["final"])}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(answer)


def show_result(result, question=""):
    show_answer(result, question)
    if result.get("sources"):
        with st.expander(f"📚 Sources ({len(result['sources'])})"):
            for i, doc in enumerate(result["sources"], start=1):
                meta = doc.metadata
                title = meta.get("source", "unknown")
                where = "".join([f" · page {meta['page']}" if meta.get("page") else "",
                                 f" · {meta['section']}" if meta.get("section") else ""])
                link = f" — [open]({meta['url']})" if meta.get("url") else ""
                score = f" · score {meta['score']}" if "score" in meta else ""
                st.markdown(f"**[{i}] {title}**{where}{link}{score}")
                st.caption(doc.page_content[:400] + "...")
    with st.expander("🧠 What the agent did"):
        for step in result["trace"]:
            st.markdown(f"- {step}")
    st.caption(f"⏱ {result['seconds']} s")


def show_turn(base, result, question=""):
    if base:
        left, right = st.columns(2)
        with left:
            st.subheader("❌ Plain LLM (no RAG)")
            st.markdown(base["answer"])
            st.caption(f"⏱ {base['seconds']} s · no sources")
        with right:
            st.subheader("✅ Our agentic RAG")
            show_result(result, question)
    else:
        show_result(result, question)


question = st.chat_input("Ask anything about Agentic RAG...")

with center:
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            show_turn(item.get("base"), item["result"], item["question"])

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                base = None
                loading = st.empty()
                loading.markdown(LOADER, unsafe_allow_html=True)
                if compare:
                    base = agent.ask_base_llm(question)
                result = agent.ask(question, style, k, search_mode)
                loading.empty()
                show_turn(base, result, question)
                st.session_state.history.append({"question": question, "base": base, "result": result})
            except Exception as error:
                st.error(f"Something went wrong: {type(error).__name__}: {error}. "
                         "Check the internet connection and the API key, then try again.")
