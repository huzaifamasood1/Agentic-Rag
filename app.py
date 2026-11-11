"""
app.py - The simple web UI (Streamlit).

Run with:   streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

import agent
import config
import ingest
import prompts
import reasoning
import vector_store

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG Research Assistant")
st.caption("Ask anything about Agentic RAG. The agent decides whether to search, rewrites your "
           "question, picks a source (our notes, arXiv, Wikipedia, code) and searches again if needed.")



# ---- Sidebar: settings + adding knowledge -----------------------------------------
with st.sidebar:
    st.header("Settings")
    style = st.selectbox("Prompt style", list(prompts.ANSWER_PROMPTS))
    search_mode = st.selectbox("Search mode (our notes)", vector_store.SEARCH_MODES,
                               help="similarity = top-k nearest | threshold = only relevant "
                                    "chunks | mmr = relevant and diverse")
    k = st.slider("k (chunks to retrieve)", 1, 10, config.DEFAULT_K)
    compare = st.toggle("Compare with plain LLM (before/after)", value=False)

    st.divider()
    st.header("Add knowledge")
    upload = st.file_uploader("PDF, image or text file",
                              type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md"])
    link = st.text_input("…or a web article / YouTube link")
    force = st.checkbox("Add even if it looks like a duplicate or off-topic")
    if (upload or link) and st.button("Add to database"):
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

    st.divider()
    st.metric("Chunks in database", vector_store.count_chunks())


# ---- Main area: chat -----------------------------------------------------------
def show_answer(result, question=""):
    """Zero/Few-shot: plain text. Chain of Thought: a chain of steps. Tree of Thoughts: a tree."""
    style, answer = result.get("style"), result["answer"]
    if style == "Chain of Thought" and (chain := reasoning.parse_chain(answer)):
        for number, (title, text) in enumerate(chain["steps"], start=1):
            with st.container(border=True):
                st.markdown(f"**Step {number} · {title}**")
                st.markdown(text)
            st.markdown("<div style='text-align:center'>⬇️</div>", unsafe_allow_html=True)
        st.success(chain["final"])
    elif style == "Tree of Thoughts" and (tree := reasoning.parse_tree(answer)):
        st.graphviz_chart(reasoning.tree_to_dot(question, tree), use_container_width=True)
        icons = {"strong": "🟢", "weak": "🟡", "none": "🔴"}
        for letter, branch in tree["branches"].items():
            star = " ⭐ **chosen**" if letter == tree["best"] else ""
            with st.expander(f"{icons.get(branch['support'], '⚪')} Branch {letter} · "
                             f"{branch['support']} support{star}", expanded=letter == tree["best"]):
                st.markdown(branch["text"])
                if branch["why"]:
                    st.caption(f"Support: {branch['why']}")
        if tree["best_why"]:
            st.info(f"**Why branch {tree['best']}:** {tree['best_why']}")
        st.success(tree["final"])
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


if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        show_turn(item.get("base"), item["result"], item["question"])

question = st.chat_input("Ask anything about Agentic RAG...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            base = None
            if compare:
                with st.spinner("Plain LLM is answering..."):
                    base = agent.ask_base_llm(question)
            with st.spinner("Planning, searching, answering..."):
                result = agent.ask(question, style, k, search_mode)
            show_turn(base, result, question)
            st.session_state.history.append({"question": question, "base": base, "result": result})
        except Exception as error:
            st.error(f"Something went wrong: {type(error).__name__}: {error}. "
                     "Check the internet connection and the API key, then try again.")
