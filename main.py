from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


# macOS: torch + faiss can load two OpenMP runtimes; without this the process may abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


APP_TITLE = "NewsURL — News Research"
PERSIST_DIR = Path(".cache/faiss_news_index")


def load_dotenv_simple(path: Path) -> None:
    """Minimal .env reader to avoid python-dotenv dependency in the tf env."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


def looks_like_google_api_key(k: str) -> bool:
    k = (k or "").strip()
    # AI Studio keys are typically ~39 chars and start with AIza
    return bool(k) and k.startswith("AIza") and len(k) >= 35


def normalize_urls(multiline: str) -> list[str]:
    urls: list[str] = []
    for line in multiline.splitlines():
        s = line.strip()
        if not s:
            continue
        urls.append(s)
    # preserve order, de-dupe
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def format_source_label(url: str) -> str:
    # cheap pretty label
    m = re.match(r"^https?://([^/]+)/?(.*)$", url.strip())
    if not m:
        return url
    host = m.group(1)
    path = m.group(2).strip("/")
    if not path:
        return host
    return f"{host}/{path[:48]}{'…' if len(path) > 48 else ''}"


@dataclass(frozen=True)
class BuildStats:
    urls: list[str]
    pages: int
    chunks: int
    built_at: str
    mode: str  # "gemini" | "local"


def get_google_api_key() -> str:
    # Priority: Streamlit secrets > environment variables > (optional) .env loader above
    # Streamlit Community Cloud injects secrets via `st.secrets`.
    secrets_key = ""
    try:
        secrets_key = (st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY") or "").strip()
    except Exception:
        secrets_key = ""

    env_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    return secrets_key or env_key


def get_embeddings(mode: str, google_api_key: str):
    if mode == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=google_api_key,
        )
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(urls: list[str], *, chunk_size: int, chunk_overlap: int, embeddings_mode: str, google_api_key: str):
    loader = WebBaseLoader(urls, continue_on_failure=True)
    pages = loader.load()
    if not pages or all(not (d.page_content or "").strip() for d in pages):
        raise ValueError("No readable content loaded from the provided URLs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("Loaded pages produced 0 text chunks. Try different URLs.")

    embeddings = get_embeddings(embeddings_mode, google_api_key)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, len(pages), len(chunks), embeddings_mode


def save_vectorstore(vs: FAISS) -> None:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(PERSIST_DIR))


def load_vectorstore(embeddings_mode: str, google_api_key: str) -> FAISS | None:
    if not PERSIST_DIR.is_dir():
        return None
    embeddings = get_embeddings(embeddings_mode, google_api_key)
    return FAISS.load_local(str(PERSIST_DIR), embeddings)


def retrieve_sources(docs: Iterable) -> list[str]:
    srcs: list[str] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("url")
        if src and src not in srcs:
            srcs.append(src)
    return srcs


def answer_from_evidence_only(*, question: str, retrieved_docs: list) -> str:
    """
    Best-effort extractive fallback (no external LLM call).

    This intentionally avoids hallucinating: it only extracts/snippets from retrieved evidence.
    """
    q = (question or "").strip().lower()
    if not retrieved_docs:
        return "No evidence retrieved for this question."

    combined = "\n".join((d.page_content or "") for d in retrieved_docs[:6])
    # Cricket-specific heuristic for common query here: "playing 11 of <team>"
    if any(k in q for k in ("playing 11", "playing xi", "playing eleven", "playing11")):
        # Try to locate "Royal Challengers Bengaluru" roster chunk from the evidence
        m = re.search(
            r"Royal Challengers Bengaluru\s*(.*?)(?:\n|$)",
            combined,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            line = m.group(0)
            # Extract a comma-separated list of names appearing after the team label
            after = re.split(r"Royal Challengers Bengaluru", line, flags=re.IGNORECASE)[-1]
            names = [n.strip() for n in after.split(",") if n.strip()]
            if names:
                # The BBC page is listing the squad; a playing XI isn't guaranteed to be present.
                top = names[:11]
                return (
                    "The retrieved article evidence appears to list the **RCB squad**, not a confirmed match playing XI.\n\n"
                    "**First 11 names shown in the squad list (not confirmed XI):**\n"
                    + "\n".join(f"- {n}" for n in top)
                    + "\n\nIf you share the specific match (date/opponent) or add a match preview URL, I can retrieve the confirmed XI from that source."
                )
        return (
            "I don’t see a confirmed playing XI in the retrieved evidence.\n\n"
            "The article chunk shown looks like a **squad list** for the season rather than a match lineup."
        )

    # Generic fallback: show the most relevant snippet from the top document
    best = retrieved_docs[0]
    txt = (best.page_content or "").strip()
    if not txt:
        return "Top retrieved document had no text content."
    snippet = txt[:900] + ("…" if len(txt) > 900 else "")
    return f"Evidence-only answer (no Gemini call):\n\n{snippet}"


def answer_with_gemini(*, question: str, retrieved_docs: list, google_api_key: str, temperature: float) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    context = "\n\n".join(d.page_content for d in retrieved_docs[:8])
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=temperature,
        max_output_tokens=700,
    )
    msg = HumanMessage(
        content=(
            "You are a news research assistant.\n"
            "Answer using ONLY the context. If the answer is not present, say you don't know.\n"
            "If helpful, quote small snippets.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
    )
    out = llm.invoke([msg])
    return getattr(out, "content", str(out))


st.set_page_config(page_title=APP_TITLE, page_icon="🗞️", layout="wide")

st.markdown(
    """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
  .stMetric { background: rgba(255,255,255,0.04); padding: 0.8rem 0.9rem; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); }
  .rb-card { background: rgba(255,255,255,0.04); padding: 1rem 1.1rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); }
  .rb-muted { opacity: 0.75; }
</style>
""",
    unsafe_allow_html=True,
)


root = Path(__file__).resolve().parent
load_dotenv_simple(root / ".env")
load_dotenv_simple(root.parent / ".env")

google_api_key = get_google_api_key()
has_real_key = looks_like_google_api_key(google_api_key)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "build_stats" not in st.session_state:
    st.session_state.build_stats = None
if "embeddings_mode" not in st.session_state:
    st.session_state.embeddings_mode = "gemini" if has_real_key else "local"
if "urls_text" not in st.session_state:
    st.session_state.urls_text = ""


with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.caption("Paste URLs, build an index, then ask questions with citations.")

    if not has_real_key:
        st.warning(
            "No valid `GOOGLE_API_KEY` / `GEMINI_API_KEY` detected.\n\n"
            "You can still build a local (offline) index using sentence-transformers, "
            "but Gemini answers won’t run until you add a real key."
        )

    st.divider()
    st.subheader("1) Sources")
    st.session_state.urls_text = st.text_area(
        "News article URLs (one per line)",
        value=st.session_state.urls_text,
        height=160,
        placeholder="https://...\nhttps://...",
        label_visibility="collapsed",
    )
    urls = normalize_urls(st.session_state.urls_text)
    if urls:
        st.caption(f"{len(urls)} URL(s) queued")
    else:
        st.caption("No URLs yet")

    st.subheader("2) Index settings")
    c1, c2 = st.columns(2)
    with c1:
        chunk_size = st.number_input("Chunk size", min_value=300, max_value=3000, value=1000, step=100)
    with c2:
        chunk_overlap = st.number_input("Overlap", min_value=0, max_value=500, value=150, step=10)

    embeddings_mode = st.selectbox(
        "Embeddings",
        options=["gemini", "local"],
        index=0 if st.session_state.embeddings_mode == "gemini" else 1,
        help="Gemini embeddings require a real key. Local uses sentence-transformers.",
    )
    if embeddings_mode == "gemini" and not has_real_key:
        st.info("Gemini embeddings selected but no key found — switching to local for this run.")
        embeddings_mode = "local"

    st.session_state.embeddings_mode = embeddings_mode

    build_clicked = st.button("Build / Refresh Index", type="primary", use_container_width=True)
    clear_clicked = st.button("Clear index", use_container_width=True)

    if clear_clicked:
        st.session_state.vectorstore = None
        st.session_state.build_stats = None
        if PERSIST_DIR.is_dir():
            # Best-effort cleanup
            for p in PERSIST_DIR.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass

    if build_clicked:
        if not urls:
            st.error("Add at least one URL first.")
        else:
            try:
                prog = st.progress(0, text="Loading pages…")
                vs, pages, chunks, mode = build_vectorstore(
                    urls,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings_mode=embeddings_mode,
                    google_api_key=google_api_key,
                )
                prog.progress(70, text="Saving index…")
                save_vectorstore(vs)
                prog.progress(100, text="Done")
                st.session_state.vectorstore = vs
                st.session_state.build_stats = BuildStats(
                    urls=urls,
                    pages=pages,
                    chunks=chunks,
                    built_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    mode=mode,
                )
                st.success(f"Index ready ({pages} page(s), {chunks} chunk(s)).")
            except Exception as exc:
                st.session_state.vectorstore = None
                st.session_state.build_stats = None
                st.error(f"Index build failed: {exc}")

    st.divider()
    st.subheader("Index status")
    if st.session_state.build_stats:
        bs: BuildStats = st.session_state.build_stats
        st.write(f"- **Mode**: `{bs.mode}`")
        st.write(f"- **Pages**: {bs.pages}")
        st.write(f"- **Chunks**: {bs.chunks}")
        st.write(f"- **Updated**: {bs.built_at}")
    else:
        st.write("- Not built yet")


st.markdown(f"## {APP_TITLE}")
st.caption("Fast RAG over news URLs — retrieve first, then (optionally) ask Gemini.")

top = st.columns([1.4, 1, 1, 1])
with top[0]:
    st.markdown('<div class="rb-card"><b>Workflow</b><div class="rb-muted">Add URLs → build index → ask questions</div></div>', unsafe_allow_html=True)
with top[1]:
    st.metric("Key detected", "Yes" if has_real_key else "No")
with top[2]:
    st.metric("Index", "Ready" if st.session_state.build_stats else "Not built")
with top[3]:
    st.metric("Embeddings", st.session_state.embeddings_mode)


tab_ask, tab_sources = st.tabs(["Ask", "Sources"])

with tab_sources:
    bs = st.session_state.build_stats
    if not bs:
        st.info("Build an index to see loaded sources.")
    else:
        st.markdown("### Loaded URLs")
        for u in bs.urls:
            st.markdown(f"- [{format_source_label(u)}]({u})")


with tab_ask:
    left, right = st.columns([2, 1])
    with right:
        st.markdown("### Answer settings")
        k = st.slider("Top-K chunks", min_value=2, max_value=12, value=6, step=1)
        temperature = st.slider("Gemini temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        prefer_evidence_only = st.toggle(
            "Evidence-only mode",
            value=not has_real_key,
            help="Skip Gemini and answer only from retrieved snippets.",
        )
        st.caption("If no key is set, the app will only show retrieved chunks.")

    with left:
        question = st.text_input(
            "Ask a question about the indexed articles",
            placeholder="e.g. What are the key takeaways and numbers mentioned across these articles?",
        )
        go = st.button("Answer", type="primary", disabled=not bool(question))

        if go:
            vs = st.session_state.vectorstore
            if vs is None:
                # try loading from disk
                vs = load_vectorstore(st.session_state.embeddings_mode, google_api_key)
                st.session_state.vectorstore = vs

            if vs is None:
                st.warning("No index found. Build the index in the sidebar first.")
            else:
                with st.spinner("Retrieving the most relevant chunks…"):
                    retriever = vs.as_retriever(search_kwargs={"k": k})
                    retrieved_docs = retriever.get_relevant_documents(question)

                st.markdown("### Retrieved evidence")
                for i, d in enumerate(retrieved_docs[: min(k, 8)], start=1):
                    src = (d.metadata or {}).get("source") or ""
                    title = format_source_label(src) if src else "Source"
                    with st.expander(f"{i}. {title}", expanded=(i <= 2)):
                        st.write(d.page_content[:1800])
                        if src:
                            st.markdown(f"**URL**: [{src}]({src})")

                st.markdown("### Answer")
                if prefer_evidence_only or not (has_real_key and st.session_state.embeddings_mode == "gemini"):
                    st.info(answer_from_evidence_only(question=question, retrieved_docs=retrieved_docs))
                else:
                    with st.spinner("Asking Gemini…"):
                        try:
                            answer = answer_with_gemini(
                                question=question,
                                retrieved_docs=retrieved_docs,
                                google_api_key=google_api_key,
                                temperature=temperature,
                            )
                            st.success(answer)
                        except Exception as exc:
                            msg = str(exc)
                            # If Gemini rate-limits/quota-exhausts, show a more helpful message and fall back.
                            retry_s = None
                            m = re.search(r"retry in\\s+([0-9]+(?:\\.[0-9]+)?)s", msg, flags=re.IGNORECASE)
                            if m:
                                try:
                                    retry_s = float(m.group(1))
                                except Exception:
                                    retry_s = None

                            st.error(f"Gemini request failed: {exc}")
                            if retry_s:
                                c1, c2 = st.columns([1, 2])
                                with c1:
                                    if st.button(f"Retry in ~{int(retry_s)}s", type="secondary"):
                                        with st.spinner(f"Waiting {int(retry_s)}s…"):
                                            time.sleep(max(0, int(retry_s)))
                                        st.rerun()
                                with c2:
                                    st.caption(
                                        "This looks like a quota/rate-limit. You can also switch on Evidence-only mode to avoid LLM calls."
                                    )
                            st.info(answer_from_evidence_only(question=question, retrieved_docs=retrieved_docs))

                srcs = retrieve_sources(retrieved_docs)
                if srcs:
                    st.markdown("### Sources")
                    for u in srcs:
                        st.markdown(f"- [{format_source_label(u)}]({u})")




