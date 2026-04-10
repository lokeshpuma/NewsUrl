from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import streamlit as st
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup


# macOS: torch + faiss can load two OpenMP runtimes; without this the process may abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


APP_TITLE = "NewsURL — News Research"
PERSIST_DIR = Path(".cache/faiss_news_index")
PERSIST_DIR_INDEX = PERSIST_DIR / "index.faiss"
PERSIST_DIR_META = PERSIST_DIR / "meta.json"


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


def _fetch_article_text(url: str, timeout_s: int = 20) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # remove noisy tags
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Prefer main/article if present, else fallback to body
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        return []
    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out


def _embed_texts_local(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    arr = np.asarray(emb, dtype=np.float32)
    return arr


def _embed_texts_gemini(texts: list[str], google_api_key: str) -> np.ndarray:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=google_api_key)
    vecs = emb.embed_documents(texts)
    arr = np.asarray(vecs, dtype=np.float32)
    return arr


def _embed_query(text: str, *, mode: str, google_api_key: str) -> np.ndarray:
    if mode == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=google_api_key)
        v = np.asarray(emb.embed_query(text), dtype=np.float32)
        return v
    return _embed_texts_local([text])[0]


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


def build_vectorstore(urls: list[str], *, chunk_size: int, chunk_overlap: int, embeddings_mode: str, google_api_key: str):
    texts: list[str] = []
    metas: list[dict] = []

    for url in urls:
        txt = _fetch_article_text(url)
        if not txt.strip():
            continue
        chunks = _chunk_text(txt, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c in chunks:
            if c.strip():
                texts.append(c)
                metas.append({"source": url})

    if not texts:
        raise ValueError("No readable text extracted from the provided URLs.")

    if embeddings_mode == "gemini":
        vecs = _embed_texts_gemini(texts, google_api_key)
    else:
        vecs = _embed_texts_local(texts)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    meta = {"texts": texts, "metas": metas, "mode": embeddings_mode}
    return index, meta, len(urls), len(texts), embeddings_mode


def save_vectorstore(index: faiss.Index, meta: dict) -> None:
    import json

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(PERSIST_DIR_INDEX))
    PERSIST_DIR_META.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_vectorstore() -> tuple[faiss.Index, dict] | None:
    import json

    if not PERSIST_DIR_INDEX.is_file() or not PERSIST_DIR_META.is_file():
        return None
    index = faiss.read_index(str(PERSIST_DIR_INDEX))
    meta = json.loads(PERSIST_DIR_META.read_text(encoding="utf-8"))
    return index, meta


def search_index(index: faiss.Index, meta: dict, *, query: str, k: int, google_api_key: str) -> list[dict]:
    mode = meta.get("mode", "local")
    qv = _embed_query(query, mode=mode, google_api_key=google_api_key).astype(np.float32)
    D, I = index.search(np.expand_dims(qv, axis=0), k)
    hits: list[dict] = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        txt = meta["texts"][idx]
        m = meta["metas"][idx]
        hits.append({"score": float(score), "text": txt, "meta": m})
    return hits


def answer_from_evidence_only(*, question: str, retrieved_docs: list) -> str:
    """
    Best-effort extractive fallback (no external LLM call).

    This intentionally avoids hallucinating: it only extracts/snippets from retrieved evidence.
    """
    q = (question or "").strip().lower()
    if not retrieved_docs:
        return "No evidence retrieved for this question."

    combined = "\n".join((d.get("text") or "") for d in retrieved_docs[:6])
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
    txt = (best.get("text") or "").strip()
    if not txt:
        return "Top retrieved document had no text content."
    snippet = txt[:900] + ("…" if len(txt) > 900 else "")
    return f"Evidence-only answer (no Gemini call):\n\n{snippet}"


def answer_with_gemini(*, question: str, retrieved_docs: list, google_api_key: str, temperature: float) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    context = "\n\n".join(d.get("text", "") for d in retrieved_docs[:8])
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
if "vector_meta" not in st.session_state:
    st.session_state.vector_meta = None
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
        st.session_state.vector_meta = None
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
                index, meta, pages, chunks, mode = build_vectorstore(
                    urls,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings_mode=embeddings_mode,
                    google_api_key=google_api_key,
                )
                prog.progress(70, text="Saving index…")
                save_vectorstore(index, meta)
                prog.progress(100, text="Done")
                st.session_state.vectorstore = index
                st.session_state.vector_meta = meta
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
                st.session_state.vector_meta = None
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
            index = st.session_state.vectorstore
            meta = st.session_state.vector_meta
            if index is None or meta is None:
                loaded = load_vectorstore()
                if loaded:
                    index, meta = loaded
                    st.session_state.vectorstore = index
                    st.session_state.vector_meta = meta

            if index is None or meta is None:
                st.warning("No index found. Build the index in the sidebar first.")
            else:
                with st.spinner("Retrieving the most relevant chunks…"):
                    retrieved_docs = search_index(
                        index,
                        meta,
                        query=question,
                        k=k,
                        google_api_key=google_api_key,
                    )

                st.markdown("### Retrieved evidence")
                for i, d in enumerate(retrieved_docs[: min(k, 8)], start=1):
                    src = (d.get("meta") or {}).get("source") or ""
                    title = format_source_label(src) if src else "Source"
                    with st.expander(f"{i}. {title}", expanded=(i <= 2)):
                        st.write((d.get("text") or "")[:1800])
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

                srcs = []
                for d in retrieved_docs:
                    src = (d.get("meta") or {}).get("source") or ""
                    if src and src not in srcs:
                        srcs.append(src)
                if srcs:
                    st.markdown("### Sources")
                    for u in srcs:
                        st.markdown(f"- [{format_source_label(u)}]({u})")




