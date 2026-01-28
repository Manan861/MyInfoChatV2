"""
Resume RAG System - Simplified v2
Features: Metadata extraction, conversation memory, query rewriting, multi-resume support
"""

import os
import json
import chromadb
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

# Setup
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vec_db = chromadb.PersistentClient(path="./chroma_db")
collection = vec_db.get_or_create_collection("resumes")


def extract_metadata(text: str, filename: str) -> dict:
    """Extract author name and summary from resume using LLM."""
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Extract from this resume (return ONLY JSON):
{text[:1500]}

{{"name": "Full Name", "summary": "One sentence summary", "skills": ["skill1", "skill2", "skill3"]}}"""
            }],
            temperature=0.1
        )
        meta = json.loads(response.choices[0].message.content)
        meta["filename"] = filename
        return meta
    except:
        return {"name": "Unknown", "summary": "Resume", "skills": [], "filename": filename}


def chunk_text(text: str, size: int = 250, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks. Smaller chunks = better precision for many resumes."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def rewrite_query(query: str, history: list, names: list) -> str:
    """Rewrite query for better retrieval using conversation context."""
    recent_context = ""
    if history:
        recent = history[-5:]
        recent_context = "Recent conversation:\n" + "\n".join([f"Q: {h['q']}\nA: {h['a'][:200]}" for h in recent])

    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Rewrite this query for semantic search.

RULES:
1. Replace "he/she/they" with actual names from context
2. Replace "it/that/this" with the actual topic from recent conversation  
3. Expand abbreviations (SWE = software engineering)
4. Keep the full meaning - don't lose important words

People in database: {', '.join(names) if names else 'Unknown'}

{recent_context}

Original query: "{query}"

Rewritten query (return ONLY the query, nothing else):"""
            }],
            temperature=0.1,
            max_tokens=60
        )
        return response.choices[0].message.content.strip().strip('"')
    except:
        return query


def search_resumes(query: str, question: str, person_filter: str = None) -> tuple[str, list]:
    """
    Hybrid search: semantic + keyword matching.
    Returns (context_string, metadata_list)
    Supports up to 20 resumes.
    """
    results_docs = []
    results_meta = []

    # Build person filter
    where = {"name": person_filter} if person_filter and person_filter != "All" else None

    # 1. Semantic search - get more results to cover multiple resumes
    try:
        semantic = collection.query(
            query_texts=[query],
            n_results=15,  # Increased for multi-resume support
            where=where,
            include=["documents", "metadatas"]
        )
        if semantic['documents'] and semantic['documents'][0]:
            for i, doc in enumerate(semantic['documents'][0]):
                if doc not in results_docs:
                    results_docs.append(doc)
                    results_meta.append(semantic['metadatas'][0][i])
    except:
        pass

    # 2. Keyword search - find exact word matches
    keywords = [w.lower().strip('?.,!') for w in question.split() if len(w) > 3]

    try:
        all_docs = collection.get(include=["documents", "metadatas"])
        for i, doc in enumerate(all_docs['documents']):
            # Apply person filter if set
            if person_filter and person_filter != "All":
                if all_docs['metadatas'][i].get('name') != person_filter:
                    continue

            doc_lower = doc.lower()
            if any(kw in doc_lower for kw in keywords):
                if doc not in results_docs:
                    results_docs.append(doc)
                    results_meta.append(all_docs['metadatas'][i])
    except:
        pass

    # Build context with person labels - increased limit for more resumes
    labeled_chunks = []
    for i, doc in enumerate(results_docs[:12]):  # Increased from 6 to 12
        name = results_meta[i].get('name', 'Unknown') if i < len(results_meta) else 'Unknown'
        labeled_chunks.append(f"[{name}'s resume]:\n{doc}")

    context = "\n\n---\n\n".join(labeled_chunks)
    return context, results_meta[:12]


def get_answer(question: str, context: str, history: list, all_names: list = None) -> str:
    """Generate answer from context with conversation history."""

    history_text = ""
    if history:
        recent = history[-5:]
        history_text = "\n\nRecent conversation:\n" + "\n".join([f"Q: {h['q']}\nA: {h['a'][:200]}" for h in recent])

    # Always provide full candidate list
    names_info = f"\n\nFULL DATABASE: {len(all_names)} candidates total: {', '.join(sorted(all_names))}" if all_names else ""

    # Build verification info - check what keywords from question appear in which chunks
    verification_info = ""
    if context.strip():
        chunks = context.split('---')
        chunk_summary = []
        for chunk in chunks:
            if "[" in chunk and "'s resume]" in chunk:
                start = chunk.find("[") + 1
                end = chunk.find("'s resume]")
                name = chunk[start:end].strip()
                # Get first 100 chars of content
                content_start = chunk.find("]:") + 2 if "]:" in chunk else 0
                preview = chunk[content_start:content_start + 150].strip().replace('\n', ' ')
                chunk_summary.append(f"- {name}: {preview}...")

        if chunk_summary:
            verification_info = "\n\nCHUNK SUMMARY (verify info exists before attributing):\n" + "\n".join(
                chunk_summary)

    context_section = f"RESUME CHUNKS:\n{context}" if context.strip() else "NO RELEVANT CHUNKS FOUND"

    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You answer questions about resumes in a database.

CRITICAL ACCURACY RULES:
1. The FULL DATABASE list shows ALL candidates - use this for "list all/how many/whose resumes" questions
2. Each chunk is labeled [Name's resume] - only attribute info to that specific person  
3. Before saying "X has Y skill" - verify Y actually appears in X's chunk
4. For "who has X" - only include people if X literally appears in their chunk text
5. If no chunks contain the requested info, say "none of the candidates have this mentioned"
6. Never invent or assume - only state what's explicitly in the chunks"""},
            {"role": "user",
             "content": f"{context_section}{names_info}{verification_info}{history_text}\n\nQUESTION: {question}"}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content


# ============== STREAMLIT APP ==============

st.set_page_config(
    page_title="Resume RAG",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #4A90A4;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📄 Resume RAG System")
st.caption("Ask questions about uploaded resumes using AI-powered search")
st.markdown('</div>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    authors = []
    if collection.count() > 0:
        try:
            results = collection.get(include=["metadatas"])
            authors = list(set(m.get("name", "Unknown") for m in results["metadatas"]))
        except:
            pass

    if collection.count() > 0:
        # Stats
        st.markdown("### 📊 Database Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", collection.count())
        with col2:
            st.metric("Resumes", len(authors))

        # List authors (scrollable for many)
        st.markdown("### 👥 Candidates")
        with st.container(height=150):
            for author in sorted(authors):
                st.markdown(f"• {author}")

        st.divider()

        # Filter
        if len(authors) > 1:
            selected = st.selectbox("🔍 Filter by person", ["All"] + authors)
        else:
            selected = "All"

        # Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Data", use_container_width=True):
                vec_db.delete_collection("resumes")
                st.session_state.history = []
                st.rerun()
        with col2:
            if st.button("💬 Clear Chat", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("No resumes uploaded yet")
        selected = "All"

    st.divider()

    # Upload section
    st.markdown("### 📤 Upload Resumes")
    files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True, label_visibility="collapsed")

    if files and st.button("⬆️ Process Uploads", use_container_width=True):
        with st.spinner("Processing..."):
            for f in files:
                pdf = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() for page in pdf.pages)
                meta = extract_metadata(text, f.name)

                chunks = chunk_text(text)
                metadatas = [{"name": meta["name"], "filename": f.name} for _ in chunks]
                collection.add(
                    documents=chunks,
                    ids=[f"{f.name}_{i}" for i in range(len(chunks))],
                    metadatas=metadatas
                )
                st.success(f"✓ {meta['name']}")
        st.rerun()

# Main chat area
if collection.count() > 0:
    # Chat history
    for h in st.session_state.history:
        with st.chat_message("user", avatar="👤"):
            st.write(h["q"])
        with st.chat_message("assistant", avatar="🤖"):
            st.write(h["a"])

    # Chat input
    if question := st.chat_input("Ask about the candidates..."):

        with st.chat_message("user", avatar="👤"):
            st.write(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                # Rewrite query
                rewritten = rewrite_query(question, st.session_state.history, authors)

                # Search
                context, meta = search_resumes(rewritten, question, selected)

                # Get answer - always pass all authors so LLM knows everyone
                answer = get_answer(question, context, st.session_state.history, authors)

            st.write(answer)

            # Debug expander
            with st.expander("🔍 Debug Info"):
                st.markdown(f"**Original:** {question}")
                st.markdown(f"**Rewritten:** {rewritten}")
                st.markdown(f"**Filter:** {selected}")
                st.markdown(f"**Chunks found:** {len(meta)}")
                if meta:
                    names_in_results = list(set(m.get('name', '?') for m in meta))
                    st.markdown(f"**Sources:** {', '.join(names_in_results)}")

        st.session_state.history.append({"q": question, "a": answer})
        st.rerun()
else:
    # Empty state
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f0f2f6; border-radius: 10px;">
            <h2>👋 Welcome!</h2>
            <p>Upload resume PDFs in the sidebar to get started.</p>
            <p style="color: #666;">You can then ask questions like:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>Who has Python experience?</li>
                <li>Compare the candidates' education</li>
                <li>Who's best for a backend role?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)