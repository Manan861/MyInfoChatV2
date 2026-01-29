"""
Resume RAG System - v3
Fixes: Metadata extraction reliability, pronoun resolution, resume counting
Using GPT-4o for better accuracy
"""

import os
import json
import re
import chromadb
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import datetime

load_dotenv()

# Setup
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vec_db = chromadb.PersistentClient(path="./chroma_db")
collection = vec_db.get_or_create_collection("resumes")


def extract_metadata(text: str, filename: str) -> dict:
    """Extract author name and summary from resume using LLM - with fallback strategies."""

    # Strategy 1: Try LLM extraction with strict formatting
    try:
        response = llm.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You extract information from resumes. Return ONLY valid JSON, no markdown, no explanation."
            }, {
                "role": "user",
                "content": f"""Extract the candidate's full name from this resume text.

RESUME TEXT (first 2000 chars):
{text[:2000]}

Return ONLY this JSON format:
{{"name": "FirstName LastName", "summary": "One sentence about their background", "skills": ["skill1", "skill2", "skill3"]}}

IMPORTANT: 
- The name is usually at the very top of the resume
- Look for the largest/first text that appears to be a person's name
- Do NOT return "Unknown" - find the actual name"""
            }],
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        meta = json.loads(content)

        if meta.get("name") and meta["name"].lower() not in ["unknown", "n/a", "not found", ""]:
            meta["filename"] = filename
            return meta

    except Exception as e:
        print(f"LLM extraction failed: {e}")

    # Strategy 2: Rule-based fallback - look for name patterns in first few lines
    try:
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if not line or '@' in line or 'http' in line.lower():
                continue
            if any(char.isdigit() for char in line) and len(line) < 30:
                continue
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(word[0].isupper() and word.isalpha() for word in words if len(word) > 1):
                    return {
                        "name": line,
                        "summary": "Resume uploaded",
                        "skills": [],
                        "filename": filename
                    }
    except:
        pass

    # Strategy 3: Use filename as last resort
    name_from_file = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    return {
        "name": name_from_file,
        "summary": "Resume uploaded",
        "skills": [],
        "filename": filename
    }


def chunk_text(text: str, size: int = 250, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def get_all_candidates() -> list[str]:
    """Get list of all unique candidate names in database."""
    if collection.count() == 0:
        return []
    try:
        results = collection.get(include=["metadatas"])
        names = set()
        for m in results["metadatas"]:
            name = m.get("name", "").strip()
            if name and name.lower() != "unknown":
                names.add(name)
        return sorted(list(names))
    except:
        return []


def get_resume_count() -> int:
    """Get actual number of unique resumes (by filename)."""
    if collection.count() == 0:
        return 0
    try:
        results = collection.get(include=["metadatas"])
        filenames = set(m.get("filename", "") for m in results["metadatas"])
        return len([f for f in filenames if f])
    except:
        return 0


def rewrite_query(query: str, history: list, names: list) -> str:
    """Rewrite query for better retrieval - handles pronouns, context, abbreviations."""

    recent_context = ""
    last_person_mentioned = None

    if history:
        recent = history[-5:]
        context_parts = []
        for h in recent:
            context_parts.append(f"Q: {h['q']}\nA: {h['a'][:300]}")
            for name in names:
                if name.lower() in h['a'].lower() or name.lower() in h['q'].lower():
                    last_person_mentioned = name
        recent_context = "Recent conversation:\n" + "\n".join(context_parts)

    try:
        response = llm.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": """You rewrite queries for semantic search over a resume database.

Your job:
1. Replace ALL pronouns (he/she/they/his/her/their/him/them) with the actual person's name from context
2. Replace references like "this person", "that candidate", "the same person" with actual names
3. Expand abbreviations (SWE→software engineering, ML→machine learning, PM→product manager)
4. Keep the original intent and all important keywords

Return ONLY the rewritten query, nothing else."""
            }, {
                "role": "user",
                "content": f"""Candidates in database: {', '.join(names) if names else 'Unknown'}

{recent_context}

{f"Note: The last person discussed was {last_person_mentioned}" if last_person_mentioned else ""}

Original query: "{query}"

Rewritten query:"""
            }],
            temperature=0.1,
            max_tokens=100
        )
        rewritten = response.choices[0].message.content.strip().strip('"').strip("'")
        return rewritten
    except:
        return query


def search_resumes(query: str, question: str, person_filter: str = None) -> tuple[str, list]:
    """Hybrid search: semantic + keyword matching."""
    results_docs = []
    results_meta = []

    where = None
    if person_filter and person_filter != "All":
        where = {"name": person_filter}

    # 1. Semantic search
    try:
        semantic = collection.query(
            query_texts=[query],
            n_results=15,
            where=where,
            include=["documents", "metadatas"]
        )
        if semantic['documents'] and semantic['documents'][0]:
            for i, doc in enumerate(semantic['documents'][0]):
                if doc not in results_docs:
                    results_docs.append(doc)
                    results_meta.append(semantic['metadatas'][0][i])
    except Exception as e:
        print(f"Semantic search error: {e}")

    # 2. Keyword search - extract meaningful words from question
    all_words = question.lower().split() + query.lower().split()
    keywords = list(set([w.strip('?.,!') for w in all_words if len(w) > 3]))

    try:
        all_docs = collection.get(include=["documents", "metadatas"])

        for i, doc in enumerate(all_docs['documents']):
            if person_filter and person_filter != "All":
                if all_docs['metadatas'][i].get('name') != person_filter:
                    continue

            doc_lower = doc.lower()
            if any(kw in doc_lower for kw in keywords):
                if doc not in results_docs:
                    results_docs.append(doc)
                    results_meta.append(all_docs['metadatas'][i])

    except Exception as e:
        print(f"Keyword search error: {e}")

    # Build labeled context
    labeled_chunks = []
    for i, doc in enumerate(results_docs[:15]):
        name = results_meta[i].get('name', 'Unknown') if i < len(results_meta) else 'Unknown'
        labeled_chunks.append(f"[{name}'s resume]:\n{doc}")

    context = "\n\n---\n\n".join(labeled_chunks)
    return context, results_meta[:15]


def get_answer(question: str, context: str, history: list, all_names: list = None) -> str:
    """Generate answer with strict accuracy rules using GPT-4o."""
    current_date = datetime.datetime.now().strftime("%B %d, %Y")

    history_text = ""
    if history:
        recent = history[-5:]
        history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(
            [f"User: {h['q']}\nAssistant: {h['a'][:300]}" for h in recent])

    names_list = f"\n\nCANDIDATES IN DATABASE ({len(all_names)} total): {', '.join(sorted(all_names))}" if all_names else ""

    context_section = f"RETRIEVED RESUME CHUNKS:\n{context}" if context.strip() else "NO RELEVANT CHUNKS FOUND"

    response = llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""You are a helpful assistant answering questions about resumes in a database.

CURRENT DATE: {current_date}

STRICT ACCURACY RULES:
1. For questions about how many resumes or listing all candidates → Use the CANDIDATES IN DATABASE list (this is the complete source of truth)
2. Each chunk is labeled [Name's resume] → ONLY attribute information to that specific person
3. Before saying "X has skill Y" → Verify Y literally appears in X's chunk text
4. For "who has X" questions → Scan all chunks for the term X, list only people whose chunks contain it
5. If a skill/term appears in NO chunks → Say "None of the candidates have [X] mentioned in their resumes"
6. NEVER invent or assume skills/experience not explicitly stated in the chunks

PRONOUN HANDLING:
- Use conversation history to resolve "he/she/they/his/her/their"
- If unclear who the user means, ask for clarification

Be concise and accurate."""},
            {"role": "user",
             "content": f"{context_section}{names_list}{history_text}\n\nQUESTION: {question}"}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content


# ============== STREAMLIT APP ==============

st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .main-header { text-align: center; padding: 1rem 0; border-bottom: 2px solid #4A90A4; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📄 Resume RAG System")
st.caption("Ask questions about uploaded resumes using AI-powered search")
st.markdown('</div>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    candidates = get_all_candidates()
    resume_count = get_resume_count()
    chunk_count = collection.count()

    if chunk_count > 0:
        st.markdown("### 📊 Database Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", chunk_count)
        with col2:
            st.metric("Resumes", resume_count)

        st.markdown("### 👥 Candidates")
        if candidates:
            with st.container(height=150):
                for name in candidates:
                    st.markdown(f"• {name}")
        else:
            st.warning("No candidate names extracted")

        st.divider()

        if len(candidates) > 1:
            selected = st.selectbox("🔍 Filter by person", ["All"] + candidates)
        else:
            selected = "All"

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

    st.markdown("### 📤 Upload Resumes")
    files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True, label_visibility="collapsed")

    if files and st.button("⬆️ Process Uploads", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        for idx, f in enumerate(files):
            status.text(f"Processing {f.name}...")

            try:
                pdf = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)

                if not text.strip():
                    st.warning(f"⚠️ {f.name}: No text extracted")
                    continue

                meta = extract_metadata(text, f.name)

                chunks = chunk_text(text)
                if not chunks:
                    st.warning(f"⚠️ {f.name}: No chunks created")
                    continue

                chunk_metas = [{"name": meta["name"], "filename": f.name} for _ in chunks]
                ids = [f"{f.name}_{i}_{hash(chunk)}" for i, chunk in enumerate(chunks)]

                collection.add(
                    documents=chunks,
                    ids=ids,
                    metadatas=chunk_metas
                )

                st.success(f"✓ {meta['name']} ({len(chunks)} chunks)")

            except Exception as e:
                st.error(f"❌ {f.name}: {str(e)}")

            progress.progress((idx + 1) / len(files))

        status.empty()
        progress.empty()
        st.rerun()

# Main chat
if collection.count() > 0:
    for h in st.session_state.history:
        with st.chat_message("user", avatar="👤"):
            st.write(h["q"])
        with st.chat_message("assistant", avatar="🤖"):
            st.write(h["a"])

    if question := st.chat_input("Ask about the candidates..."):
        with st.chat_message("user", avatar="👤"):
            st.write(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                candidates = get_all_candidates()
                rewritten = rewrite_query(question, st.session_state.history, candidates)
                context, meta = search_resumes(rewritten, question, selected)
                answer = get_answer(question, context, st.session_state.history, candidates)

            st.write(answer)

            with st.expander("🔍 Debug Info"):
                st.markdown(f"**Original:** {question}")
                st.markdown(f"**Rewritten:** {rewritten}")
                st.markdown(f"**Filter:** {selected}")
                st.markdown(f"**Chunks found:** {len(meta)}")
                if meta:
                    sources = list(set(m.get('name', '?') for m in meta))
                    st.markdown(f"**Sources:** {', '.join(sources)}")

        st.session_state.history.append({"q": question, "a": answer})
        st.rerun()
else:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f0f2f6; border-radius: 10px;">
            <h2>👋 Welcome!</h2>
            <p>Upload resume PDFs in the sidebar to get started.</p>
            <p style="color: #666;">Example questions:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>Whose resumes do you have?</li>
                <li>Who has Python experience?</li>
                <li>Tell me about [Name]'s background</li>
                <li>What are his skills? (after discussing someone)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)