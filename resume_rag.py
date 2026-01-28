"""
Resume RAG System - v3
Fixes: Metadata extraction reliability, pronoun resolution, resume counting
"""

import os
import json
import re
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
    """Extract author name and summary from resume using LLM - with fallback strategies."""

    # Strategy 1: Try LLM extraction with strict formatting
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
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
        # Clean up potential markdown formatting
        content = content.replace("```json", "").replace("```", "").strip()

        meta = json.loads(content)

        # Validate we got a real name
        if meta.get("name") and meta["name"].lower() not in ["unknown", "n/a", "not found", ""]:
            meta["filename"] = filename
            return meta

    except Exception as e:
        print(f"LLM extraction failed: {e}")

    # Strategy 2: Rule-based fallback - look for name patterns in first few lines
    try:
        lines = text.split('\n')[:10]  # First 10 lines
        for line in lines:
            line = line.strip()
            # Skip empty lines, emails, phones, links
            if not line or '@' in line or 'http' in line.lower():
                continue
            if any(char.isdigit() for char in line) and len(line) < 30:
                continue  # Likely phone number
            # Check if it looks like a name (2-4 words, mostly letters)
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
    """
    Rewrite query for better retrieval.
    Handles: pronouns, coreference, abbreviations, context from history.
    """

    # Build recent context
    recent_context = ""
    last_person_mentioned = None

    if history:
        recent = history[-5:]
        context_parts = []
        for h in recent:
            context_parts.append(f"Q: {h['q']}\nA: {h['a'][:300]}")
            # Track last person mentioned
            for name in names:
                if name.lower() in h['a'].lower() or name.lower() in h['q'].lower():
                    last_person_mentioned = name
        recent_context = "Recent conversation:\n" + "\n".join(context_parts)

    # Quick pronoun check - if query has pronouns and we know who was discussed
    pronouns = ['he', 'his', 'him', 'she', 'her', 'hers', 'they', 'their', 'them', 'this person', 'that person']
    has_pronoun = any(
        f" {p} " in f" {query.lower()} " or query.lower().startswith(f"{p} ") or query.lower().endswith(f" {p}") for p
        in pronouns)

    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
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

        # Sanity check - if we had a pronoun but it's still there, force replace
        if has_pronoun and last_person_mentioned:
            for p in pronouns:
                pattern = re.compile(rf'\b{p}\b', re.IGNORECASE)
                if pattern.search(rewritten):
                    rewritten = pattern.sub(last_person_mentioned, rewritten)

        return rewritten
    except:
        # Fallback: simple pronoun replacement
        if has_pronoun and last_person_mentioned:
            result = query
            for p in pronouns:
                pattern = re.compile(rf'\b{p}\b', re.IGNORECASE)
                result = pattern.sub(last_person_mentioned, result)
            return result
        return query


def detect_skill_query(question: str) -> list[str]:
    """
    Detect if this is a skill/tool search and extract the skill terms.
    Returns list of skills to search for.
    """
    question_lower = question.lower()

    # Patterns that indicate skill search
    skill_patterns = [
        r"(?:does|do)\s+(?:anyone|anybody|any candidate|someone)\s+(?:have|know|use|work with)\s+(.+?)[\?]?$",
        r"(?:who|which candidate)\s+(?:has|knows|uses|works with|worked with)\s+(.+?)[\?]?$",
        r"(?:anyone|anybody)\s+(?:with|having)\s+(.+?)\s+(?:experience|skills?)[\?]?$",
        r"(?:experience|skills?)\s+(?:in|with)\s+(.+?)[\?]?$",
        r"(?:familiar|proficient)\s+(?:with|in)\s+(.+?)[\?]?$",
    ]

    for pattern in skill_patterns:
        match = re.search(pattern, question_lower)
        if match:
            skill_text = match.group(1).strip()
            # Split on common separators
            skills = re.split(r'[,/&]|\s+(?:and|or)\s+', skill_text)
            return [s.strip() for s in skills if s.strip()]

    return []


def search_resumes(query: str, question: str, person_filter: str = None) -> tuple[str, list]:
    """
    Hybrid search: semantic + keyword + skill-specific matching.
    Returns (context_string, metadata_list)
    """
    results_docs = []
    results_meta = []

    # Build filter
    where = None
    if person_filter and person_filter != "All":
        where = {"name": person_filter}

    # Check if this is a skill-specific query
    skill_terms = detect_skill_query(question)

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

    # 2. General keyword search
    keywords = [w.lower().strip('?.,!') for w in question.split() if len(w) > 3]
    rewritten_keywords = [w.lower().strip('?.,!') for w in query.split() if len(w) > 3]
    keywords = list(set(keywords + rewritten_keywords))

    # Remove common words that aren't useful for search
    stopwords = {'does', 'anyone', 'have', 'what', 'which', 'their', 'there', 'they', 'this', 'that', 'with', 'about',
                 'from', 'were', 'doing'}
    keywords = [k for k in keywords if k not in stopwords]

    # 3. EXHAUSTIVE skill search - scan ALL chunks if looking for specific skill
    try:
        all_docs = collection.get(include=["documents", "metadatas"])

        for i, doc in enumerate(all_docs['documents']):
            # Apply person filter if set
            if person_filter and person_filter != "All":
                if all_docs['metadatas'][i].get('name') != person_filter:
                    continue

            doc_lower = doc.lower()
            should_add = False

            # Priority 1: Exact skill term match (case-insensitive)
            if skill_terms:
                for skill in skill_terms:
                    if skill.lower() in doc_lower:
                        should_add = True
                        break

            # Priority 2: General keyword match
            if not should_add and any(kw in doc_lower for kw in keywords):
                should_add = True

            if should_add and doc not in results_docs:
                results_docs.append(doc)
                results_meta.append(all_docs['metadatas'][i])

    except Exception as e:
        print(f"Keyword search error: {e}")

    # Build labeled context - prioritize chunks with exact skill matches
    if skill_terms:
        # Sort to put skill-matching chunks first
        paired = list(zip(results_docs, results_meta))

        def has_skill(pair):
            doc_lower = pair[0].lower()
            return any(skill.lower() in doc_lower for skill in skill_terms)

        paired.sort(key=has_skill, reverse=True)
        results_docs, results_meta = zip(*paired) if paired else ([], [])
        results_docs, results_meta = list(results_docs), list(results_meta)

    labeled_chunks = []
    for i, doc in enumerate(results_docs[:15]):  # Increased to 15 for skill searches
        name = results_meta[i].get('name', 'Unknown') if i < len(results_meta) else 'Unknown'
        labeled_chunks.append(f"[{name}'s resume]:\n{doc}")

    context = "\n\n---\n\n".join(labeled_chunks)
    return context, results_meta[:15]


def normalize_date_in_query(question: str) -> str:
    """
    Normalize ambiguous date references.
    'Jan 2025' in a question asked in Jan 2025 likely means recently.
    """
    import datetime
    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    # Common month abbreviations
    months = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }

    # If question mentions a month without year, add current year context
    question_lower = question.lower()
    for month_name, month_num in months.items():
        if month_name in question_lower:
            # Check if there's already a year mentioned
            if not re.search(r'20\d{2}', question):
                # No year mentioned - assume current/recent context
                return question  # Keep as-is, let the LLM handle with current date context

    return question


def get_answer(question: str, context: str, history: list, all_names: list = None) -> str:
    """Generate answer with strict accuracy rules."""
    import datetime
    current_date = datetime.datetime.now().strftime("%B %d, %Y")

    history_text = ""
    if history:
        recent = history[-5:]
        history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(
            [f"User: {h['q']}\nAssistant: {h['a'][:300]}" for h in recent])

    names_list = f"\n\nCANDIDATES IN DATABASE ({len(all_names)} total): {', '.join(sorted(all_names))}" if all_names else ""

    # Build chunk verification summary
    chunk_summary = ""
    if context.strip():
        chunks = context.split('---')
        summaries = []
        for chunk in chunks:
            if "[" in chunk and "'s resume]" in chunk:
                start = chunk.find("[") + 1
                end = chunk.find("'s resume]")
                name = chunk[start:end].strip()
                content_start = chunk.find("]:") + 2 if "]:" in chunk else 0
                preview = chunk[content_start:content_start + 200].strip().replace('\n', ' ')
                summaries.append(f"• {name}: {preview}...")
        if summaries:
            chunk_summary = "\n\nCHUNK PREVIEW (verify before attributing):\n" + "\n".join(summaries)

    context_section = f"RETRIEVED RESUME CHUNKS:\n{context}" if context.strip() else "NO RELEVANT CHUNKS FOUND"

    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are a helpful assistant answering questions about resumes in a database.

CURRENT DATE: January 2026

STRICT ACCURACY RULES:
1. For "whose resumes/how many/list all candidates" → Use the CANDIDATES IN DATABASE list (this is the source of truth)
2. Each chunk is labeled [Name's resume] → ONLY attribute information to that specific person
3. Before saying "X has skill Y" → Verify Y literally appears in X's chunk text
4. For "who has X" / "does anyone have X" → SCAN ALL chunks for the exact term X, list everyone who has it
5. If the skill/term appears NOWHERE in any chunk → Say "None of the candidates have [X] mentioned in their resumes"
6. NEVER invent or assume skills/experience not explicitly stated

SKILL/TOOL SEARCHES:
- When asked "does anyone have <specific experience>" → Look for that specific experience in ALL chunks
- Be thorough - check every chunk, not just the first few
- If you find it in someone's chunk, report it. If not in any chunk, say so clearly.

DATE HANDLING:
- Today is January 2026
- "Jan 2025" means January 2025 (about a year ago from now)
- "Recently" or "currently" means late 2025 / early 2026
- Resume dates like "Jan 2025 - Present" mean ongoing roles

PRONOUN HANDLING:
- If the user says "his/her/their" and you can tell from conversation history who they mean, answer about that person
- If unclear, ask for clarification

Be concise and direct. Don't apologize excessively."""},
            {"role": "user",
             "content": f"{context_section}{names_list}{chunk_summary}{history_text}\n\nQUESTION: {question}"}
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

    # Get accurate counts
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

        # Filter
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

    # Upload
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

                # Create metadata for each chunk
                chunk_metas = [{"name": meta["name"], "filename": f.name} for _ in chunks]

                # Generate unique IDs
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
                # Get fresh candidate list
                candidates = get_all_candidates()

                # Rewrite query
                rewritten = rewrite_query(question, st.session_state.history, candidates)

                # Search
                context, meta = search_resumes(rewritten, question, selected)

                # Answer
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