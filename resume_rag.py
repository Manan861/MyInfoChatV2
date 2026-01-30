"""
Resume RAG System - v4
Improvements: Scales to 40+ resumes, better specific question handling, improved accuracy
"""

import os
import json
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


def is_resume(text: str) -> tuple[bool, str]:
    """Check if the document appears to be a resume."""
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",  # Use mini for quick validation
            messages=[{
                "role": "system",
                "content": "You determine if a document is a resume/CV. Return ONLY 'YES' or 'NO' followed by a brief reason."
            }, {
                "role": "user",
                "content": f"Is this a resume/CV?\n\nTEXT:\n{text[:1500]}\n\nAnswer YES or NO with reason:"
            }],
            temperature=0.1,
            max_tokens=50
        )
        answer = response.choices[0].message.content.strip()
        return answer.upper().startswith("YES"), answer
    except:
        return True, "Could not verify"


def extract_metadata(text: str, filename: str) -> dict:
    """Extract candidate name from resume."""
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "Extract the person's full name from this resume. Return ONLY valid JSON with no markdown."
            }, {
                "role": "user",
                "content": f"""Resume text (first 2500 chars):
{text[:2500]}

Return JSON format: {{"name": "Full Name", "summary": "Brief summary"}}

RULES:
- Find the PERSON'S NAME, usually at the very top
- Names look like "John Smith", "Maria Garcia" 
- Do NOT return job titles like "Software Engineer", "Data Analyst"
- Do NOT return "Not Provided", "Unknown", "N/A"
- The name is typically near email/phone/LinkedIn info"""
            }],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        meta = json.loads(content)

        name = meta.get("name", "").strip()

        # Reject bad names
        bad_names = ["not provided", "unknown", "n/a", "name", "resume", "cv", ""]
        if name.lower() in bad_names:
            raise ValueError("Bad name extracted")

        meta["filename"] = filename
        return meta
    except Exception as e:
        print(f"Extraction failed: {e}")

    # Fallback: use filename but clean it up
    name = filename.replace('.pdf', '').replace('.PDF', '')
    name = name.replace('_', ' ').replace('-', ' ')
    # Remove common words
    for word in ['resume', 'Resume', 'RESUME', 'cv', 'CV', '2024', '2025', '2026']:
        name = name.replace(word, '')
    name = ' '.join(name.split()).strip()

    if not name:
        name = f"Candidate from {filename}"

    return {"name": name, "summary": "Resume", "filename": filename}


def chunk_text(text: str, size: int = 300, overlap: int = 75) -> list[str]:
    """Split text into overlapping chunks - slightly larger for better context."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def get_all_candidates() -> list[str]:
    """Get all unique candidate names."""
    if collection.count() == 0:
        return []
    try:
        results = collection.get(include=["metadatas"])
        names = set(m.get("name", "").strip() for m in results["metadatas"])
        return sorted([n for n in names if n and n.lower() != "unknown"])
    except:
        return []


def get_resume_count() -> int:
    """Get number of unique resumes."""
    if collection.count() == 0:
        return 0
    try:
        results = collection.get(include=["metadatas"])
        return len(set(m.get("filename", "") for m in results["metadatas"] if m.get("filename")))
    except:
        return 0


def rewrite_query(query: str, history: list, names: list) -> str:
    """Rewrite query - resolve pronouns and expand abbreviations."""
    if not history and not any(p in query.lower() for p in ['he', 'she', 'they', 'his', 'her', 'their', 'him', 'them']):
        return query  # No rewrite needed

    recent_context = ""
    last_person = None

    if history:
        recent = history[-5:]
        recent_context = "\n".join([f"Q: {h['q']}\nA: {h['a'][:200]}" for h in recent])
        for h in recent:
            for name in names:
                if name.lower() in h['a'].lower() or name.lower() in h['q'].lower():
                    last_person = name

    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "Rewrite the query replacing pronouns with actual names from context. Return ONLY the rewritten query."
            }, {
                "role": "user",
                "content": f"Candidates: {', '.join(names[:20])}\n\n{recent_context}\n\nLast discussed: {last_person}\n\nQuery: \"{query}\"\n\nRewritten:"
            }],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip().strip('"\'')
    except:
        return query


def search_resumes(query: str, question: str, person_filter: str = None, all_candidates: list = None) -> tuple[
    str, list]:
    """
    Search all chunks - full scan + semantic search.
    No query type detection - just find everything relevant.
    """
    results_docs = []
    results_meta = []

    where = {"name": person_filter} if person_filter and person_filter != "All" else None

    # Extract search terms from question
    stop_words = {'does', 'anyone', 'have', 'has', 'who', 'what', 'when', 'where', 'which', 'their', 'about',
                  'they', 'them', 'this', 'that', 'with', 'from', 'were', 'doing', 'experience', 'tell',
                  'know', 'knows', 'someone', 'the', 'and', 'for', 'are', 'can', 'could', 'would', 'me', 'if',
                  'all', 'list', 'show', 'give', 'get', 'find', 'search', 'look', 'any', 'some'}

    # Get all meaningful words from question
    all_words = question.lower().replace('?', '').replace('.', '').replace(',', '').split()
    search_terms = [w for w in all_words if len(w) > 2 and w not in stop_words]

    # Also from rewritten query
    query_words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()
    search_terms += [w for w in query_words if len(w) > 2 and w not in stop_words]
    search_terms = list(set(search_terms))

    print(f"DEBUG: search_terms={search_terms}")

    # ALWAYS do full scan for any search terms
    if search_terms:
        try:
            all_docs = collection.get(include=["documents", "metadatas"])
            print(f"DEBUG: Scanning {len(all_docs['documents'])} chunks")

            for i, doc in enumerate(all_docs['documents']):
                if where and all_docs['metadatas'][i].get('name') != person_filter:
                    continue
                doc_lower = doc.lower()

                # Check if ANY search term is in this chunk
                for term in search_terms:
                    if term in doc_lower:
                        if doc not in results_docs:
                            results_docs.append(doc)
                            results_meta.append(all_docs['metadatas'][i])
                        break

        except Exception as e:
            print(f"Full scan error: {e}")

    # Also do semantic search
    try:
        semantic = collection.query(
            query_texts=[query],
            n_results=50,
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

    # Ensure all candidates have at least one chunk (for broad queries)
    if all_candidates:
        candidates_in_results = set(m.get('name') for m in results_meta)
        missing = set(all_candidates) - candidates_in_results

        if missing:
            try:
                all_docs = collection.get(include=["documents", "metadatas"])
                for i, doc in enumerate(all_docs['documents']):
                    name = all_docs['metadatas'][i].get('name')
                    if name in missing:
                        results_docs.append(doc)
                        results_meta.append(all_docs['metadatas'][i])
                        missing.discard(name)
                    if not missing:
                        break
            except:
                pass

    # Build labeled context - prioritize chunks with search terms
    chunks_by_candidate = {}
    for i, doc in enumerate(results_docs):
        name = results_meta[i].get('name', 'Unknown')
        if name not in chunks_by_candidate:
            chunks_by_candidate[name] = []
        chunks_by_candidate[name].append(doc)

    labeled_chunks = []

    # First: chunks that contain search terms
    if search_terms:
        for name, chunks in chunks_by_candidate.items():
            for chunk in chunks:
                if len(labeled_chunks) >= 60:
                    break
                chunk_lower = chunk.lower()
                if any(term in chunk_lower for term in search_terms):
                    labeled_chunks.append(f"[{name}'s resume]:\n{chunk}")
            if len(labeled_chunks) >= 60:
                break

    # Then: one chunk per candidate (for coverage)
    for name, chunks in chunks_by_candidate.items():
        if len(labeled_chunks) >= 70:
            break
        chunk_text = f"[{name}'s resume]:\n{chunks[0]}"
        if chunk_text not in labeled_chunks:
            labeled_chunks.append(chunk_text)

    print(f"DEBUG: Returning {len(labeled_chunks)} chunks from {len(chunks_by_candidate)} candidates")
    return "\n\n---\n\n".join(labeled_chunks), results_meta


def get_answer(question: str, context: str, history: list, all_names: list = None) -> str:
    """Generate accurate answer using GPT-4o."""
    current_date = datetime.datetime.now().strftime("%B %d, %Y")

    # Build conversation history
    history_text = ""
    if history:
        history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join(
            [f"User: {h['q']}\nAssistant: {h['a'][:250]}" for h in history[-5:]]
        )

    # Build complete candidate list
    if all_names:
        candidate_list = f"""

COMPLETE DATABASE ({len(all_names)} CANDIDATES):
{chr(10).join([f"{i + 1}. {name}" for i, name in enumerate(sorted(all_names))])}

"""
    else:
        candidate_list = ""

    # Determine context status
    if context.strip():
        context_section = f"RESUME DATA:\n{context}"
    else:
        context_section = "NO MATCHING RESUME DATA FOUND"

    response = llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": f"""You are a resume database assistant helping recruiters find and evaluate candidates. Today's date is {current_date}.

You have two information sources:
1. COMPLETE DATABASE LIST - Shows ALL {len(all_names) if all_names else 0} candidates in the system (this is the authoritative count)
2. RESUME DATA - Contains actual resume text chunks, each labeled [Name's resume]

LISTING AND COUNTING:
- List all candidates, count resumes, show who is in the database
- Always use the COMPLETE DATABASE LIST for accurate counts
- Include the total number

SEARCHING FOR SKILLS, TOOLS, TECHNOLOGIES:
- Find candidates with specific skills, tools, languages, frameworks, or technologies
- Search ALL chunks and list EVERY matching candidate
- Describe each candidate's relevant experience with that skill
- If not found in any chunk, say so clearly

EXPERIENCE AND WORK HISTORY:
- Answer questions about job titles, roles, companies, industries
- Find who worked at specific companies or in specific roles
- Identify years of experience, career progression, job responsibilities
- Find candidates with specific types of experience (startup, enterprise, remote, etc.)

EDUCATION AND QUALIFICATIONS:
- Find candidates with specific degrees, majors, universities, certifications
- Identify educational background, GPA if mentioned, graduation dates
- Find candidates with specific training or courses

DATES AND TIME PERIODS:
- What were candidates doing at specific times
- Find current roles, recent experience, or historical positions
- Identify employment gaps if visible
- Find candidates available from certain dates

SPECIFIC CANDIDATE QUESTIONS:
- Detailed information about a particular person
- Their complete background, skills, experience, education
- Only use information from that person's chunks

COMPARING CANDIDATES:
- Compare two or more candidates side by side
- Highlight similarities and differences
- Compare skills, experience levels, education, qualifications
- Be objective and fair

RECOMMENDING AND RANKING:
- Who is best for a specific role, job, or project
- Rank candidates by relevance to requirements
- Explain reasoning with specific evidence from resumes
- Shortlist candidates meeting certain criteria

FILTERING AND SCREENING:
- Find candidates matching multiple criteria
- Filter by location, experience level, skills combination
- Identify candidates who meet minimum requirements
- Exclude candidates missing required qualifications

SUMMARIZING:
- Summarize individual resumes or groups of candidates
- Provide quick overviews or detailed breakdowns
- Highlight key strengths, notable achievements, unique qualities

ANALYZING PATTERNS:
- What skills are most common across candidates
- What companies or industries are represented
- What education backgrounds are present
- Identify trends or patterns in the candidate pool

IDENTIFYING GAPS AND STRENGTHS:
- What skills or experience is missing from the candidate pool
- What are the strongest areas of expertise available
- Which candidates have unique or rare skills

ANSWERING FOLLOW-UP QUESTIONS:
- Use conversation history to understand context
- Resolve pronouns (he, she, they, them) from previous messages
- Build on previous answers without repeating everything
- If unclear who or what is being referenced, ask for clarification

HANDLING AMBIGUITY:
- If a question could be interpreted multiple ways, address the most likely interpretation
- If you need more information to give a good answer, ask
- If a candidate name is misspelled or partial, try to match it

ACCURACY RULES:
1. Never invent or fabricate information
2. Never mix up information between candidates
3. Always attribute information to the correct person based on chunk labels
4. Include ALL matching candidates, not just a few
5. If unsure or cannot find information, say so clearly
6. Base all recommendations and comparisons only on resume content
7. Be honest about limitations of the information available

OFF-TOPIC:
For messages completely unrelated to resumes or hiring (jokes, weather, personal chat, general knowledge):
- Respond: "I'm here to help with questions about the resumes. What would you like to know about the candidates?"

Answer thoroughly and accurately based on the data provided."""},
            {"role": "user", "content": f"{context_section}{candidate_list}{history_text}\n\nQUESTION: {question}"}
        ],
        temperature=0,
        max_tokens=4000
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

st.title("📄 Resume RAG System")
st.caption("Ask questions about uploaded resumes using AI-powered search")

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    candidates = get_all_candidates()
    resume_count = get_resume_count()
    chunk_count = collection.count()

    if chunk_count > 0:
        st.markdown("### 📊 Stats")
        col1, col2 = st.columns(2)
        col1.metric("Resumes", resume_count)
        col2.metric("Chunks", chunk_count)

        st.markdown("### 👥 Candidates")
        with st.container(height=250):
            for name in candidates:
                st.markdown(f"• {name}")

        st.divider()

        selected = st.selectbox("🔍 Filter", ["All"] + candidates) if len(candidates) > 1 else "All"

        col1, col2 = st.columns(2)
        if col1.button("🗑️ Clear Data", use_container_width=True):
            vec_db.delete_collection("resumes")
            st.session_state.history = []
            st.rerun()
        if col2.button("💬 Clear Chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No resumes uploaded")
        selected = "All"

    st.divider()
    st.markdown("### 📤 Upload")
    files = st.file_uploader("PDFs", type=['pdf'], accept_multiple_files=True, label_visibility="collapsed")

    if files and st.button("⬆️ Process", use_container_width=True):
        progress = st.progress(0)

        for idx, f in enumerate(files):
            try:
                pdf = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)

                if not text.strip():
                    st.warning(f"⚠️ {f.name}: Empty")
                    continue

                is_valid, reason = is_resume(text)
                if not is_valid:
                    st.warning(f"⚠️ {f.name}: Not a resume")
                    continue

                meta = extract_metadata(text, f.name)
                chunks = chunk_text(text)

                if chunks:
                    collection.add(
                        documents=chunks,
                        ids=[f"{f.name}_{i}_{hash(c)}" for i, c in enumerate(chunks)],
                        metadatas=[{"name": meta["name"], "filename": f.name} for _ in chunks]
                    )
                    st.success(f"✓ {meta['name']}")
            except Exception as e:
                st.error(f"❌ {f.name}: {e}")

            progress.progress((idx + 1) / len(files))

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
            with st.spinner("Searching..."):
                candidates = get_all_candidates()
                rewritten = rewrite_query(question, st.session_state.history, candidates)
                context, meta = search_resumes(rewritten, question, selected, candidates)
                answer = get_answer(question, context, st.session_state.history, candidates)

            st.write(answer)

            with st.expander("🔍 Debug"):
                st.text(f"Original: {question}")
                st.text(f"Rewritten: {rewritten}")
                st.text(f"Chunks retrieved: {len(meta)}")
                if meta:
                    sources = list(set(m.get('name', '?') for m in meta))
                    st.text(f"Sources ({len(sources)}): {', '.join(sources[:15])}")
                    if len(sources) > 15:
                        st.text(f"... and {len(sources) - 15} more")

        st.session_state.history.append({"q": question, "a": answer})
        st.rerun()
else:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #1a1a2e; border-radius: 15px; border: 1px solid #4A90A4; max-width: 600px; margin: auto;">
        <h2 style="color: #4A90A4;">👋 Welcome!</h2>
        <p style="color: #fff;">Upload resume PDFs in the sidebar to get started.</p>
        <p style="color: #aaa;">Example questions:</p>
        <p style="color: #fff;">📋 "Whose resumes do you have?"</p>
        <p style="color: #fff;">🔍 "Who has Python experience?"</p>
        <p style="color: #fff;">📅 "What was John doing in Jan 2025?"</p>
        <p style="color: #fff;">📊 "Compare candidates for backend role"</p>
    </div>
    """, unsafe_allow_html=True)