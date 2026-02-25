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
import requests
import re
import html

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


# ============== TOOLS (Me mode: 3 tools = semantic search, web search about person, GitHub) ==============

def tool_web_search(candidate_name: str, question: str, for_weather: bool = False) -> str:
    """Web search: only for (1) info about the person or (2) weather. Nothing else."""
    try:
        if for_weather:
            location = extract_weather_location(question)
            if not location:
                return "Please include a city for weather, for example: 'weather in Boston'."
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            geo.raise_for_status()
            geo_data = geo.json()
            results = geo_data.get("results") or []
            if not results:
                return f"Could not find location '{location}'. Try a city and state/country."

            place = results[0]
            lat, lon = place.get("latitude"), place.get("longitude")
            city = place.get("name") or location
            admin = place.get("admin1") or ""
            country = place.get("country") or ""

            weather = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                    "timezone": "auto",
                },
                timeout=10,
            )
            weather.raise_for_status()
            cur = (weather.json() or {}).get("current", {})
            if not cur:
                return f"No current weather data returned for {city}."

            code = int(cur.get("weather_code", -1))
            label = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Fog", 48: "Depositing rime fog",
                51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
                61: "Slight rain", 63: "Rain", 65: "Heavy rain",
                71: "Slight snow", 73: "Snow", 75: "Heavy snow",
                80: "Rain showers", 81: "Rain showers", 82: "Violent rain showers",
                95: "Thunderstorm",
            }.get(code, f"Code {code}")
            loc = ", ".join([x for x in [city, admin, country] if x])
            return (
                f"{loc}: {label}. "
                f"Temperature {cur.get('temperature_2m')}°C, "
                f"humidity {cur.get('relative_humidity_2m')}%, "
                f"wind {cur.get('wind_speed_10m')} km/h."
            )

        # Person web search path
        specific_terms = extract_search_focus(question, candidate_name)
        query = f"\"{candidate_name}\" {specific_terms}".strip()
        query_variants = [
            query,
            f"\"{candidate_name}\" linkedin",
            f"\"{candidate_name}\" github",
        ]
        lines = []
        seen = set()

        # Seed with profile links found in resume text (high-confidence identity anchors)
        profile_links = extract_profile_links_from_resume(candidate_name)
        for link in profile_links:
            if link not in seen:
                seen.add(link)
                lines.append(f"- Profile found in resume: {link}")

        try:
            from duckduckgo_search import DDGS
            for qv in query_variants:
                ddg_results = list(DDGS().text(qv[:220], max_results=5))
                for r in ddg_results:
                    title = (r.get("title") or "").strip()
                    body = (r.get("body") or "").strip()
                    href = (r.get("href") or "").strip()
                    dedupe_key = href or f"{title}-{body[:80]}"
                    if dedupe_key in seen:
                        continue
                    if title or body:
                        seen.add(dedupe_key)
                        lines.append(f"- {title}: {body[:220]} ({href})" if href else f"- {title}: {body[:220]}")
                if len(lines) >= 8:
                    break
        except Exception:
            pass

        if len(lines) < 3:
            # Fallback: scrape DDG html endpoint
            try:
                html_resp = requests.get(
                    "https://duckduckgo.com/html/",
                    params={"q": query},
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                html_resp.raise_for_status()
                for title, url in parse_ddg_html_results(html_resp.text, max_results=8):
                    dedupe_key = url or title
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    lines.append(f"- {title} ({url})" if url else f"- {title}")
                    if len(lines) >= 8:
                        break
            except Exception:
                pass

        if len(lines) < 3:
            # Fallback: DuckDuckGo instant answer API
            ia = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=10,
            )
            ia.raise_for_status()
            data = ia.json()
            abstract = (data.get("AbstractText") or "").strip()
            abstract_url = (data.get("AbstractURL") or "").strip()
            heading = (data.get("Heading") or candidate_name).strip()
            if abstract:
                lines.append(f"- {heading}: {abstract[:300]} ({abstract_url})" if abstract_url else f"- {heading}: {abstract[:300]}")

            for topic in (data.get("RelatedTopics") or [])[:5]:
                if isinstance(topic, dict):
                    text = (topic.get("Text") or "").strip()
                    url = (topic.get("FirstURL") or "").strip()
                    dedupe_key = url or text[:120]
                    if text and dedupe_key not in seen:
                        seen.add(dedupe_key)
                        lines.append(f"- {text[:220]} ({url})" if url else f"- {text[:220]}")

        return "\n".join(lines) if lines else "No web results found for this person."
    except Exception as e:
        return f"Web search failed: {e}"


def parse_ddg_html_results(page_html: str, max_results: int = 8) -> list[tuple[str, str]]:
    """Parse result titles/urls from DuckDuckGo HTML page."""
    out = []
    if not page_html:
        return out
    pattern = re.compile(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE)
    for href, raw_title in pattern.findall(page_html):
        title = re.sub(r"<.*?>", "", raw_title).strip()
        if title:
            out.append((title, href))
        if len(out) >= max_results:
            break
    return out


def tool_github(username: str) -> str:
    """Fetch GitHub profile and top repos for the given username."""
    if not (username or "").strip():
        return "No GitHub username set. Add your GitHub username in the sidebar (Me mode)."
    try:
        u = (username or "").strip()
        r = requests.get(f"https://api.github.com/users/{u}", timeout=10)
        if r.status_code == 404:
            return f"GitHub user '{u}' not found."
        r.raise_for_status()
        profile = r.json()
        bio = profile.get("bio") or ""
        name = profile.get("name") or u
        public_repos = profile.get("public_repos", 0)
        repos_r = requests.get(
            f"https://api.github.com/users/{u}/repos?sort=updated&per_page=8",
            timeout=10,
        )
        repos_r.raise_for_status()
        repos = repos_r.json()
        repo_lines = [f"- {repo.get('name', '')}: {repo.get('description') or 'No description'}" for repo in repos]
        return f"GitHub: {name} (@{u}). Bio: {bio}. Public repos: {public_repos}.\nTop repos:\n" + "\n".join(repo_lines)
    except Exception as e:
        return f"GitHub lookup failed: {e}"


def get_all_resume_chunks_for_candidate(candidate_name: str) -> str:
    """Get every chunk for one candidate (for open-ended summary requests)."""
    try:
        result = collection.get(
            where={"name": candidate_name},
            include=["documents", "metadatas"],
        )
        if not result or not result.get("documents"):
            return ""
        docs = result["documents"]
        return "\n\n---\n\n".join(f"[{candidate_name}'s resume]:\n{d}" for d in docs)
    except Exception:
        return ""


def needs_full_resume(question: str) -> bool:
    """Ask LLM whether this question needs the full resume (summary/intro) vs specific search."""
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Does this question ask for a full summary, introduction, or overview of the person (so we need the entire resume)? Reply with ONLY YES or NO.",
                },
                {"role": "user", "content": question.strip()},
            ],
            temperature=0,
            max_tokens=5,
        )
        return (response.choices[0].message.content or "").strip().upper().startswith("YES")
    except Exception:
        return False


def extract_weather_location(question: str) -> str:
    """Extract likely location from weather question, e.g. 'weather in Boston'."""
    q = (question or "").strip()
    if not q:
        return ""
    lowered = q.lower()
    match = re.search(r"\b(?:in|at|for)\s+([a-zA-Z][a-zA-Z\s,.-]{1,60})$", lowered)
    if match:
        return match.group(1).strip(" ?!.,")
    match = re.search(r"\bweather\s+([a-zA-Z][a-zA-Z\s,.-]{1,60})$", lowered)
    if match:
        return match.group(1).strip(" ?!.,")
    return ""


def extract_search_focus(question: str, candidate_name: str) -> str:
    """Keep only useful focus terms from person web-search question."""
    q = (question or "").lower()
    q = q.replace(candidate_name.lower(), " ")
    for phrase in (
        "search the web", "web search", "google", "look up", "look me up",
        "find online", "what does the web say", "about me", "about this person",
        "search for", "search me", "find me online", "online",
    ):
        q = q.replace(phrase, " ")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", q)
    tokens = [t for t in cleaned.split() if len(t) > 2]
    return " ".join(tokens[:8])


def is_weather_query(question: str) -> bool:
    """Detect weather-related requests."""
    q = (question or "").lower()
    return any(phrase in q for phrase in (
        "weather", "temperature", "forecast", "how hot", "how cold",
        "weather like", "rain", "sunny", "humidity", "wind",
    ))


def is_github_query(question: str) -> bool:
    q = (question or "").lower()
    return any(phrase in q for phrase in ("github", "git hub", "repos", "repositories", "repo", "projects on github"))


def is_self_resume_query(question: str) -> bool:
    """Recruiter-style prompts that should use resume context in Me mode."""
    q = (question or "").strip().lower()
    return any(phrase in q for phrase in (
        "tell me about yourself",
        "tell me more about yourself",
        "introduce yourself",
        "walk me through your background",
        "your background",
        "your experience",
        "your skills",
        "your resume",
        "why should we hire you",
        "what are your strengths",
    ))


def is_small_talk(question: str) -> bool:
    q = (question or "").strip().lower()
    return any(phrase in q for phrase in (
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "nice to meet you", "how are you",
    ))


def resolve_person_from_question(question: str, candidates: list, selected: str = None) -> str:
    """Resolve target candidate by explicit mention, then selected filter, then first candidate."""
    if not candidates:
        return "The candidate"
    q = (question or "").lower()
    for name in sorted(candidates, key=len, reverse=True):
        if name and name.lower() in q:
            return name
    if selected and selected != "All" and selected in candidates:
        return selected
    return candidates[0]


def select_tool(question: str) -> str:
    """Choose which tool: resume (semantic search), web_search (person or weather only), github, or deny."""
    q = (question or "").strip().lower()
    if is_self_resume_query(question):
        return "resume"
    if is_github_query(question):
        return "github"
    # Weather → web search only (no other web use)
    if is_weather_query(question):
        return "web_search"
    # Any explicit web lookup intent should route to the constrained person web tool
    if ("search" in q or "web" in q or "online" in q or "google" in q or "look up" in q) and "github" not in q:
        return "web_search"
    # Explicit web search about the person → web search
    if any(phrase in q for phrase in (
        "web search", "search for", "search me", "what does the web say",
        "find me online", "google me", "look me up", "what's online about",
        "what do you find about", "search the web for", "look up", "find online",
    )):
        return "web_search"
    # Follow-ups about a specific term = resume
    if any(phrase in q for phrase in ("what about", "tell me about", "and ", "also ", "your ", "my ")):
        if not any(phrase in q for phrase in ("github", "search the")):
            return "resume"
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You decide which tool to use. Reply with ONLY one word:
- resume: background, experience, skills, education, career, job history, resume (semantic search)
- web_search: ONLY when they ask (a) what the web says about this person, OR (b) weather/temperature/forecast. No other web search.
- github: when they ask for GitHub, repos, code, "my github", "my projects"
- deny: general news, random facts, any other web search. Weather and "search for me" are web_search, not deny."""
                },
                {"role": "user", "content": question.strip()},
            ],
            temperature=0,
            max_tokens=15,
        )
        choice = (response.choices[0].message.content or "").strip().lower().replace(" ", "_")
        if choice in ("resume", "web_search", "web_search_person", "github", "deny"):
            return "web_search" if choice == "web_search_person" else choice
    except Exception:
        pass
    return "resume"


def infer_github_username_from_resume(candidate_name: str) -> str:
    """Infer GitHub username from resume links for the candidate."""
    if not candidate_name or candidate_name == "The candidate":
        return ""
    try:
        result = collection.get(where={"name": candidate_name}, include=["documents"])
        docs = result.get("documents") or []
        text = "\n".join(docs)
        # Handle github.com/user or github.com/user/repo
        matches = re.findall(r"github\.com/([A-Za-z0-9-]{1,39})", text, flags=re.IGNORECASE)
        if not matches:
            return ""
        bad = {"features", "topics", "orgs", "organizations", "settings", "marketplace", "apps"}
        for m in matches:
            if m.lower() not in bad:
                return m
        return matches[0]
    except Exception:
        return ""


def extract_profile_links_from_resume(candidate_name: str) -> list[str]:
    """Extract likely online profile URLs from resume content."""
    if not candidate_name or candidate_name == "The candidate":
        return []
    try:
        result = collection.get(where={"name": candidate_name}, include=["documents"])
        docs = result.get("documents") or []
        text = "\n".join(docs)
        urls = re.findall(r"https?://[^\s)>\"]+", text, flags=re.IGNORECASE)
        keep = []
        for u in urls:
            ul = u.lower()
            if any(d in ul for d in ("github.com/", "linkedin.com/in/", "medium.com/", "x.com/", "twitter.com/")):
                keep.append(u.rstrip(".,);"))
        deduped = []
        seen = set()
        for u in keep:
            if u not in seen:
                seen.add(u)
                deduped.append(u)
        return deduped[:6]
    except Exception:
        return []


def get_answer_impersonation(
    question: str,
    resume_context: str,
    web_search_data: str,
    github_data: str,
    candidate_name: str,
    history: list,
) -> str:
    """Generate answer as the candidate (you) talking to a recruiter, using only tool data."""
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    history_text = ""
    if history:
        history_text = "\n\nCONVERSATION:\n" + "\n".join(
            [f"Recruiter: {h['q']}\nYou: {h['a'][:200]}" for h in history[-5:]]
        )

    tools_block = f"""
RESUME DATA (semantic search – use for background/skills/experience):
{resume_context or "No resume data for this question."}

WEB SEARCH (about this person or weather – only these use the web):
{web_search_data or "Not fetched."}

GITHUB:
{github_data or "Not fetched."}
"""

    response = llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""You are {candidate_name}, a candidate. A recruiter is chatting with you. Today is {current_date}.

STRICT RULES – NO HALLUCINATION:
1. Your ONLY sources of truth are the TOOLS block below. Do not use your training knowledge for resume, experience, skills, companies, or education.
2. If the RESUME DATA does not mention something, say "That's not on my resume" or "I don't have that in the info you have" – do NOT make it up.
3. For resume/career: answer only from RESUME DATA. Paraphrase or quote that text only.
4. For "what does the web say about me" / "search for me": use only the WEB SEARCH block. Summarize what was found.
5. For weather / temperature / forecast: use only the WEB SEARCH block (live weather from web). Answer briefly.
6. For GitHub / repos / code: use only the GITHUB block. If it says no username set, say you can add it in the sidebar.
7. Never invent job titles, companies, dates, skills, or education. If it's not in the block, say you don't have that information.
8. For "tell me about yourself" or similar: include EVERY notable detail from RESUME DATA (all companies, projects, roles, skills, achievements). Do not omit anything that appears in the data.""",

            },
            {
                "role": "user",
                "content": f"{tools_block}{history_text}\n\nRecruiter asks: {question}\n\nYour reply (only from the data above):",
            },
        ],
        temperature=0,
        max_tokens=1500,
    )
    return response.choices[0].message.content


# ============== STREAMLIT APP ==============

st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .main-header { text-align: center; padding: 1rem 0; border-bottom: 2px solid #4A90A4; margin-bottom: 2rem; }
    .chat-row { display: flex; margin: 0.55rem 0; }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .chat-bubble {
        max-width: 78%;
        padding: 0.85rem 1rem;
        border-radius: 14px;
        line-height: 1.45;
        font-size: 1.02rem;
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 4px 18px rgba(0,0,0,0.2);
        white-space: normal;
        word-wrap: break-word;
    }
    .chat-row.user .chat-bubble {
        background: linear-gradient(135deg, #2f3d63, #445a9c);
        color: #f7f9ff;
    }
    .chat-row.assistant .chat-bubble {
        background: rgba(255,255,255,0.05);
        color: #f4f6ff;
    }
    .chat-tag {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        opacity: 0.8;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Resume RAG System")
st.caption("Ask questions about uploaded resumes using AI-powered search")

if "history" not in st.session_state:
    st.session_state.history = []


def render_chat_bubble(role: str, message: str) -> None:
    safe = html.escape(message or "").replace("\n", "<br>")
    tag = "Recruiter" if role == "user" else "Candidate"
    st.markdown(
        f"""
        <div class="chat-row {role}">
            <div class="chat-bubble">
                <div class="chat-tag">{tag}</div>
                <div>{safe}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    app_mode = st.radio(
        "Mode",
        ["Recruiter (browse candidates)", "Me (recruiter talks to me)"],
        index=0,
        help="Recruiter: multi-candidate Q&A. Me: you as candidate, 3 tools (semantic search, web search about you, GitHub).",
    )
    me_mode = "Me" in app_mode

    if me_mode:
        if "github_username" not in st.session_state:
            st.session_state.github_username = ""
        st.session_state.github_username = st.text_input(
            "GitHub username (for Me mode)",
            value=st.session_state.github_username,
            placeholder="e.g. octocat",
            help="Your GitHub handle so the bot can fetch your profile and repos.",
        )

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
        if me_mode:
            render_chat_bubble("user", h["q"])
            render_chat_bubble("assistant", h["a"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.write(h["q"])
            with st.chat_message("assistant", avatar="🤖"):
                st.write(h["a"])

    prompt_label = "Ask about the candidates..." if not me_mode else "Chat with the candidate (background, projects, GitHub)..."
    if question := st.chat_input(prompt_label):
        if me_mode:
            render_chat_bubble("user", question)
        else:
            with st.chat_message("user", avatar="👤"):
                st.write(question)

        if me_mode:
            # Me mode: 3 tools = semantic search (resume), web search about person, GitHub
            tool = select_tool(question)
            candidate_name = resolve_person_from_question(question, candidates, selected)
            resume_ctx = ""
            web_search_data = ""
            github_data = ""

            if tool == "deny":
                if is_small_talk(question):
                    answer = "Great to meet you. I can walk you through my background, experience, key projects, and GitHub work."
                else:
                    # In Me mode, default to resume context for recruiter-style chat instead of hard denying.
                    use_full = me_mode and candidates and needs_full_resume(question)
                    if use_full:
                        resume_ctx = get_all_resume_chunks_for_candidate(
                            selected if selected and selected != "All" else candidates[0]
                        )
                    else:
                        rewritten = rewrite_query(question, st.session_state.history, candidates)
                        resume_ctx, _ = search_resumes(rewritten, question, selected, candidates)
                    answer = get_answer_impersonation(
                        question,
                        resume_context=resume_ctx,
                        web_search_data="",
                        github_data="",
                        candidate_name=candidate_name,
                        history=st.session_state.history,
                    )
            else:
                if tool == "resume":
                    use_full = me_mode and candidates and needs_full_resume(question)
                    if use_full:
                        resume_ctx = get_all_resume_chunks_for_candidate(
                            selected if selected and selected != "All" else candidates[0]
                        )
                    else:
                        rewritten = rewrite_query(question, st.session_state.history, candidates)
                        resume_ctx, _ = search_resumes(rewritten, question, selected, candidates)
                if tool == "web_search":
                    is_weather = is_weather_query(question)
                    web_search_data = tool_web_search(candidate_name, question, for_weather=is_weather)
                if tool == "github":
                    github_username = st.session_state.get("github_username", "").strip() or infer_github_username_from_resume(candidate_name)
                    github_data = tool_github(github_username)

                if tool == "web_search":
                    if is_weather_query(question):
                        answer = f"Weather:\n{web_search_data}"
                    else:
                        answer = f"Web results about {candidate_name}:\n{web_search_data}"
                elif tool == "github":
                    answer = github_data
                else:
                    answer = get_answer_impersonation(
                        question,
                        resume_context=resume_ctx,
                        web_search_data=web_search_data,
                        github_data=github_data,
                        candidate_name=candidate_name,
                        history=st.session_state.history,
                    )

            with st.spinner("Thinking..."):
                pass  # already computed above

            render_chat_bubble("assistant", answer)
            with st.expander("🔍 Debug (Me mode)"):
                st.text(f"Tool used: {tool}")
        else:
            with st.spinner("Searching..."):
                candidates_list = get_all_candidates()
                tool = select_tool(question)
                target_name = resolve_person_from_question(question, candidates_list, selected)

                if tool == "web_search":
                    data = tool_web_search(target_name, question, for_weather=is_weather_query(question))
                    if is_weather_query(question):
                        answer = f"Weather:\n{data}"
                    else:
                        answer = f"Web results about {target_name}:\n{data}"
                    rewritten = question
                    meta = []
                elif tool == "github":
                    gh_user = infer_github_username_from_resume(target_name)
                    answer = tool_github(gh_user)
                    rewritten = question
                    meta = []
                else:
                    rewritten = rewrite_query(question, st.session_state.history, candidates_list)
                    context, meta = search_resumes(rewritten, question, selected, candidates_list)
                    answer = get_answer(question, context, st.session_state.history, candidates_list)

            with st.chat_message("assistant", avatar="🤖"):
                st.write(answer)
            with st.expander("🔍 Debug"):
                st.text(f"Original: {question}")
                st.text(f"Rewritten: {rewritten}")
                st.text(f"Tool used: {tool}")
                st.text(f"Chunks retrieved: {len(meta)}")
                if meta:
                    sources = list(set(m.get("name", "?") for m in meta))
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
