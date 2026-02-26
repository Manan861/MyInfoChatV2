"""
Resume RAG System - v4
Improvements: Scales to 40+ resumes, better specific question handling, improved accuracy
"""

import os
import json
import io
import uuid
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import datetime
import requests
import re
import uvicorn
from fastapi import Body, FastAPI, File, Request as FastAPIRequest, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

        # Person web search path via Tavily
        tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()
        if not tavily_key:
            return "Tavily API key is missing. Add TAVILY_API_KEY to your .env to enable web search."

        from tavily import TavilyClient

        specific_terms = extract_search_focus(question, candidate_name)
        base_query = f"{candidate_name} {specific_terms}".strip()
        query_variants = [
            base_query,
            f"{candidate_name} profile",
            f"{candidate_name} linkedin github",
            f"{candidate_name} portfolio projects",
        ]
        query_variants = [q.strip() for q in query_variants if q and q.strip()]

        client = TavilyClient(api_key=tavily_key)

        lines = []
        seen = set()
        answers = []

        # Seed with profile links found in resume text (high-confidence identity anchors)
        profile_links = extract_profile_links_from_resume(candidate_name)
        for link in profile_links:
            if link not in seen:
                seen.add(link)
                lines.append(f"- Profile found in resume: {link}")

        for q in query_variants:
            try:
                search_data = client.search(
                    query=q,
                    search_depth="advanced",
                    max_results=6,
                    include_answer=True,
                    include_raw_content=False,
                )
            except Exception:
                continue

            answer = (search_data or {}).get("answer") or ""
            answer = answer.strip()
            if answer and answer not in answers:
                answers.append(answer)

            for r in (search_data or {}).get("results", []):
                title = (r.get("title") or "").strip()
                url = (r.get("url") or "").strip()
                content = (r.get("content") or "").strip()
                dedupe_key = url or title
                if dedupe_key in seen:
                    continue
                if title or content:
                    seen.add(dedupe_key)
                    snippet = content[:220] if content else "No summary available."
                    lines.append(f"- {title or 'Result'}: {snippet} ({url})" if url else f"- {title or 'Result'}: {snippet}")
                if len(lines) >= 12:
                    break
            if len(lines) >= 12:
                break

        if answers:
            summary_lines = [f"- Tavily summary: {a[:260]}" for a in answers[:2]]
            lines = summary_lines + lines

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


def normalize_github_username(value: str) -> str:
    """Normalize GitHub username from plain handle, @handle, or URL."""
    raw = (value or "").strip()
    if not raw:
        return ""
    raw = raw.strip("()[]{}<>.,;:!?'\"")
    if raw.startswith("@"):
        raw = raw[1:]
    m = re.search(r"github\.com/([A-Za-z0-9-]{1,39})", raw, flags=re.IGNORECASE)
    if m:
        raw = m.group(1)
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    raw = re.sub(r"[^A-Za-z0-9-]", "", raw)
    return raw[:39]


def infer_github_username_from_web(candidate_name: str) -> str:
    """Fallback GitHub username inference using Tavily person search."""
    if not candidate_name or candidate_name == "The candidate":
        return ""
    tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not tavily_key:
        return ""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_key)
        queries = [
            f"{candidate_name} github",
            f"{candidate_name} github profile",
            f"{candidate_name} github.com",
        ]
        bad = {"features", "topics", "orgs", "organizations", "settings", "marketplace", "apps", "events", "users"}

        for q in queries:
            search_data = client.search(
                query=q,
                search_depth="advanced",
                max_results=6,
                include_answer=True,
                include_raw_content=False,
            )
            answer = (search_data or {}).get("answer") or ""
            answer_match = re.search(r"github\.com/([A-Za-z0-9-]{1,39})", answer, flags=re.IGNORECASE)
            if answer_match:
                candidate = normalize_github_username(answer_match.group(1))
                if candidate and candidate.lower() not in bad:
                    return candidate

            for r in (search_data or {}).get("results", []):
                url = (r.get("url") or "").strip()
                content = (r.get("content") or "").strip()
                blob = f"{url}\n{content}"
                m = re.search(r"github\.com/([A-Za-z0-9-]{1,39})", blob, flags=re.IGNORECASE)
                if m:
                    candidate = normalize_github_username(m.group(1))
                    if candidate and candidate.lower() not in bad:
                        return candidate
        return ""
    except Exception:
        return ""


def tool_github(username: str, mode: str = "me", candidate_name: str = "") -> str:
    """Fetch GitHub profile and top repos for the given username."""
    u = normalize_github_username(username)
    if not u:
        if mode == "recruiter":
            label = candidate_name or "this candidate"
            return (
                f"I couldn't find a GitHub username for {label} from resume/web signals. "
                "Ask for an explicit GitHub handle or add a GitHub link in the resume."
            )
        return "No GitHub username set. Add your GitHub username in the sidebar (Me mode)."
    try:
        r = requests.get(f"https://api.github.com/users/{u}", timeout=10)
        if r.status_code == 404:
            return f"GitHub user '{u}' not found."
        r.raise_for_status()
        profile = r.json()
        bio = profile.get("bio") or ""
        name = profile.get("name") or u
        public_repos = profile.get("public_repos", 0)
        followers = profile.get("followers", 0)
        following = profile.get("following", 0)
        company = profile.get("company") or ""
        location = profile.get("location") or ""
        blog = profile.get("blog") or ""
        repos_r = requests.get(
            f"https://api.github.com/users/{u}/repos?sort=updated&per_page=8",
            timeout=10,
        )
        repos_r.raise_for_status()
        repos = repos_r.json()
        repos = sorted(repos, key=lambda x: (x.get("stargazers_count") or 0), reverse=True)[:8]
        repo_lines = []
        for repo in repos:
            repo_name = repo.get("name", "")
            repo_desc = repo.get("description") or "No description"
            repo_url = repo.get("html_url") or ""
            stars = repo.get("stargazers_count", 0)
            forks = repo.get("forks_count", 0)
            lang = repo.get("language") or "n/a"
            if repo_url:
                repo_lines.append(f"- {repo_name} [{lang}, ★{stars}, forks {forks}]: {repo_url} ({repo_desc})")
            else:
                repo_lines.append(f"- {repo_name} [{lang}, ★{stars}, forks {forks}]: {repo_desc}")
        details = []
        if company:
            details.append(f"Company: {company}")
        if location:
            details.append(f"Location: {location}")
        if blog:
            details.append(f"Website: {blog}")
        details_text = (" " + " | ".join(details)) if details else ""
        return (
            f"GitHub: {name} (@{u}). Bio: {bio}. Public repos: {public_repos}. "
            f"Followers: {followers}, Following: {following}.{details_text}\nTop repos:\n"
            + "\n".join(repo_lines)
        )
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


def is_meeting_query(question: str) -> bool:
    q = (question or "").strip().lower()
    return any(phrase in q for phrase in (
        "let's meet", "lets meet", "can we meet", "schedule", "set up a call",
        "book time", "interview at", "meet at", "talk at", "connect at",
    ))


def resolve_person_from_question(question: str, candidates: list, selected: str = None, history: list | None = None) -> str:
    """Resolve target candidate by mention, selection, conversation history, then first candidate."""
    if not candidates:
        return "The candidate"
    q = (question or "").lower()
    q_norm = re.sub(r"[^a-z0-9\s]", " ", q)

    # Direct full-name mention
    for name in sorted(candidates, key=len, reverse=True):
        if name and name.lower() in q:
            return name

    # First-name / possessive mention (e.g., "manans github")
    for name in candidates:
        parts = [p for p in re.split(r"\s+", name.lower().strip()) if p]
        if not parts:
            continue
        first = re.sub(r"[^a-z0-9]", "", parts[0])
        if len(first) >= 3:
            if re.search(rf"\b{re.escape(first)}\b", q_norm) or re.search(rf"\b{re.escape(first)}s\b", q_norm):
                return name

    if selected and selected != "All" and selected in candidates:
        return selected

    # Pronoun follow-up fallback to most recently referenced candidate in recruiter history
    if history:
        recent = history[-6:]
        for h in reversed(recent):
            combined = f"{h.get('q', '')} {h.get('a', '')}".lower()
            for name in sorted(candidates, key=len, reverse=True):
                if name.lower() in combined:
                    return name

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


def get_meeting_reply(question: str) -> str:
    """Light conversational response for scheduling-style recruiter messages."""
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a job candidate replying to a recruiter. "
                        "Reply in 1-2 short sentences. Confirm the meeting professionally "
                        "if a time is proposed. If no concrete time is provided, ask for one."
                    ),
                },
                {"role": "user", "content": question.strip()},
            ],
            temperature=0.2,
            max_tokens=80,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return "That sounds good. Please share the exact time and time zone, and I’ll confirm."


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


app = FastAPI(title="Resume RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

APP_STATE = {
    "history": {"recruiter": [], "me": []},
    "me_candidate": "",
    "github_username": "",
}


def ensure_me_candidate(candidates: list[str]) -> str:
    """Keep selected Me profile valid and default to first candidate."""
    if not candidates:
        APP_STATE["me_candidate"] = ""
        return ""
    current = APP_STATE.get("me_candidate", "")
    if current not in candidates:
        APP_STATE["me_candidate"] = candidates[0]
    return APP_STATE["me_candidate"]


def snapshot(mode: str = "recruiter") -> dict:
    """Return current UI state payload for frontend."""
    candidates = get_all_candidates()
    me_candidate = ensure_me_candidate(candidates)
    return {
        "mode": mode,
        "candidates": candidates,
        "resume_count": get_resume_count(),
        "chunk_count": collection.count(),
        "history": APP_STATE["history"].get(mode, []),
        "me_candidate": me_candidate,
        "github_username": APP_STATE.get("github_username", ""),
    }


@app.get("/")
def index(request: FastAPIRequest):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/state")
def api_state(mode: str = "recruiter"):
    mode = (mode or "recruiter").strip().lower()
    mode = "me" if mode == "me" else "recruiter"
    return snapshot(mode)


@app.post("/api/clear-chat")
def api_clear_chat(payload: dict | None = Body(default=None)):
    payload = payload or {}
    mode = (payload.get("mode") or "recruiter").strip().lower()
    if mode in APP_STATE["history"]:
        APP_STATE["history"][mode] = []
    else:
        APP_STATE["history"] = {"recruiter": [], "me": []}
    return {"ok": True, "state": snapshot(mode if mode in ("me", "recruiter") else "recruiter")}


@app.post("/api/clear-data")
def api_clear_data():
    global collection
    try:
        vec_db.delete_collection("resumes")
    except Exception:
        pass
    collection = vec_db.get_or_create_collection("resumes")
    APP_STATE["history"] = {"recruiter": [], "me": []}
    APP_STATE["me_candidate"] = ""
    return {"ok": True, "state": snapshot("recruiter")}


@app.post("/api/upload")
async def api_upload(files: list[UploadFile] | None = File(default=None)):
    if not files:
        return JSONResponse(status_code=400, content={"ok": False, "error": "No files uploaded."})

    processed = []
    skipped = []

    for f in files:
        filename = (f.filename or "").strip()
        if not filename or not filename.lower().endswith(".pdf"):
            skipped.append({"file": filename or "unknown", "reason": "Only PDF files are supported."})
            continue
        try:
            raw = await f.read()
            if not raw:
                skipped.append({"file": filename, "reason": "Empty file."})
                continue

            pdf = PyPDF2.PdfReader(io.BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if not text.strip():
                skipped.append({"file": filename, "reason": "No extractable text found."})
                continue

            valid, _reason = is_resume(text)
            if not valid:
                skipped.append({"file": filename, "reason": "Document is not a resume."})
                continue

            meta = extract_metadata(text, filename)
            chunks = chunk_text(text)
            if not chunks:
                skipped.append({"file": filename, "reason": "No chunks generated."})
                continue

            ids = [f"{filename}_{i}_{uuid.uuid4().hex}" for i, _ in enumerate(chunks)]
            metadatas = [{"name": meta["name"], "filename": filename} for _ in chunks]
            collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            processed.append({"file": filename, "candidate": meta["name"], "chunks": len(chunks)})
        except Exception as e:
            skipped.append({"file": filename, "reason": str(e)})

    candidates = get_all_candidates()
    ensure_me_candidate(candidates)
    return {
        "ok": True,
        "processed": processed,
        "skipped": skipped,
        "state": snapshot("recruiter"),
    }


@app.post("/api/chat")
def api_chat(payload: dict | None = Body(default=None)):
    payload = payload or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Message cannot be empty."})

    mode = (payload.get("mode") or "recruiter").strip().lower()
    mode = "me" if mode == "me" else "recruiter"

    selected = (payload.get("selected") or "All").strip()
    incoming_me_candidate = (payload.get("me_candidate") or "").strip()
    incoming_github = (payload.get("github_username") or "").strip()

    if incoming_github:
        APP_STATE["github_username"] = incoming_github

    candidates = get_all_candidates()
    ensure_me_candidate(candidates)

    if incoming_me_candidate and incoming_me_candidate in candidates:
        APP_STATE["me_candidate"] = incoming_me_candidate

    if mode == "me":
        candidate_name = APP_STATE.get("me_candidate") or (candidates[0] if candidates else "The candidate")
        tool = "meeting" if is_meeting_query(question) else select_tool(question)
        resume_ctx = ""
        web_search_data = ""
        github_data = ""

        if tool == "meeting":
            answer = get_meeting_reply(question)
        elif tool == "deny":
            if is_small_talk(question):
                answer = "Great to meet you. I can walk you through my background, experience, key projects, and GitHub work."
            else:
                use_full = bool(candidate_name and candidate_name != "The candidate") and needs_full_resume(question)
                if use_full:
                    resume_ctx = get_all_resume_chunks_for_candidate(candidate_name)
                else:
                    rewritten = rewrite_query(question, APP_STATE["history"]["me"], [candidate_name] if candidate_name else [])
                    resume_ctx, _ = search_resumes(rewritten, question, candidate_name, None)
                answer = get_answer_impersonation(
                    question,
                    resume_context=resume_ctx,
                    web_search_data="",
                    github_data="",
                    candidate_name=candidate_name,
                    history=APP_STATE["history"]["me"],
                )
        else:
            if tool == "resume":
                use_full = bool(candidate_name and candidate_name != "The candidate") and needs_full_resume(question)
                if use_full:
                    resume_ctx = get_all_resume_chunks_for_candidate(candidate_name)
                else:
                    rewritten = rewrite_query(question, APP_STATE["history"]["me"], [candidate_name] if candidate_name else [])
                    resume_ctx, _ = search_resumes(rewritten, question, candidate_name, None)
            if tool == "web_search":
                web_search_data = tool_web_search(candidate_name, question, for_weather=is_weather_query(question))
            if tool == "github":
                github_username = (
                    APP_STATE.get("github_username", "").strip()
                    or infer_github_username_from_resume(candidate_name)
                    or infer_github_username_from_web(candidate_name)
                )
                github_data = tool_github(github_username, mode="me", candidate_name=candidate_name)

            if tool == "web_search":
                answer = f"Weather:\n{web_search_data}" if is_weather_query(question) else f"Web results about {candidate_name}:\n{web_search_data}"
            elif tool == "github":
                answer = github_data
            else:
                answer = get_answer_impersonation(
                    question,
                    resume_context=resume_ctx,
                    web_search_data=web_search_data,
                    github_data=github_data,
                    candidate_name=candidate_name,
                    history=APP_STATE["history"]["me"],
                )

        APP_STATE["history"]["me"].append({"q": question, "a": answer})
    else:
        tool = select_tool(question)
        target_name = resolve_person_from_question(
            question,
            candidates,
            selected,
            APP_STATE["history"]["recruiter"],
        )

        if tool == "web_search":
            data = tool_web_search(target_name, question, for_weather=is_weather_query(question))
            answer = f"Weather:\n{data}" if is_weather_query(question) else f"Web results about {target_name}:\n{data}"
        elif tool == "github":
            gh_user = infer_github_username_from_resume(target_name) or infer_github_username_from_web(target_name)
            answer = tool_github(gh_user, mode="recruiter", candidate_name=target_name)
        else:
            rewritten = rewrite_query(question, APP_STATE["history"]["recruiter"], candidates)
            context, _meta = search_resumes(rewritten, question, selected, candidates)
            answer = get_answer(question, context, APP_STATE["history"]["recruiter"], candidates)

        APP_STATE["history"]["recruiter"].append({"q": question, "a": answer})

    return {"ok": True, "answer": answer, "tool": tool, "state": snapshot(mode)}


if __name__ == "__main__":
    uvicorn.run("resume_rag:app", host="0.0.0.0", port=int(os.getenv("PORT", "5050")), reload=True)
