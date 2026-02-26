const { useEffect, useMemo, useRef, useState } = React;

function esc(str) {
  return (str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function toHtml(text) {
  return esc(text || "")
    .replace(/(https?:\/\/[^\s<]+)/g, '<a class="text-brand underline underline-offset-4" href="$1" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\n/g, "<br/>");
}

function Avatar({ role }) {
  const isUser = role === "user";
  const style = isUser
    ? {
        background: "linear-gradient(135deg, #2152ff, #112a84)",
        color: "#ffffff",
        border: "1px solid rgba(17, 42, 132, 0.4)",
      }
    : {
        background: "#ffffff",
        color: "#0d1b3d",
        border: "1px solid #d9e2f2",
      };
  return (
    <div
      className="shrink-0"
      style={{
        width: 40,
        height: 40,
        borderRadius: 12,
        display: "grid",
        placeItems: "center",
        fontSize: 12,
        fontWeight: 700,
        boxShadow: "0 10px 24px rgba(15, 23, 42, 0.1)",
        ...style,
      }}
    >
      {isUser ? "You" : "AI"}
    </div>
  );
}

function MessageBubble({ mode, role, text }) {
  const isUser = role === "user";
  const label = isUser ? "Recruiter" : (mode === "me" ? "Candidate" : "Assistant");
  const bubbleStyle = isUser
    ? {
        background: "linear-gradient(135deg, #2152ff, #112a84)",
        color: "#ffffff",
        border: "1px solid rgba(33, 82, 255, 0.28)",
      }
    : mode === "me"
      ? {
          background: "#ffffff",
          color: "#0d1b3d",
          border: "1px solid rgba(8, 168, 141, 0.35)",
        }
      : {
          background: "#ffffff",
          color: "#0d1b3d",
          border: "1px solid #d9e2f2",
        };

  return (
    <div className={`message-rise flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && <Avatar role="assistant" />}
      <div
        style={{
          maxWidth: "86%",
          borderRadius: 16,
          padding: "12px 16px",
          boxShadow: "0 10px 24px rgba(15, 23, 42, 0.08)",
          ...bubbleStyle,
        }}
      >
        <p
          style={{
            marginBottom: 4,
            fontSize: 11,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            fontWeight: 700,
            color: isUser ? "#dbeafe" : "#5d6b8a",
          }}
        >
          {label}
        </p>
        <div style={{ fontSize: 16, lineHeight: 1.55 }} dangerouslySetInnerHTML={{ __html: toHtml(text) }} />
      </div>
      {isUser && <Avatar role="user" />}
    </div>
  );
}

function TypingBubble({ mode }) {
  const label = mode === "me" ? "Candidate" : "Assistant";
  const style = mode === "me"
    ? { background: "#ffffff", border: "1px solid rgba(8, 168, 141, 0.35)" }
    : { background: "#ffffff", border: "1px solid #d9e2f2" };
  return (
    <div className="message-rise flex gap-3 justify-start">
      <Avatar role="assistant" />
      <div
        style={{
          maxWidth: "86%",
          borderRadius: 16,
          padding: "12px 16px",
          boxShadow: "0 10px 24px rgba(15, 23, 42, 0.08)",
          ...style,
        }}
      >
        <p style={{ marginBottom: 4, fontSize: 11, letterSpacing: "0.16em", textTransform: "uppercase", fontWeight: 700, color: "#5d6b8a" }}>{label}</p>
        <div className="flex items-center gap-1.5">
          <span className="animate-pulse" style={{ width: 8, height: 8, borderRadius: 999, background: "rgba(93,107,138,0.75)" }}></span>
          <span className="animate-pulse [animation-delay:120ms]" style={{ width: 8, height: 8, borderRadius: 999, background: "rgba(93,107,138,0.75)" }}></span>
          <span className="animate-pulse [animation-delay:240ms]" style={{ width: 8, height: 8, borderRadius: 999, background: "rgba(93,107,138,0.75)" }}></span>
        </div>
      </div>
    </div>
  );
}

function StatTile({ label, value, accent }) {
  const style = accent === "brand"
    ? {
        background: "linear-gradient(135deg, #2152ff, #112a84)",
        color: "#ffffff",
        border: "1px solid rgba(33,82,255,0.3)",
      }
    : {
        background: "linear-gradient(135deg, #e8fff9, #dcfff4)",
        color: "#066f5f",
        border: "1px solid rgba(8,168,141,0.28)",
      };
  return (
    <div className="rounded-2xl px-4 py-3 shadow-card" style={style}>
      <p className="text-[11px] uppercase tracking-[0.16em] font-semibold opacity-80">{label}</p>
      <p className="font-display text-3xl mt-1 leading-none">{value}</p>
    </div>
  );
}

function App() {
  const [mode, setMode] = useState("recruiter");
  const [candidates, setCandidates] = useState([]);
  const [selected, setSelected] = useState("All");
  const [meCandidate, setMeCandidate] = useState("");
  const [githubUsername, setGithubUsername] = useState("");
  const [resumeCount, setResumeCount] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [history, setHistory] = useState([]);
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploadReport, setUploadReport] = useState({ processed: [], skipped: [] });
  const [pendingUserMessage, setPendingUserMessage] = useState("");
  const [awaitingReply, setAwaitingReply] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const fileRef = useRef(null);
  const chatRef = useRef(null);

  const hasResumes = resumeCount > 0;

  const panelTitle = mode === "me" ? "Candidate Conversation" : "Recruiter Screening Chat";
  const panelSubtitle = mode === "me"
    ? "Talk naturally as a recruiter. Responses stay in first-person from the selected profile."
    : "Ask hiring questions, compare candidates, and request specific profile insights.";

  const inputPlaceholder = mode === "me"
    ? "Type as recruiter: tell me about yourself, can we meet tomorrow at 2 PM..."
    : "Ask about skills, comparisons, strengths, or specific candidates...";

  const emptyHint = mode === "me"
    ? "Start with: Tell me about yourself, walk me through your experience, share your GitHub work"
    : "Start with: Who is strongest in backend? Compare top candidates for this role.";

  useEffect(() => {
    loadState(mode);
  }, [mode]);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [history, pendingUserMessage, awaitingReply]);

  async function loadState(requestMode) {
    try {
      const res = await fetch(`/api/state?mode=${encodeURIComponent(requestMode)}`);
      const data = await res.json();

      const list = data.candidates || [];
      setCandidates(list);
      setResumeCount(data.resume_count || 0);
      setChunkCount(data.chunk_count || 0);
      setHistory(data.history || []);
      setGithubUsername(data.github_username || "");
      setSelected((prev) => (["All", ...list].includes(prev) ? prev : "All"));
      setMeCandidate(list.includes(data.me_candidate) ? data.me_candidate : (list[0] || ""));
    } catch {
      setStatus("Unable to load app state. Refresh and try again.");
    }
  }

  function onPickFiles(e) {
    setFiles(Array.from(e.target.files || []));
  }

  function onDropFiles(e) {
    e.preventDefault();
    setDragOver(false);
    const dropped = Array.from(e.dataTransfer.files || []).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (!dropped.length) {
      setStatus("Only PDF files are accepted.");
      return;
    }
    setFiles(dropped);
    setStatus("");
  }

  async function uploadFiles() {
    if (!files.length || busy) {
      setStatus("Choose at least one PDF resume to process.");
      return;
    }

    setBusy(true);
    setStatus("Processing resumes...");

    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Upload failed.");

      const processed = data.processed || [];
      const skipped = data.skipped || [];
      setUploadReport({ processed, skipped });
      setStatus(`Processed ${processed.length} file(s). Skipped ${skipped.length}.`);

      setFiles([]);
      if (fileRef.current) fileRef.current.value = "";
      await loadState(mode);
    } catch (err) {
      setStatus(err.message || "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function clearChat() {
    if (busy) return;
    setBusy(true);
    setStatus("");
    try {
      await fetch("/api/clear-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      setPendingUserMessage("");
      setAwaitingReply(false);
      await loadState(mode);
    } finally {
      setBusy(false);
    }
  }

  async function clearData() {
    if (busy) return;
    setBusy(true);
    setStatus("");
    try {
      await fetch("/api/clear-data", { method: "POST" });
      setFiles([]);
      setUploadReport({ processed: [], skipped: [] });
      setPendingUserMessage("");
      setAwaitingReply(false);
      if (fileRef.current) fileRef.current.value = "";
      await loadState(mode);
    } finally {
      setBusy(false);
    }
  }

  async function sendMessage(e) {
    e.preventDefault();
    const text = message.trim();
    if (!text || busy) return;

    if (!hasResumes) {
      setStatus("Upload and process resumes first from the Resume Intake panel.");
      return;
    }

    setBusy(true);
    setStatus("");
    setMessage("");
    setPendingUserMessage(text);
    setAwaitingReply(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          selected,
          me_candidate: meCandidate,
          github_username: githubUsername.trim(),
          message: text,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Message failed.");

      if (data.state) {
        const list = data.state.candidates || [];
        setCandidates(list);
        setResumeCount(data.state.resume_count || 0);
        setChunkCount(data.state.chunk_count || 0);
        setHistory(data.state.history || []);
        setGithubUsername(data.state.github_username || githubUsername);
        setSelected((prev) => (["All", ...list].includes(prev) ? prev : "All"));
        setMeCandidate(list.includes(data.state.me_candidate) ? data.state.me_candidate : (list[0] || ""));
      }
      setPendingUserMessage("");
      setAwaitingReply(false);
    } catch (err) {
      setPendingUserMessage("");
      setAwaitingReply(false);
      setStatus(err.message || "Message failed.");
    } finally {
      setBusy(false);
    }
  }

  const statusTone = useMemo(() => {
    const msg = (status || "").toLowerCase();
    if (!msg) return "";
    if (msg.includes("failed") || msg.includes("unable") || msg.includes("error") || msg.includes("first")) return "text-rose-700";
    return "text-emerald-700";
  }, [status]);

  return (
    <div className="min-h-screen p-4 md:p-6">
      <div className="mx-auto max-w-[1700px]">
        <header className="rounded-3xl border border-stroke bg-white/90 p-5 shadow-bloom backdrop-blur">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="font-display text-3xl md:text-4xl tracking-tight">Recruiter Copilot</p>
              <p className="mt-1 text-sm md:text-base text-muted">Resume intelligence + conversational interview flow in one workspace.</p>
            </div>
            <div className="inline-flex rounded-2xl border border-stroke bg-panelSoft p-1">
              <button
                onClick={() => setMode("recruiter")}
                className="rounded-xl px-4 py-2 text-sm font-semibold transition"
                style={mode === "recruiter"
                  ? { background: "#2152ff", color: "#fff", boxShadow: "0 8px 22px rgba(33,82,255,0.28)" }
                  : { color: "#5d6b8a", background: "transparent" }}
              >
                Recruiter Mode
              </button>
              <button
                onClick={() => setMode("me")}
                className="rounded-xl px-4 py-2 text-sm font-semibold transition"
                style={mode === "me"
                  ? { background: "#08a88d", color: "#fff", boxShadow: "0 8px 22px rgba(8,168,141,0.28)" }
                  : { color: "#5d6b8a", background: "transparent" }}
              >
                Me Mode
              </button>
            </div>
          </div>
        </header>

        <div className="mt-5 grid gap-4 xl:grid-cols-[360px_1fr]">
          <aside className="space-y-4">
            <section className="rounded-3xl border border-stroke bg-white p-4 shadow-card">
              <div className="flex items-center justify-between mb-3">
                <p className="font-display text-lg">Resume Intake</p>
                <span className="rounded-full bg-brand/10 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-brand">Step 1</span>
              </div>

              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDropFiles}
                className={`rounded-2xl border-2 border-dashed p-4 transition ${dragOver ? "border-brand bg-brand/5" : "border-stroke bg-panelSoft"}`}
              >
                <p className="text-sm font-semibold">Drop PDF resumes here</p>
                <p className="text-xs text-muted mt-1">or browse files and process them to add candidates.</p>
                <input
                  ref={fileRef}
                  type="file"
                  accept="application/pdf"
                  multiple
                  className="hidden"
                  onChange={onPickFiles}
                />
                <button
                  onClick={() => fileRef.current && fileRef.current.click()}
                  className="mt-3 rounded-xl border border-stroke bg-white px-3 py-2 text-sm font-semibold hover:bg-slate-50"
                >
                  Browse PDFs
                </button>
              </div>

              <div className="mt-3 rounded-2xl border border-stroke bg-panelSoft p-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted">Step 2: Selected files</p>
                {files.length ? (
                  <ul className="mt-2 space-y-1 max-h-28 overflow-auto chat-scroll text-sm">
                    {files.map((f) => (
                      <li key={f.name} className="truncate">{f.name}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 text-sm text-muted">No files selected yet.</p>
                )}
              </div>

              <div className="mt-3 rounded-2xl border border-brand/25 bg-brand/5 p-3">
                <p className="text-xs uppercase tracking-[0.14em] text-brand mb-2">Step 3: Process resumes</p>
                <button
                  disabled={busy}
                  onClick={uploadFiles}
                  className="w-full rounded-xl px-4 py-3 text-sm font-bold transition"
                  style={busy
                    ? {
                        cursor: "not-allowed",
                        border: "1px solid #cbd5e1",
                        background: "#cbd5e1",
                        color: "#475569",
                        minHeight: 48,
                      }
                    : files.length
                      ? {
                          border: "1px solid #112a84",
                          background: "#112a84",
                          color: "#ffffff",
                          minHeight: 48,
                        }
                      : {
                          border: "1px solid #112a84",
                          background: "#ffffff",
                          color: "#112a84",
                          minHeight: 48,
                        }}
                >
                  {busy ? "Processing..." : files.length ? `Process ${files.length} Resume${files.length === 1 ? "" : "s"}` : "Process Resumes"}
                </button>
                <p className="mt-2 text-xs text-muted">
                  {files.length
                    ? "This creates candidate entries and searchable chunks."
                    : "Choose one or more PDFs in Step 1, then click Process Resumes."}
                </p>
              </div>

              {(uploadReport.processed.length > 0 || uploadReport.skipped.length > 0) && (
                <div className="mt-3 space-y-2">
                  {uploadReport.processed.length > 0 && (
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-2 text-sm text-emerald-800">
                      Processed: {uploadReport.processed.map((x) => x.candidate || x.file).join(", ")}
                    </div>
                  )}
                  {uploadReport.skipped.length > 0 && (
                    <div className="rounded-xl border border-amber-200 bg-amber-50 p-2 text-sm text-amber-800">
                      Skipped: {uploadReport.skipped.map((x) => x.file).join(", ")}
                    </div>
                  )}
                </div>
              )}
            </section>

            <section className="rounded-3xl border border-stroke bg-white p-4 shadow-card">
              <p className="font-display text-lg mb-3">Workspace</p>
              <div className="grid grid-cols-2 gap-2 mb-3">
                <StatTile label="Resumes" value={resumeCount} accent="brand" />
                <StatTile label="Chunks" value={chunkCount} accent="mint" />
              </div>

              {mode === "recruiter" && (
                <div className="mb-3">
                  <p className="text-xs uppercase tracking-[0.14em] text-muted mb-2">Recruiter Filter</p>
                  <select
                    value={selected}
                    onChange={(e) => setSelected(e.target.value)}
                    className="w-full rounded-xl border border-stroke bg-white px-3 py-2 text-sm"
                  >
                    <option value="All">All Candidates</option>
                    {candidates.map((name) => <option key={name} value={name}>{name}</option>)}
                  </select>
                </div>
              )}

              {mode === "me" && (
                <div className="space-y-3 mb-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.14em] text-muted mb-2">My Resume Profile</p>
                    <select
                      value={meCandidate}
                      onChange={(e) => setMeCandidate(e.target.value)}
                      className="w-full rounded-xl border border-stroke bg-white px-3 py-2 text-sm"
                    >
                      {candidates.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.14em] text-muted mb-2">GitHub Username</p>
                    <input
                      value={githubUsername}
                      onChange={(e) => setGithubUsername(e.target.value)}
                      className="w-full rounded-xl border border-stroke bg-white px-3 py-2 text-sm"
                      placeholder="e.g. octocat"
                    />
                  </div>
                </div>
              )}

              <div className="rounded-2xl border border-stroke bg-panelSoft p-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted">Candidates</p>
                {candidates.length ? (
                  <ul className="mt-2 space-y-1 max-h-32 overflow-auto chat-scroll text-sm">
                    {candidates.map((name) => (
                      <li key={name} className="truncate rounded-lg bg-white px-2 py-1 border border-stroke">{name}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 text-sm text-muted">No candidates yet. Process resumes first.</p>
                )}
              </div>

              <div className="mt-3 grid grid-cols-2 gap-2">
                <button
                  disabled={busy}
                  onClick={clearChat}
                  className="rounded-xl border border-stroke bg-white px-3 py-2 text-sm font-semibold disabled:opacity-50"
                >
                  Clear Chat
                </button>
                <button
                  disabled={busy}
                  onClick={clearData}
                  className="rounded-xl border border-rose-300 bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700 disabled:opacity-50"
                >
                  Clear Data
                </button>
              </div>
            </section>
          </aside>

          <main className="flex min-h-[74vh] flex-col overflow-hidden rounded-3xl border border-stroke bg-white shadow-bloom">
            <header className="border-b border-stroke bg-[linear-gradient(120deg,#f6f8ff,#f1fff9)] px-5 py-4">
              <p className="font-display text-2xl tracking-tight">{panelTitle}</p>
              <p className="mt-1 text-sm text-muted">{panelSubtitle}</p>
            </header>

            {!hasResumes && (
              <div className="mx-4 mt-4 rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                Process at least one resume in the left panel before chatting.
              </div>
            )}

            <section ref={chatRef} className="chat-scroll flex-1 space-y-3 overflow-y-auto px-4 py-4 md:px-6">
              {history.length === 0 && !pendingUserMessage && !awaitingReply && (
                <div className="rounded-2xl border border-dashed border-stroke bg-panelSoft px-4 py-4 text-sm text-muted">
                  {emptyHint}
                </div>
              )}

              {history.map((entry, idx) => (
                <React.Fragment key={idx}>
                  <MessageBubble mode={mode} role="user" text={entry.q} />
                  <MessageBubble mode={mode} role="assistant" text={entry.a} />
                </React.Fragment>
              ))}

              {pendingUserMessage && <MessageBubble mode={mode} role="user" text={pendingUserMessage} />}
              {awaitingReply && <TypingBubble mode={mode} />}
            </section>

            <form onSubmit={sendMessage} className="border-t border-stroke bg-panelSoft px-3 py-3 md:px-4">
              <div className="mx-auto flex max-w-5xl gap-2">
                <input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder={inputPlaceholder}
                  className="flex-1 rounded-xl border border-stroke bg-white px-4 py-3 text-[15px]"
                />
                <button
                  disabled={busy || !message.trim()}
                  className="rounded-xl bg-gradient-to-r from-brand to-brandDeep px-6 py-3 text-sm font-bold text-white disabled:opacity-50"
                >
                  Send
                </button>
              </div>
              {status && <p className={`mx-auto mt-2 max-w-5xl text-sm ${statusTone}`}>{status}</p>}
            </form>
          </main>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
