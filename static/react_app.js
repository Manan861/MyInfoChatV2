const { useEffect, useMemo, useRef, useState } = React;
const { createPortal } = ReactDOM;

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

// ----- Toasts (render to #toast-container)
function ToastItem({ id, message, type, onDismiss }) {
  const isError = type === "error";
  useEffect(() => {
    const t = setTimeout(onDismiss, 5000);
    return () => clearTimeout(t);
  }, [id, onDismiss]);
  return (
    <div
      role="alert"
      className={`toast-enter pointer-events-auto rounded-xl border px-4 py-3 text-sm font-medium shadow-lg ${
        isError
          ? "border-rose-200 bg-rose-50 text-rose-800"
          : "border-emerald-200 bg-white text-ink shadow-card"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <span>{message}</span>
        <button
          type="button"
          onClick={onDismiss}
          className="shrink-0 rounded-lg p-1 hover:bg-black/5 focus-visible:ring-2 focus-visible:ring-brand"
          aria-label="Dismiss"
        >
          <span aria-hidden>×</span>
        </button>
      </div>
    </div>
  );
}

function ToastContainer({ toasts, removeToast }) {
  const container = document.getElementById("toast-container");
  if (!container) return null;
  return createPortal(
    <div className="flex flex-col gap-2">
      {toasts.map((t) => (
        <ToastItem
          key={t.id}
          id={t.id}
          message={t.message}
          type={t.type}
          onDismiss={() => removeToast(t.id)}
        />
      ))}
    </div>,
    container
  );
}

// ----- Avatar
function Avatar({ role }) {
  const isUser = role === "user";
  const style = isUser
    ? {
        background: "linear-gradient(135deg, #2563eb, #1d4ed8)",
        color: "#ffffff",
        border: "1px solid rgba(29, 78, 216, 0.4)",
        boxShadow: "0 4px 12px rgba(37, 99, 235, 0.3)",
      }
    : {
        background: "#ffffff",
        color: "#0f172a",
        border: "1px solid #c8d4e8",
        boxShadow: "0 4px 12px rgba(15, 23, 42, 0.08)",
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
        ...style,
      }}
      aria-hidden
    >
      {isUser ? "You" : "AI"}
    </div>
  );
}

// ----- Message bubble with copy (assistant only)
function MessageBubble({ mode, role, text }) {
  const isUser = role === "user";
  const [copied, setCopied] = useState(false);
  const label = isUser ? "Recruiter" : (mode === "me" ? "Candidate" : "Assistant");
  const bubbleStyle = isUser
    ? {
        background: "linear-gradient(135deg, #2563eb, #1d4ed8)",
        color: "#ffffff",
        border: "1px solid rgba(37, 99, 235, 0.35)",
        boxShadow: "0 4px 14px rgba(37, 99, 235, 0.25)",
      }
    : mode === "me"
      ? {
          background: "#ffffff",
          color: "#0f172a",
          border: "1px solid rgba(5, 150, 105, 0.3)",
          boxShadow: "0 4px 14px rgba(5, 150, 105, 0.08)",
        }
      : {
          background: "#ffffff",
          color: "#0f172a",
          border: "1px solid #c8d4e8",
          boxShadow: "0 4px 14px rgba(15, 23, 42, 0.06)",
        };

  async function copyToClipboard() {
    try {
      await navigator.clipboard.writeText(text || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (_) {}
  }

  return (
    <div className={`message-rise flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && <Avatar role="assistant" />}
      <div
        style={{
          maxWidth: "86%",
          borderRadius: 18,
          padding: "14px 18px",
          ...bubbleStyle,
        }}
        className="group relative"
      >
        <div className="flex items-center justify-between gap-2 mb-1">
          <p
            style={{
              fontSize: 11,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              fontWeight: 700,
              color: isUser ? "#bfdbfe" : "#64748b",
            }}
          >
            {label}
          </p>
          {!isUser && (
            <button
              type="button"
              onClick={copyToClipboard}
              className="opacity-70 hover:opacity-100 rounded-lg p-1.5 hover:bg-black/5 transition focus-visible:ring-2 focus-visible:ring-brand"
              title={copied ? "Copied!" : "Copy"}
              aria-label={copied ? "Copied" : "Copy message"}
            >
              {copied ? (
                <span className="text-xs font-semibold text-emerald-600">Copied!</span>
              ) : (
                <span className="text-muted" aria-hidden>⎘</span>
              )}
            </button>
          )}
        </div>
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
    <div className="message-rise flex gap-3 justify-start" aria-live="polite" aria-busy="true">
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
        background: "linear-gradient(135deg, #2563eb, #1d4ed8)",
        color: "#ffffff",
        border: "1px solid rgba(29, 78, 216, 0.4)",
        boxShadow: "0 4px 14px rgba(37, 99, 235, 0.25)",
      }
    : {
        background: "linear-gradient(135deg, #ecfdf5, #d1fae5)",
        color: "#047857",
        border: "1px solid rgba(5, 150, 105, 0.3)",
        boxShadow: "0 4px 14px rgba(5, 150, 105, 0.1)",
      };
  return (
    <div className="rounded-2xl px-4 py-3" style={style}>
      <p className="text-[11px] uppercase tracking-[0.16em] font-semibold opacity-80">{label}</p>
      <p className="font-display text-3xl mt-1 leading-none">{value}</p>
    </div>
  );
}

// ----- Confirm modal
function ConfirmModal({ open, title, body, confirmLabel, variant, onConfirm, onCancel }) {
  if (!open) return null;
  const isDanger = variant === "danger";
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 drawer-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      aria-describedby="modal-desc"
      onClick={onCancel}
    >
      <div
        className="rounded-2xl border border-stroke bg-white p-6 max-w-md w-full"
        style={{ boxShadow: "0 12px 40px rgba(15, 23, 42, 0.15)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="modal-title" className="font-display text-xl tracking-tight text-ink">{title}</h2>
        <p id="modal-desc" className="mt-2 text-sm text-muted">{body}</p>
        <div className="mt-6 flex gap-3 justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-xl border border-stroke bg-white px-4 py-2.5 text-sm font-semibold hover:bg-panelSoft focus-visible:ring-2 focus-visible:ring-brand"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`rounded-xl px-4 py-2.5 text-sm font-semibold text-white focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-brand ${
              isDanger ? "bg-rose-600 hover:bg-rose-700" : "bg-brand hover:bg-brand-deep"
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

// ----- Loading skeleton for sidebar
function SidebarSkeleton() {
  return (
    <aside className="space-y-4">
      <section className="card-panel p-5 animate-pulse">
        <div className="h-6 w-32 rounded-lg bg-slate-200 mb-4" />
        <div className="h-28 rounded-2xl bg-slate-100 mb-3" />
        <div className="h-20 rounded-2xl bg-slate-100 mb-3" />
        <div className="h-16 rounded-2xl bg-slate-100" />
      </section>
      <section className="card-panel p-5 animate-pulse">
        <div className="h-6 w-24 rounded-lg bg-slate-200 mb-3" />
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div className="h-20 rounded-2xl bg-slate-100" />
          <div className="h-20 rounded-2xl bg-slate-100" />
        </div>
        <div className="h-10 rounded-xl bg-slate-100 mb-3" />
        <div className="h-28 rounded-2xl bg-slate-100" />
      </section>
    </aside>
  );
}

// ----- Suggested prompts (chips)
const RECRUITER_PROMPTS = [
  "Who has the strongest backend experience?",
  "Compare top candidates for a senior role.",
  "Who has Python and AWS experience?",
  "Summarize all candidates in one paragraph.",
];
const ME_PROMPTS = [
  "Tell me about yourself",
  "Walk me through your experience",
  "What are your key strengths?",
  "Share your GitHub or side projects",
];

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
  const [initialLoad, setInitialLoad] = useState(true);
  const [busy, setBusy] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploadReport, setUploadReport] = useState({ processed: [], skipped: [] });
  const [pendingUserMessage, setPendingUserMessage] = useState("");
  const [awaitingReply, setAwaitingReply] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [confirmModal, setConfirmModal] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const fileRef = useRef(null);
  const chatRef = useRef(null);
  const inputRef = useRef(null);

  const hasResumes = resumeCount > 0;

  const addToast = (message, type = "info") => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, message, type }]);
  };
  const removeToast = (id) => setToasts((prev) => prev.filter((t) => t.id !== id));

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

  useEffect(() => {
    if (sidebarOpen) {
      const onEscape = (e) => e.key === "Escape" && setSidebarOpen(false);
      document.addEventListener("keydown", onEscape);
      return () => document.removeEventListener("keydown", onEscape);
    }
  }, [sidebarOpen]);

  async function loadState(requestMode) {
    setInitialLoad(true);
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
      addToast("Unable to load app state. Refresh and try again.", "error");
    } finally {
      setInitialLoad(false);
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
      addToast("Only PDF files are accepted.", "error");
      return;
    }
    setFiles(dropped);
  }

  function removeFile(index) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  async function uploadFiles() {
    if (!files.length || busy) {
      addToast("Choose at least one PDF resume to process.", "error");
      return;
    }
    setBusy(true);
    addToast("Processing resumes…", "info");
    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Upload failed.");
      const processed = data.processed || [];
      const skipped = data.skipped || [];
      setUploadReport({ processed, skipped });
      setFiles([]);
      if (fileRef.current) fileRef.current.value = "";
      await loadState(mode);
      addToast(`Processed ${processed.length} file(s). Skipped ${skipped.length}.`, "info");
    } catch (err) {
      addToast(err.message || "Upload failed.", "error");
    } finally {
      setBusy(false);
    }
  }

  async function clearChat() {
    setConfirmModal(null);
    if (busy) return;
    setBusy(true);
    try {
      await fetch("/api/clear-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      setPendingUserMessage("");
      setAwaitingReply(false);
      await loadState(mode);
      addToast("Chat cleared.", "info");
    } finally {
      setBusy(false);
    }
  }

  async function clearData() {
    setConfirmModal(null);
    if (busy) return;
    setBusy(true);
    try {
      await fetch("/api/clear-data", { method: "POST" });
      setFiles([]);
      setUploadReport({ processed: [], skipped: [] });
      setPendingUserMessage("");
      setAwaitingReply(false);
      if (fileRef.current) fileRef.current.value = "";
      await loadState(mode);
      addToast("All data cleared.", "info");
    } finally {
      setBusy(false);
    }
  }

  async function sendMessage(e, suggestedText) {
    if (e) e.preventDefault();
    const text = (suggestedText != null ? suggestedText : message).trim();
    if (!text || busy) return;
    if (!hasResumes) {
      addToast("Upload and process resumes first from the Resume Intake panel.", "error");
      return;
    }
    setBusy(true);
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
      setTimeout(() => inputRef.current?.focus(), 100);
    } catch (err) {
      setPendingUserMessage("");
      setAwaitingReply(false);
      addToast(err.message || "Message failed.", "error");
    } finally {
      setBusy(false);
    }
  }

  function onSuggestedPrompt(prompt) {
    setMessage(prompt);
    inputRef.current?.focus();
  }

  const sidebarContent = (
    <>
      <section className="card-panel p-5">
        <div className="flex items-center justify-between mb-4">
          <p className="font-display text-lg font-semibold text-ink">Resume Intake</p>
          <span className="rounded-full bg-blue-100 px-2.5 py-1 text-[11px] font-bold uppercase tracking-wider text-blue-700">Step 1</span>
        </div>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDropFiles}
          className={`rounded-2xl border-2 border-dashed p-5 transition-all duration-200 ${dragOver ? "border-blue-500 bg-blue-50 scale-[1.01]" : "border-stroke bg-slate-50/80"}`}
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
            aria-label="Select PDF files"
          />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="mt-3 rounded-xl border border-stroke bg-white px-3 py-2 text-sm font-semibold hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-brand"
          >
            Browse PDFs
          </button>
        </div>
        <div className="mt-3 rounded-2xl border border-stroke bg-slate-50/80 p-3">
          <p className="text-xs uppercase tracking-wider font-semibold text-muted">Step 2: Selected files</p>
          {files.length ? (
            <ul className="mt-2 space-y-1 max-h-28 overflow-auto chat-scroll text-sm">
              {files.map((f, i) => (
                <li key={`${f.name}-${i}`} className="flex items-center justify-between gap-2 rounded-lg bg-white px-2 py-1 border border-stroke">
                  <span className="truncate min-w-0">{f.name}</span>
                  <button
                    type="button"
                    onClick={() => removeFile(i)}
                    className="shrink-0 rounded p-1 hover:bg-rose-50 text-muted hover:text-rose-600"
                    aria-label={`Remove ${f.name}`}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-muted">No files selected yet.</p>
          )}
        </div>
        <div className="mt-3 rounded-2xl border-2 border-blue-200 bg-blue-50/80 p-4">
          <p className="text-xs uppercase tracking-wider font-semibold text-blue-700 mb-2">Step 3: Process resumes</p>
          <button
            type="button"
            disabled={busy}
            onClick={uploadFiles}
            className="w-full rounded-xl px-4 py-3.5 text-sm font-bold transition disabled:cursor-not-allowed focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
            style={busy
              ? { border: "1px solid #cbd5e1", background: "#e2e8f0", color: "#64748b", minHeight: 52 }
              : files.length
                ? { border: "none", background: "linear-gradient(135deg, #2563eb, #1d4ed8)", color: "#fff", minHeight: 52, boxShadow: "0 4px 14px rgba(37, 99, 235, 0.35)" }
                : { border: "2px solid #2563eb", background: "#fff", color: "#1d4ed8", minHeight: 52 }}
          >
            {busy ? "Processing…" : files.length ? `Process ${files.length} Resume${files.length === 1 ? "" : "s"}` : "Process Resumes"}
          </button>
          <p className="mt-2 text-xs text-muted">
            {files.length ? "This creates candidate entries and searchable chunks." : "Choose one or more PDFs in Step 1, then click Process Resumes."}
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

      <section className="card-panel p-5">
        <p className="font-display text-lg font-semibold text-ink mb-4">Workspace</p>
        <div className="grid grid-cols-2 gap-2 mb-3">
          <StatTile label="Resumes" value={resumeCount} accent="brand" />
          <StatTile label="Chunks" value={chunkCount} accent="mint" />
        </div>
        {mode === "recruiter" && (
          <div className="mb-3">
            <label htmlFor="recruiter-filter" className="text-xs uppercase tracking-[0.14em] text-muted mb-2 block">Recruiter Filter</label>
            <select
              id="recruiter-filter"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="w-full rounded-xl border border-stroke bg-white px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-brand"
              aria-label="Filter by candidate"
            >
              <option value="All">All Candidates</option>
              {candidates.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </div>
        )}
        {mode === "me" && (
          <div className="space-y-3 mb-3">
            <div>
              <label htmlFor="me-profile" className="text-xs uppercase tracking-[0.14em] text-muted mb-2 block">My Resume Profile</label>
              <select
                id="me-profile"
                value={meCandidate}
                onChange={(e) => setMeCandidate(e.target.value)}
                className="w-full rounded-xl border border-stroke bg-white px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-brand"
                aria-label="Select your profile"
              >
                {candidates.map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor="github-username" className="text-xs uppercase tracking-[0.14em] text-muted mb-2 block">GitHub Username</label>
              <input
                id="github-username"
                value={githubUsername}
                onChange={(e) => setGithubUsername(e.target.value)}
                className="w-full rounded-xl border border-stroke bg-white px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-brand"
                placeholder="e.g. octocat"
                aria-label="GitHub username"
              />
            </div>
          </div>
        )}
        <div className="rounded-2xl border border-stroke bg-slate-50/80 p-3">
          <p className="text-xs uppercase tracking-wider font-semibold text-muted">Candidates</p>
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
            type="button"
            disabled={busy}
            onClick={() => setConfirmModal({ key: "clear-chat", title: "Clear chat", body: "This will remove all messages in this conversation. You can keep uploading and chatting.", confirmLabel: "Clear chat", variant: "default" })}
            className="rounded-xl border-2 border-stroke bg-white px-3 py-2.5 text-sm font-semibold hover:bg-slate-50 disabled:opacity-50 focus-visible:ring-2 focus-visible:ring-blue-500"
          >
            Clear Chat
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => setConfirmModal({ key: "clear-data", title: "Clear all data", body: "This will delete all resumes and chat history. This cannot be undone.", confirmLabel: "Clear all data", variant: "danger" })}
            className="rounded-xl border-2 border-rose-300 bg-rose-50 px-3 py-2.5 text-sm font-semibold text-rose-700 hover:bg-rose-100 disabled:opacity-50 focus-visible:ring-2 focus-visible:ring-rose-500"
          >
            Clear Data
          </button>
        </div>
      </section>
    </>
  );

  return (
    <div className="min-h-screen p-4 md:p-6">
      <ToastContainer toasts={toasts} removeToast={removeToast} />
      <ConfirmModal
        open={confirmModal?.key === "clear-chat"}
        title={confirmModal?.title}
        body={confirmModal?.body}
        confirmLabel={confirmModal?.confirmLabel}
        variant={confirmModal?.variant}
        onConfirm={clearChat}
        onCancel={() => setConfirmModal(null)}
      />
      <ConfirmModal
        open={confirmModal?.key === "clear-data"}
        title={confirmModal?.title}
        body={confirmModal?.body}
        confirmLabel={confirmModal?.confirmLabel}
        variant={confirmModal?.variant}
        onConfirm={clearData}
        onCancel={() => setConfirmModal(null)}
      />

      <div className="mx-auto max-w-[1700px]">
        <header className="header-bar rounded-3xl p-6 md:p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setSidebarOpen((o) => !o)}
                className="xl:hidden rounded-xl bg-white/15 hover:bg-white/25 p-2.5 text-white focus-visible:ring-2 focus-visible:ring-white/80"
                aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
                aria-expanded={sidebarOpen}
              >
                <span className="text-lg" aria-hidden>{sidebarOpen ? "✕" : "☰"}</span>
              </button>
              <div>
                <h1 className="font-display text-2xl md:text-3xl font-bold tracking-tight text-white">Recruiter Copilot</h1>
                <p className="mt-1 text-sm md:text-base text-blue-100">Resume intelligence + conversational interview flow in one workspace.</p>
              </div>
            </div>
            <div className="inline-flex rounded-2xl bg-white/10 p-1.5 backdrop-blur-sm" role="tablist" aria-label="Mode">
              <button
                role="tab"
                aria-selected={mode === "recruiter"}
                onClick={() => setMode("recruiter")}
                className="rounded-xl px-5 py-2.5 text-sm font-semibold transition focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-[#1e3a5f]"
                style={mode === "recruiter" ? { background: "#fff", color: "#1d4ed8", boxShadow: "0 2px 8px rgba(0,0,0,0.15)" } : { color: "rgba(255,255,255,0.9)", background: "transparent" }}
              >
                Recruiter Mode
              </button>
              <button
                role="tab"
                aria-selected={mode === "me"}
                onClick={() => setMode("me")}
                className="rounded-xl px-5 py-2.5 text-sm font-semibold transition focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-[#1e3a5f]"
                style={mode === "me" ? { background: "#059669", color: "#fff", boxShadow: "0 2px 8px rgba(5,150,105,0.4)" } : { color: "rgba(255,255,255,0.9)", background: "transparent" }}
              >
                Me Mode
              </button>
            </div>
          </div>
        </header>

        <div className="mt-5 grid gap-4 xl:grid-cols-[360px_1fr]">
          {/* Sidebar: desktop always visible; mobile as overlay */}
          <div className={`xl:block ${sidebarOpen ? "fixed inset-0 z-40 xl:relative xl:inset-auto" : "hidden xl:block"}`}>
            {sidebarOpen && (
              <div
                className="absolute inset-0 drawer-backdrop xl:hidden"
                onClick={() => setSidebarOpen(false)}
                aria-hidden
              />
            )}
            <div className={`relative h-full xl:block ${sidebarOpen ? "bg-page overflow-y-auto p-4 max-h-screen xl:bg-transparent xl:p-0 xl:max-h-none" : ""}`}>
              {sidebarOpen && (
                <div className="flex justify-end xl:hidden mb-2">
                  <button
                    type="button"
                    onClick={() => setSidebarOpen(false)}
                    className="rounded-xl border border-stroke bg-white px-3 py-2 text-sm font-semibold"
                    aria-label="Close sidebar"
                  >
                    Close
                  </button>
                </div>
              )}
              {initialLoad ? <SidebarSkeleton /> : sidebarContent}
            </div>
          </div>

          <main
            id="main-content"
            className="card-panel flex min-h-[74vh] flex-col overflow-hidden"
            role="main"
          >
            <header className="border-b border-stroke bg-gradient-to-r from-slate-50 to-blue-50/50 px-6 py-5">
              <h2 className="font-display text-xl md:text-2xl font-bold tracking-tight text-ink">{panelTitle}</h2>
              <p className="mt-1.5 text-sm text-muted">{panelSubtitle}</p>
            </header>

            {!hasResumes && (
              <div className="mx-4 mt-4 rounded-2xl border-2 border-amber-200 bg-amber-50 px-5 py-4 text-sm font-medium text-amber-900 shadow-sm" role="alert">
                Upload and process at least one resume in the left panel before chatting.
              </div>
            )}

            <section
              ref={chatRef}
              className="chat-scroll flex-1 space-y-4 overflow-y-auto px-4 py-5 md:px-6"
              aria-label="Chat messages"
            >
              {history.length === 0 && !pendingUserMessage && !awaitingReply && (
                <div className="space-y-6 max-w-2xl">
                  <div className="empty-state-box px-6 py-8 text-center">
                    <div className="text-4xl mb-3 opacity-80" aria-hidden>💬</div>
                    <p className="text-base font-medium text-ink">{emptyHint}</p>
                    <p className="mt-2 text-sm text-muted">Or pick a suggestion below to get started.</p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-wider font-semibold text-muted mb-3">Suggested questions</p>
                    <div className="flex flex-wrap gap-2">
                      {(mode === "recruiter" ? RECRUITER_PROMPTS : ME_PROMPTS).map((p) => (
                        <button
                          key={p}
                          type="button"
                          disabled={busy || !hasResumes}
                          onClick={() => onSuggestedPrompt(p)}
                          className="prompt-chip rounded-xl border border-stroke bg-white px-4 py-2.5 text-sm text-ink hover:border-blue-400 hover:bg-blue-50/50 hover:text-blue-700 disabled:opacity-50 focus-visible:ring-2 focus-visible:ring-blue-500"
                        >
                          {p}
                        </button>
                      ))}
                    </div>
                  </div>
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

            <form onSubmit={(e) => sendMessage(e)} className="border-t border-stroke bg-gradient-to-b from-slate-50/80 to-white px-4 py-4 md:px-5">
              <div className="mx-auto flex max-w-4xl gap-3">
                <input
                  ref={inputRef}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder={inputPlaceholder}
                  className="input-ring flex-1 rounded-xl border-2 border-stroke bg-white px-4 py-3.5 text-[15px] placeholder:text-muted focus:border-blue-500 focus:outline-none"
                  aria-label="Message"
                  disabled={busy}
                  maxLength={2000}
                />
                <button
                  type="submit"
                  disabled={busy || !message.trim()}
                  className="rounded-xl px-6 py-3.5 text-sm font-bold text-white shadow-md hover:shadow-lg disabled:opacity-50 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-blue-500 transition-shadow"
                  style={{ background: "linear-gradient(135deg, #2563eb, #1d4ed8)", boxShadow: "0 4px 14px rgba(37, 99, 235, 0.35)" }}
                  aria-label="Send message"
                >
                  Send
                </button>
              </div>
              <p className="mx-auto mt-2 max-w-4xl text-xs text-muted text-right">{message.length}/2000</p>
            </form>
          </main>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
