import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const INTRO_LINES = [
  "안녕하세요 백엔드 개발자 홍준호입니다.",
  "방문해주셔서 감사합니다.",
  "제 go-commerce 프로젝트에 관해 궁금하신 것들을 질문해주세요.",
];
const PDF_ASSETS = {
  resume: {
    label: "이력서 보기",
    path: "/docs/resume.pdf",
  },
  portfolio: {
    label: "포트폴리오 보기",
    path: "/docs/portfolio.pdf",
  },
};
const GITHUB_URL = "https://github.com/SundaePorkCutlet/go-commerce";

function useTypingLines(lines, speedMs = 30, lineDelayMs = 350, replayDelayMs = 10000) {
  const [currentLine, setCurrentLine] = useState(0);
  const [currentText, setCurrentText] = useState("");
  const [doneLines, setDoneLines] = useState([]);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (currentLine >= lines.length) {
      setDone(true);
      const t = setTimeout(() => {
        setCurrentLine(0);
        setCurrentText("");
        setDoneLines([]);
        setDone(false);
      }, replayDelayMs);
      return () => clearTimeout(t);
    }
    const target = lines[currentLine];
    if (currentText.length < target.length) {
      const t = setTimeout(() => {
        setCurrentText(target.slice(0, currentText.length + 1));
      }, speedMs);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      setDoneLines((prev) => [...prev, target]);
      setCurrentLine((prev) => prev + 1);
      setCurrentText("");
    }, lineDelayMs);
    return () => clearTimeout(t);
  }, [currentLine, currentText, lines, speedMs, lineDelayMs, replayDelayMs]);

  return { doneLines, currentText, done };
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [service, setService] = useState("ALL");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [pdfModal, setPdfModal] = useState(null);
  const { doneLines, currentText, done } = useTypingLines(INTRO_LINES);

  const placeholders = useMemo(
    () => [
      "예) ORDERFC에서 CheckOutOrder는 어디 구현돼?",
      "예) Kafka 멱등성은 어떤 코드로 처리했나요?",
      "예) 결제 실패 시 재고 롤백은 어디서 발행되나요?",
    ],
    []
  );

  const onSubmit = async (e) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setQuestion("");

    try {
      const payload = { question: q };
      if (service !== "ALL") {
        payload.service = service;
      }
      const res = await fetch(`${API_BASE_URL}/api/v1/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      const botText = data?.answer ?? "응답이 비어 있습니다.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: botText, mode: data?.mode, confidence: data?.confidence },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `요청 실패: ${String(err)}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page">
      <section className="hero">
        {doneLines.map((line, idx) => (
          <p key={line} className="hero-line">
            {line}
            {done && idx === doneLines.length - 1 ? <span className="cursor">|</span> : null}
          </p>
        ))}
        {!done ? (
          <p className="hero-line">
            {currentText}
            <span className="cursor">|</span>
          </p>
        ) : null}
        <div className="hero-actions">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="doc-btn doc-btn-github"
          >
            GitHub 보기
          </a>
          <button
            type="button"
            className="doc-btn doc-btn-resume"
            onClick={() => setPdfModal("resume")}
          >
            {PDF_ASSETS.resume.label}
          </button>
          <button
            type="button"
            className="doc-btn doc-btn-portfolio"
            onClick={() => setPdfModal("portfolio")}
          >
            {PDF_ASSETS.portfolio.label}
          </button>
        </div>
      </section>

      <section className="chat-card">
        <div className="chat-header">
          <h2>go-commerce Q&A Chat</h2>
          <select value={service} onChange={(e) => setService(e.target.value)}>
            <option value="ALL">ALL</option>
            <option value="ORDERFC">ORDERFC</option>
            <option value="PAYMENTFC">PAYMENTFC</option>
            <option value="PRODUCTFC">PRODUCTFC</option>
            <option value="USERFC">USERFC</option>
          </select>
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <div className="hint">
              {placeholders.map((p) => (
                <p key={p}>{p}</p>
              ))}
            </div>
          )}
          {messages.map((m, idx) => (
            <div key={`${m.role}-${idx}`} className={`msg ${m.role}`}>
              {m.mode || m.confidence ? (
                <div className="meta-row">
                  {m.mode ? <div className="mode-tag">mode: {m.mode}</div> : null}
                  {m.confidence ? (
                    <div className={`confidence-badge confidence-${m.confidence}`}>
                      {m.confidence.toUpperCase()}
                    </div>
                  ) : null}
                </div>
              ) : null}
              <pre>{m.text}</pre>
            </div>
          ))}
          {loading && (
            <div className="msg assistant typing">
              <div className="typing-label">입력 중...</div>
              <div className="typing-dots" aria-label="assistant is typing">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
        </div>

        <form onSubmit={onSubmit} className="composer">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="go-commerce 프로젝트 질문을 입력하세요..."
          />
          <button type="submit" disabled={loading}>
            {loading ? "질문 중..." : "질문하기"}
          </button>
        </form>
      </section>

      {pdfModal ? (
        <div className="modal-backdrop" role="presentation" onClick={() => setPdfModal(null)}>
          <div className="pdf-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <div className="pdf-modal-header">
              <strong>{PDF_ASSETS[pdfModal].label}</strong>
              <div className="pdf-modal-actions">
                <a
                  href={encodeURI(PDF_ASSETS[pdfModal].path)}
                  target="_blank"
                  rel="noreferrer"
                  className="pdf-link"
                >
                  새 탭에서 열기
                </a>
                <button type="button" className="close-btn" onClick={() => setPdfModal(null)}>
                  닫기
                </button>
              </div>
            </div>
            <iframe
              title={`${pdfModal}-pdf`}
              src={encodeURI(PDF_ASSETS[pdfModal].path)}
              className="pdf-frame"
            />
          </div>
        </div>
      ) : null}
    </main>
  );
}
