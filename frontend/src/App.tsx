import { useEffect, useState } from "react";
import { Navigate, NavLink, Outlet, Route, Routes, useParams } from "react-router-dom";

import { api } from "./api/client";
import type { Health } from "./api/types";
import { AppDataProvider, useAppData } from "./appData";
import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";

export default function App() {
  return (
    <AppDataProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/chat" replace />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="chat/:conversationId" element={<ChatPage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Route>
      </Routes>
    </AppDataProvider>
  );
}

function Layout() {
  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Hỏi đáp tài liệu</h1>
        <nav className="nav">
          <NavLink to="/chat" end={false}>
            Trò chuyện
          </NavLink>
          <NavLink to="/documents">Tài liệu</NavLink>
        </nav>
        <ConversationList />
        <div className="sidebar-footer">Hybrid search (bge-m3) · rerank · LangGraph agent (Gemini)</div>
      </aside>
      <main className="main">
        <HealthBanner />
        <Outlet />
      </main>
    </div>
  );
}

function ConversationList() {
  const { conversations } = useAppData();
  const { conversationId } = useParams();
  return (
    <div className="sidebar-section grow">
      <div className="row spread">
        <span className="sidebar-label">Lịch sử trò chuyện</span>
        <NavLink to="/chat" className="btn small" end>
          Mới
        </NavLink>
      </div>
      <div className="conversation-list">
        {conversations.length === 0 && <span className="muted small" style={{ margin: "0 4px" }}>Chưa có cuộc trò chuyện nào.</span>}
        {conversations.map((c) => (
          <NavLink
            key={c.id}
            to={`/chat/${c.id}`}
            className={c.id === conversationId ? "active" : undefined}
            title={c.title}
          >
            {c.title}
          </NavLink>
        ))}
      </div>
    </div>
  );
}

/** A thin warning bar when the backend, Elasticsearch or the models aren't available,
 * so problems show up before the user asks a question. */
function HealthBanner() {
  const [health, setHealth] = useState<Health | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let alive = true;
    const check = () =>
      api
        .health()
        .then((h) => alive && (setHealth(h), setUnreachable(false)))
        .catch(() => alive && setUnreachable(true));
    void check();
    const timer = window.setInterval(check, 30_000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  let message: string | null = null;
  if (unreachable) {
    message = "Không kết nối được tới backend. Kiểm tra backend đã chạy chưa (cổng 8000).";
  } else if (health && !health.database) {
    message = "Không kết nối được PostgreSQL (lịch sử chat). Kiểm tra Docker: docker compose up -d";
  } else if (health && !health.elasticsearch) {
    message = "Không kết nối được Elasticsearch. Kiểm tra Docker: docker compose up -d";
  } else if (health && !health.llm_key_configured) {
    message = "Chưa cấu hình GOOGLE_API_KEY trong .env, nên chưa thể trả lời câu hỏi.";
  }
  if (!message) return null;
  return <div className="notice warn banner small">{message}</div>;
}
