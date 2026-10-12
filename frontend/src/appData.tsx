import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

import { api, type ApiError } from "./api/client";
import type { ConversationSummary, DocumentSummary } from "./api/types";

/** Lists shown in several places (sidebar, scope picker, library), loaded once and
 * refreshed after anything that changes them. */
interface AppData {
  conversations: ConversationSummary[];
  refreshConversations: () => Promise<void>;
  documents: DocumentSummary[];
  documentsError: ApiError | null;
  refreshDocuments: () => Promise<void>;
}

const AppDataContext = createContext<AppData | null>(null);

export function AppDataProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [documentsError, setDocumentsError] = useState<ApiError | null>(null);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await api.conversations());
    } catch {
      // the sidebar just keeps its current list; the health banner explains outages
    }
  }, []);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await api.documents());
      setDocumentsError(null);
    } catch (error) {
      setDocumentsError(error as ApiError);
    }
  }, []);

  useEffect(() => {
    void refreshConversations();
    void refreshDocuments();
  }, [refreshConversations, refreshDocuments]);

  return (
    <AppDataContext.Provider
      value={{ conversations, refreshConversations, documents, documentsError, refreshDocuments }}
    >
      {children}
    </AppDataContext.Provider>
  );
}

export function useAppData(): AppData {
  const value = useContext(AppDataContext);
  if (!value) throw new Error("useAppData must be used inside AppDataProvider");
  return value;
}
