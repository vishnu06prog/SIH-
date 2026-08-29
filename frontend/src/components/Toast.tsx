import { createContext, useCallback, useContext, useMemo, useState } from "react";
import { AlertIcon, CheckIcon, CloseIcon } from "./Icons";

type ToastKind = "success" | "error" | "info";

interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  body?: string;
}

interface ToastApi {
  notify: (kind: ToastKind, title: string, body?: string) => void;
}

const ToastContext = createContext<ToastApi>({ notify: () => undefined });

export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const notify = useCallback(
    (kind: ToastKind, title: string, body?: string) => {
      const id = Date.now() + Math.random();
      setToasts((current) => [...current, { id, kind, title, body }]);
      window.setTimeout(() => dismiss(id), kind === "error" ? 7000 : 4200);
    },
    [dismiss],
  );

  const value = useMemo(() => ({ notify }), [notify]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toast-stack" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast toast-${toast.kind}`}>
            {toast.kind === "error" ? (
              <AlertIcon size={16} style={{ color: "var(--danger-600)", marginTop: 2 }} />
            ) : (
              <CheckIcon size={16} style={{ color: "var(--ok-700)", marginTop: 2 }} />
            )}
            <div style={{ minWidth: 0, flex: 1 }}>
              <strong>{toast.title}</strong>
              {toast.body ? <p>{toast.body}</p> : null}
            </div>
            <button
              className="btn-ghost"
              style={{ border: 0, background: "none", padding: 2, lineHeight: 1 }}
              onClick={() => dismiss(toast.id)}
              aria-label="Dismiss notification"
            >
              <CloseIcon size={14} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
