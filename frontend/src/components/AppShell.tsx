import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { api } from "../lib/api";
import type { HealthInfo } from "../lib/types";
import { LibraryIcon, MoonIcon, SparkIcon, SunIcon, UploadIcon } from "./Icons";

const NAV = [
  { to: "/", label: "Document Library", icon: LibraryIcon, end: true },
  { to: "/upload", label: "Upload Document", icon: UploadIcon, end: false },
];

const PAGE_META: Record<string, { title: string; subtitle: string }> = {
  "/": {
    title: "Document Library",
    subtitle: "Search, filter and open every document KMRL has processed",
  },
  "/upload": {
    title: "Upload Document",
    subtitle: "Extraction, OCR and AI analysis run automatically after upload",
  },
};

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const stored = window.localStorage.getItem("kmrl-theme");
    return stored === "dark" ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("kmrl-theme", theme);
  }, [theme]);

  return { theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) };
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const { theme, toggle } = useTheme();
  const location = useLocation();
  const [health, setHealth] = useState<HealthInfo | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const meta =
    PAGE_META[location.pathname] ??
    (location.pathname.startsWith("/documents")
      ? { title: "Document Intelligence", subtitle: "Summary, actions, deadlines and source text" }
      : { title: "KMRL Document Intelligence", subtitle: "" });

  const engine = health?.capabilities.ai_analysis;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">KM</div>
          <div className="brand-text">
            <span className="brand-title">KMRL Docs</span>
            <span className="brand-sub">Intelligence</span>
          </div>
        </div>

        <nav className="nav">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              <item.icon size={16} className="nav-icon" />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="engine-card">
            <strong>Processing engine</strong>
            <div className="engine-row">
              <SparkIcon size={13} />
              <span>
                {engine === "claude"
                  ? `Claude · ${health?.capabilities.analysis_model}`
                  : engine === "rule-based"
                    ? "KMRL rule engine"
                    : "Checking…"}
              </span>
            </div>
            <div className="engine-row">
              <span
                className="dot"
                style={{ background: health?.capabilities.ocr ? "var(--brand-400)" : "var(--ink-400)" }}
              />
              <span>
                OCR {health?.capabilities.ocr ? `on · ${health.capabilities.ocr_languages}` : "unavailable"}
              </span>
            </div>
          </div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{meta.title}</h1>
            <span>{meta.subtitle}</span>
          </div>
          <div className="topbar-actions">
            <button
              type="button"
              className="btn btn-icon btn-ghost"
              onClick={toggle}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            >
              {theme === "dark" ? <SunIcon size={16} /> : <MoonIcon size={16} />}
            </button>
          </div>
        </header>
        <main className="page">{children}</main>
      </div>
    </div>
  );
}
