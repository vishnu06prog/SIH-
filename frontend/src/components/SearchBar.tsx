import { useEffect, useRef, useState } from "react";
import { CloseIcon, SearchIcon } from "./Icons";

interface Props {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
}

/** Debounced search input — typing never fires a request per keystroke. */
export function SearchBar({ value, onChange, placeholder, autoFocus }: Props) {
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => setDraft(value), [value]);

  useEffect(() => {
    if (draft === value) return;
    const timer = window.setTimeout(() => onChange(draft), 280);
    return () => window.clearTimeout(timer);
  }, [draft, value, onChange]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "/" && document.activeElement?.tagName !== "INPUT") {
        event.preventDefault();
        inputRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="searchbar">
      <SearchIcon size={18} />
      <input
        ref={inputRef}
        type="search"
        value={draft}
        autoFocus={autoFocus}
        placeholder={placeholder ?? "Search documents…"}
        aria-label="Search documents"
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") onChange(draft);
          if (event.key === "Escape") {
            setDraft("");
            onChange("");
          }
        }}
      />
      {draft ? (
        <button
          type="button"
          className="searchbar-clear"
          aria-label="Clear search"
          onClick={() => {
            setDraft("");
            onChange("");
          }}
        >
          <CloseIcon size={14} />
        </button>
      ) : null}
    </div>
  );
}
