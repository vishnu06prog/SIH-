import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { IntelligencePage } from "./pages/IntelligencePage";
import { LibraryPage } from "./pages/LibraryPage";
import { UploadPage } from "./pages/UploadPage";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<LibraryPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/documents/:documentId" element={<IntelligencePage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
