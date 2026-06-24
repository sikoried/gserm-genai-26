import { NavLink, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/ChatPage.jsx";
import ComparisonPage from "./pages/ComparisonPage.jsx";

const navLink = ({ isActive }) => ({
  color: "#fff",
  textDecoration: "none",
  padding: "0.4rem 0.9rem",
  borderRadius: "6px",
  fontWeight: 600,
  fontSize: "0.9rem",
  background: isActive ? "rgba(255,255,255,0.22)" : "transparent",
});

export default function App() {
  return (
    <div style={styles.shell}>
      <nav style={styles.nav}>
        <span style={styles.brand}>Oracle</span>
        <div style={styles.links}>
          <NavLink to="/" end style={navLink}>
            Chat
          </NavLink>
          <NavLink to="/compare" style={navLink}>
            Comparison
          </NavLink>
        </div>
      </nav>
      <div style={styles.content}>
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/compare" element={<ComparisonPage />} />
        </Routes>
      </div>
    </div>
  );
}

const styles = {
  shell: { display: "flex", flexDirection: "column", height: "100%" },
  nav: {
    height: "52px",
    flexShrink: 0,
    background: "var(--hsg-green)",
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 1.25rem",
  },
  brand: { fontWeight: 700, fontSize: "1.05rem", letterSpacing: "0.02em" },
  links: { display: "flex", gap: "0.4rem" },
  content: { flex: 1, minHeight: 0, overflow: "hidden" },
};
