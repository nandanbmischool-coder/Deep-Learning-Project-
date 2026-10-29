import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import Layout from "./components/Layout";
import { Spinner } from "./components/ui";
import { useAuth } from "./lib/auth";
import Analyze from "./pages/Analyze";
import { Login, Register } from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import DatasetPage from "./pages/DatasetPage";
import Details from "./pages/Details";
import HistoryPage from "./pages/History";
import Landing from "./pages/Landing";
import Performance from "./pages/Performance";
import SettingsPage from "./pages/SettingsPage";
import Training from "./pages/Training";

function Protected({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const loc = useLocation();
  if (loading) return <div className="p-8"><Spinner /></div>;
  if (!user) return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/app" element={<Protected><Layout /></Protected>}>
        <Route index element={<Dashboard />} />
        <Route path="analyze" element={<Analyze />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="history/:id" element={<Details />} />
        <Route path="performance" element={<Performance />} />
        <Route path="dataset" element={<DatasetPage />} />
        <Route path="training" element={<Training />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
