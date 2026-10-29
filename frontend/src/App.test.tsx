import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import { validateFile } from "./pages/Analyze";

vi.mock("./lib/api", async (orig) => {
  const mod = await orig<typeof import("./lib/api")>();
  return { ...mod, api: { get: vi.fn().mockRejectedValue(new Error("401")), post: vi.fn(), interceptors: mod.api.interceptors } };
});

import App from "./App";
import { AuthProvider } from "./lib/auth";

describe("validateFile", () => {
  it("accepts supported images", () => expect(validateFile(new File(["x"], "a.png", { type: "image/png" }))).toBeNull());
  it("rejects wrong types", () => expect(validateFile(new File(["x"], "a.pdf", { type: "application/pdf" }))).toMatch(/JPEG, PNG or WebP/));
  it("rejects empty files", () => expect(validateFile(new File([], "a.png", { type: "image/png" }))).toMatch(/empty/));
  it("rejects oversize files", () => {
    const big = new File(["x"], "a.png", { type: "image/png" });
    Object.defineProperty(big, "size", { value: 11 * 1024 * 1024 });
    expect(validateFile(big)).toMatch(/larger/);
  });
});

describe("routing", () => {
  it("shows the landing page with the research disclaimer", async () => {
    render(<MemoryRouter initialEntries={["/"]}><AuthProvider><App /></AuthProvider></MemoryRouter>);
    expect(await screen.findByRole("heading", { name: /see what the model saw/i })).toBeInTheDocument();
    expect(screen.getByText(/not a clinically validated diagnostic device/i)).toBeInTheDocument();
  });

  it("redirects protected pages to login when signed out", async () => {
    render(<MemoryRouter initialEntries={["/app"]}><AuthProvider><App /></AuthProvider></MemoryRouter>);
    expect(await screen.findByRole("heading", { name: /sign in/i })).toBeInTheDocument();
  });

  it("shows validation errors on the login form", async () => {
    render(<MemoryRouter initialEntries={["/login"]}><AuthProvider><App /></AuthProvider></MemoryRouter>);
    await userEvent.click(await screen.findByRole("button", { name: /sign in/i }));
    expect(await screen.findByText(/enter a valid email/i)).toBeInTheDocument();
  });
});
