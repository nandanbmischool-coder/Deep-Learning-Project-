/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#0F1D24", 2: "#17303A", 3: "#24444F" },
        box: "#0A1115",
        paper: "#F2F4F1",
        line: "#D9DFDC",
        teal: { DEFAULT: "#0E7C86", strong: "#0A5E66", soft: "#DCEDEE" },
        amber: { DEFAULT: "#D98A1F", soft: "#FBEFD9" },
        brick: { DEFAULT: "#B4432E", soft: "#F6E1DC" },
        moss: { DEFAULT: "#2F7D5B", soft: "#DFEFE6" },
        mute: "#5B6B70",
      },
      fontFamily: {
        display: ['"Newsreader"', "Georgia", "serif"],
        sans: ['"IBM Plex Sans"', "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
      borderRadius: { panel: "6px" },
    },
  },
  plugins: [],
};
