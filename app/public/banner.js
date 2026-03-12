addEventListener("DOMContentLoaded", () => {
  const s = getComputedStyle(document.documentElement);
  const a = s.getPropertyValue("--model-a").trim() || "oklch(0.65 0.15 220)";
  console.log(
    "%c\n  ██████╗         █████╗    ██████╗\n  ██╔══██╗       ██╔══██╗   ██╔════╝\n  ██████╔╝       ███████║   ██║  ███╗\n  ██╔═══╝ █████╗ ██╔══██║   ██║   ██║\n  ██║     ╚════╝ ██║  ██║ ▄ ╚██████╔╝\n  ╚═╝            ╚═╝  ╚═╝ ▀  ╚═════╝\n",
    "color:" + a + ";font-family:monospace;",
  );
  console.log(
    "%c microgpt-lab — par P-A.G ",
    "background:" + a + ";color:oklch(0.95 0 0);padding:6px 12px;border-radius:4px;font-weight:bold;font-family:monospace;",
  );
  console.log(
    "%c Curieux ? Le code est open-source \u2192 github.com/mon-atelier-ia ",
    "color:oklch(0.65 0.15 250);font-size:11px;",
  );
});
