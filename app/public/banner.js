addEventListener("DOMContentLoaded", () => {
  const s = getComputedStyle(document.documentElement);
  const a = s.getPropertyValue("--model-a").trim() || "oklch(0.65 0.15 220)";
  const b = s.getPropertyValue("--model-b").trim() || a;
  console.log(
    "%c\n  ██████╗         █████╗    ██████╗\n  ██╔══██╗       ██╔══██╗   ██╔════╝\n  ██████╔╝       ███████║   ██║  ███╗\n  ██╔═══╝ █████╗ ██╔══██║   ██║   ██║\n  ██║     ╚════╝ ██║  ██║ ▄ ╚██████╔╝\n  ╚═╝            ╚═╝  ╚═╝ ▀  ╚═════╝\n",
    "background:linear-gradient(135deg," + a + "," + b + ");" +
    "-webkit-background-clip:text;" +
    "-webkit-text-fill-color:transparent;" +
    "color:" + a + ";" +
    "font-family:monospace;font-size:14px;line-height:1.4;",
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
