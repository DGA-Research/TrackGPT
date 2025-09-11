function nsLine(c) {
  // Netscape cookies.txt line
  const domain = c.domain.startsWith(".") ? c.domain : "." + c.domain;
  const includeSub = domain.startsWith(".") ? "TRUE" : "FALSE";
  const path = c.path || "/";
  const secure = c.secure ? "TRUE" : "FALSE";
  const expiry = Math.floor(c.expirationDate || (Date.now()/1000 + 3600));
  return `${domain}\t${includeSub}\t${path}\t${secure}\t${expiry}\t${c.name}\t${c.value}\n`;
}

async function exportNetscape() {
  const cookies = await chrome.cookies.getAll({ domain: ".youtube.com" });
  let txt = "# Netscape HTTP Cookie File\n";
  for (const c of cookies) txt += nsLine(c);
  return txt;
}

async function sendCookies() {
  const log = (m) => (document.querySelector("#log").textContent = m);
  try {
    const endpoint = document.querySelector("#endpoint").value.trim();
    const otc = document.querySelector("#otc").value.trim();
    if (!endpoint || !otc) return log("Enter endpoint and one-time code.");

    // Optional: ensure a YouTube tab exists (helps some browsers)
    const tabs = await chrome.tabs.query({ url: "*://*.youtube.com/*" });
    if (tabs.length === 0) log("Tip: open youtube.com in a tab.");

    const netscape = await exportNetscape();
    const cookies_b64 = btoa(unescape(encodeURIComponent(netscape)));

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ otc, cookies_b64 })
    });

    if (!res.ok) {
      const t = await res.text();
      return log(`Upload failed: ${res.status} ${t}`);
    }
    log("Cookies sent! Return to TrackGPT and press ‘Claim cookies’. ");
  } catch (e) {
    console.error(e);
    document.querySelector("#log").textContent = "Error: " + (e && e.message || e);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  // remember endpoint between uses
  const { endpoint } = await chrome.storage.local.get(["endpoint"]);
  if (endpoint) document.querySelector("#endpoint").value = endpoint;
  document.querySelector("#endpoint").addEventListener("change", e => {
    chrome.storage.local.set({ endpoint: e.target.value.trim() });
  });
  document.querySelector("#send").addEventListener("click", sendCookies);
});
