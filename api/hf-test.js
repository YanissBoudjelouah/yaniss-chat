module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const token = process.env.HF_TOKEN;
  if (!token) return res.status(500).json({ ok: false, error: "HF_TOKEN missing" });

  const r = await fetch("https://huggingface.co/api/whoami-v2", {
    headers: { Authorization: `Bearer ${token}` }
  });
  const ok = r.ok;
  const body = await r.text(); // on ne renvoie pas la clé, juste l'état

  res.status(ok ? 200 : 500).json({
    ok,
    note: ok ? "HF token valid (server-side)" : "HF token invalid or permissions issue",
    sample: ok ? "hidden" : body.slice(0, 200)
  });
};
