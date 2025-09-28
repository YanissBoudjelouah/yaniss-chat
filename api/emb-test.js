module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const token = process.env.HF_TOKEN;
  if (!token) return res.status(500).json({ ok: false, error: "HF_TOKEN missing" });

  const model = process.env.HF_EMBEDDINGS_MODEL || "sentence-transformers/all-MiniLM-L6-v2";
  const resp = await fetch(`https://api-inference.huggingface.co/pipeline/feature-extraction/${model}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "x-wait-for-model": "true"
    },
    body: JSON.stringify({ inputs: "test embedding yaniss" })
  });

  const ok = resp.ok;
  const out = await resp.text();
  res.status(ok ? 200 : 500).json({
    ok,
    hint: ok ? "embeddings ok" : out.slice(0, 200)
  });
};
