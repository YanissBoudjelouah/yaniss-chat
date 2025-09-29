module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');

  const token = process.env.HF_TOKEN;
  if (!token) return res.status(500).json({ ok: false, error: "HF_TOKEN missing" });

  // Use a model that returns embeddings via /models
  const model = "thenlper/gte-small";

  try {
    const r = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        "x-wait-for-model": "true"
      },
      // Plain string input works with gte-small
      body: JSON.stringify({ inputs: "test embedding yaniss" })
    });

    const txt = await r.text();
    return res
      .status(r.ok ? 200 : 500)
      .json({ ok: r.ok, hint: r.ok ? "embeddings ok" : txt.slice(0, 300) });
  } catch (e) {
    return res.status(500).json({ ok: false, hint: String(e) });
  }
};
