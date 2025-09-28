module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');

  const token = process.env.HF_TOKEN;
  if (!token) return res.status(500).json({ ok: false, error: "HF_TOKEN missing" });

  const model = process.env.HF_EMBEDDINGS_MODEL || "thenlper/gte-small";

  try {
    const resp = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        "x-wait-for-model": "true"
      },
      body: JSON.stringify({ inputs: "test embedding yaniss" })
    });

    const text = await resp.text();
    return res
      .status(resp.ok ? 200 : 500)
      .json({ ok: resp.ok, hint: resp.ok ? "embeddings ok" : text.slice(0, 200) });
  } catch (e) {
    return res.status(500).json({ ok: false, hint: String(e) });
  }
};
