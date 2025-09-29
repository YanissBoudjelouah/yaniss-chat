module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const token = process.env.HF_TOKEN;
  const model = "sentence-transformers/all-MiniLM-L6-v2";

  if (!token) return res.status(500).json({ step: "env", error: "HF_TOKEN missing" });

  try {
    const url = `https://api-inference.huggingface.co/pipeline/feature-extraction/${model}`;
    const r = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        "x-wait-for-model": "true"
      },
      body: JSON.stringify({ inputs: ["debug ping"] })
    });

    const text = await r.text();
    return res.status(200).json({
      status: r.status,
      ok: r.ok,
      url,
      note: "If ok=false, look at bodyStart for the reason.",
      bodyStart: text.slice(0, 500)
    });
  } catch (e) {
    return res.status(500).json({ step: "fetch", error: String(e) });
  }
};
