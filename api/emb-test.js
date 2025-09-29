module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');

  const token = process.env.HF_TOKEN;
  if (!token) {
    return res.status(500).json({ ok: false, hint: "HF_TOKEN missing" });
  }

  // Use a model that plays nicely with feature-extraction
  const model = "thenlper/gte-small";
  const url = `https://api-inference.huggingface.co/pipeline/feature-extraction/${model}`;

  try {
    const r = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        "x-wait-for-model": "true"
      },
      // Send a LIST of strings; this avoids the SentenceSimilarity auto-pipeline trap
      body: JSON.stringify({ inputs: ["debug embedding"] })
    });

    const text = await r.text();
    if (!r.ok) {
      return res.status(500).json({
        ok: false,
        status: r.status,
        url,
        hint: text.slice(0, 300)
      });
    }

    // r.ok: HF returns a JSON array of arrays (embeddings). We don’t dump it all.
    return res.status(200).json({ ok: true, hint: "embeddings ok" });
  } catch (e) {
    return res.status(500).json({ ok: false, hint: String(e) });
  }
};
