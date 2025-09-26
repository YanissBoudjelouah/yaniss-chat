// Serverless function for Vercel: place this file at /api/chat.js in a repo
// Env vars to set in Vercel: HF_TOKEN (required), optional HF_EMBEDDINGS_MODEL, HF_TEXT_MODEL
// Default models keep it light and free-ish. You can change them in Vercel → Settings → Environment Variables.

const HF_TOKEN = process.env.HF_TOKEN;
const EMB_MODEL = process.env.HF_EMBEDDINGS_MODEL || 'sentence-transformers/all-MiniLM-L6-v2'; // 384 dims
const GEN_MODEL = process.env.HF_TEXT_MODEL || 'google/flan-t5-base'; // text2text-generation

// ───────────────────────────────────────────────────────────────────────────────
// 1) Minimal corpus (replace with YOUR content). Keep it short & factual.
// You can also load from files; but embedding at cold start is fine for a tiny corpus.
const DOCS = [
  { id: 'about', text: `Ingénieur consultant spécialisé en CCaaS (Contact Center as a Service), intégrations Salesforce (Service Cloud) et téléphonie cloud. Focalisé sur des architectures simples, SSO (Single Sign-On)/MFA (Multi-Factor Authentication), et performance.` },
  { id: 'experience-1', text: `Consultant CX chez Devoteam (2022–2025). Projets Suez/SAUR: Amazon Connect, Genesys Cloud, CTI Salesforce, BYOC (Bring Your Own Carrier), monitoring, réduction MTTR (Mean Time To Repair).` },
  { id: 'projects-1', text: `Déploiement Amazon Connect multi-régions: SVI maintenable, routage, intégration Salesforce, optimisation coûts-minutes et latence.` },
  { id: 'skills', text: `Compétences: Amazon Connect, Genesys Cloud, Salesforce Service Cloud, Open CTI, SBC (Session Border Controller), SIP (Session Initiation Protocol), KPI (Key Performance Indicator).` }
];

// Cache embeddings between invocations (warm lambda)
let DOC_EMBEDS = null;

module.exports = async (req, res) => {
  // CORS: allow your static site to call this function
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method !== 'POST') return res.status(405).send('POST /api/chat');

  try {
    if (!HF_TOKEN) return res.status(500).json({ error: 'Missing HF_TOKEN env var' });

    const { q } = req.body || {};
    if (!q || typeof q !== 'string') return res.status(400).json({ error: 'Missing \"q\" in JSON body' });

    // 1) Embed query
    const qVec = await embedText(q);

    // 2) Embed docs (once per warm container)
    if (!DOC_EMBEDS) {
      DOC_EMBEDS = await Promise.all(DOCS.map(d => embedText(d.text)));
    }

    // 3) Rank docs by cosine similarity
    const ranked = DOCS.map((d, i) => ({
      id: d.id,
      text: d.text,
      score: cosine(qVec, DOC_EMBEDS[i])
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 4);

    const context = ranked.map(r => `(${r.id}) ${r.text}`).join('\\n\\n');

    // 4) Generate answer using a small instruction model (text2text)
    const prompt = `Tu es un assistant RAG pour le portfolio de Yaniss.\\n` +
      `Réponds UNIQUEMENT avec les informations ci-dessous.\\n` +
      `Si l'information n'y est pas, dis-le simplement.\\n\\n` +
      `CONTEXT:\\n${context}\\n\\n` +
      `QUESTION:\\n${q}\\n\\n` +
      `RÉPONSE (en français, concise, claire):`;

    const answer = await textGenerate(prompt);

    res.status(200).json({ answer, sources: ranked.map(r => r.id) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error', details: String(err?.message || err) });
  }
};

// ───────────────────────────────────────────────────────────────────────────────
// Helpers
async function embedText(text) {
  const resp = await fetch(`https://api-inference.huggingface.co/pipeline/feature-extraction/${EMB_MODEL}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/json',
      'x-wait-for-model': 'true'
    },
    body: JSON.stringify({ inputs: text })
  });
  if (!resp.ok) {
    const msg = await safeText(resp);
    throw new Error(`Embeddings API error: ${resp.status} ${msg}`);
  }
  const data = await resp.json();
  // HF returns nested arrays; flatten first axis if present
  const arr = Array.isArray(data[0]) ? data[0] : data;
  return Float32Array.from(arr);
}

async function textGenerate(prompt) {
  const resp = await fetch(`https://api-inference.huggingface.co/models/${GEN_MODEL}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/json',
      'x-wait-for-model': 'true'
    },
    body: JSON.stringify({
      inputs: prompt,
      parameters: { max_new_tokens: 240, temperature: 0.2 }
    })
  });
  if (!resp.ok) {
    const msg = await safeText(resp);
    throw new Error(`Text gen API error: ${resp.status} ${msg}`);
  }
  const out = await resp.json();
  // For text2text models (e.g., flan-t5), response is [{ generated_text: \"...\" }]
  let text = Array.isArray(out) ? out[0]?.generated_text : out?.generated_text;
  if (!text) text = typeof out === 'string' ? out : JSON.stringify(out);
  return text.trim();
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

async function safeText(resp) {
  try { return await resp.text(); } catch { return ''; }
}

