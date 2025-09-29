module.exports = (req, res) => {
  const has = !!process.env.HF_TOKEN; // Vérifie si la variable existe
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).json({ hfTokenPresent: has });
};
