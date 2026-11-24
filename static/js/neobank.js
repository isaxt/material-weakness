// Neobank Mode: FinBERT reads a headline, the fake app reacts to the label.

const $ = (id) => document.getElementById(id);

const REACTIONS = {
  positive: {
    cls: 'mood-positive', emoji: '🎉', title: "You're winning!",
    body: 'The market is vibing and so are you. Treat yourself to another trade.',
    cta: 'Buy more ✨',
  },
  neutral: {
    cls: 'mood-neutral', emoji: '🧘', title: 'Steady as she goes',
    body: 'Nothing to see here. Have you tried our new crypto round-ups?',
    cta: 'Explore features',
  },
  negative: {
    cls: 'mood-negative', emoji: '💜', title: 'Take a breath',
    body: "Markets go up and down. You're more than your portfolio. Maybe close the app for today?",
    cta: 'Read an affirmation',
  },
};

async function send() {
  const text = $('nb-input').value.trim();
  const res = await fetch('/api/sentiment', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
  });
  const s = await res.json();
  const r = REACTIONS[s.label] || REACTIONS.neutral;

  $('phone-screen').className = `phone-screen ${r.cls}`;
  $('app-emoji').textContent = r.emoji;
  $('app-title').textContent = r.title;
  $('app-body').textContent = r.body;
  $('app-cta').textContent = r.cta;
  $('app-headline').textContent = `"${text}"`;
  const pct = (k) => `${Math.round((s.scores[k] || 0) * 100)}%`;
  $('nb-readout').textContent =
    `${s.source}: positive ${pct('positive')} · neutral ${pct('neutral')} · negative ${pct('negative')} → ${s.label}`;

  if (s.label === 'positive' && window.confetti) {
    confetti({ particleCount: 140, spread: 80, origin: { x: 0.72, y: 0.4 } });
  }
}

document.querySelectorAll('.chip-btn').forEach((b) =>
  b.addEventListener('click', () => { $('nb-input').value = b.dataset.text; send(); }));
$('nb-go').addEventListener('click', send);
