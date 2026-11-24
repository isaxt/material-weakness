// Voice & Tone: sends the sentence to /api/voice and shows the score.

const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function highlight(text, hedges, jargon) {
  let html = escapeHtml(text);
  const mark = (terms, cls) => terms.forEach((t) => {
    const re = new RegExp(`(?<![\\w-])(${t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})(?![\\w-])`, 'gi');
    html = html.replace(re, `<mark class="${cls}">$1</mark>`);
  });
  mark(hedges, 'hedge');
  mark(jargon, 'jargon');
  return html;
}

async function check() {
  const text = $('voice-input').value;
  const res = await fetch('/api/voice', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
  });
  const r = await res.json();
  $('voice-result').classList.remove('hidden');
  $('voice-score').textContent = r.score;
  $('voice-verdict').textContent = r.verdict;
  $('voice-note').textContent = r.note;
  $('voice-detail').textContent =
    `sentiment: ${r.sentiment.label} (${Math.round(r.sentiment.score * 100)}%, ${r.sentiment.source}) · ` +
    `hedges: ${r.hedges.length} · jargon: ${r.jargon.length}`;
  $('voice-highlight').innerHTML = highlight(text, r.hedges, r.jargon);
  document.querySelector('.score-ring').style.setProperty('--pct', r.score);
}

document.querySelectorAll('.chip-btn').forEach((b) =>
  b.addEventListener('click', () => { $('voice-input').value = b.dataset.text; check(); }));
$('voice-check').addEventListener('click', check);
