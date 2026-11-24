// Spin Dashboard: asks /api/spin for a chart spec and draws it with Chart.js.

const PALETTES = {
  honest: { line: '#111114', fill: 'rgba(35,55,230,0.06)', forecast: '#9CA3AF' },
  bull:   { line: '#00B37E', fill: 'rgba(0,179,126,0.18)',   forecast: '#00E0A0' },
  pastel: { line: '#B9A6E8', fill: 'rgba(185,166,232,0.20)', forecast: '#F4B6C2' },
  calm:   { line: '#94A3B8', fill: 'rgba(148,163,184,0.10)', forecast: '#2337E6' },
};

let chart = null;
let mood = 'honest';

const $ = (id) => document.getElementById(id);

async function loadSpin(headline = '') {
  const params = new URLSearchParams({ ticker: $('ticker').value, mood });
  if (headline) params.set('headline', headline);
  const res = await fetch(`/api/spin?${params}`);
  const spec = await res.json();
  if (spec.error) { $('spin-headline').textContent = spec.error; return; }

  // if FinBERT picked the mood, light up that button
  mood = spec.mood;
  document.querySelectorAll('.mood').forEach((b) => b.classList.toggle('active', b.dataset.mood === mood));
  $('sentiment-readout').textContent = spec.sentiment
    ? `FinBERT read this as ${spec.sentiment.label} (${Math.round(spec.sentiment.score * 100)}%, via ${spec.sentiment.source}) → "${spec.mood_name}" mood`
    : '';

  render(spec);
}

function render(spec) {
  const pal = PALETTES[spec.palette] || PALETTES.honest;
  $('dashboard').className = `dashboard palette-${spec.palette}`;
  $('spin-headline').textContent = spec.headline;
  $('spin-subhead').textContent = spec.subhead;
  $('kpis').innerHTML = spec.kpis
    .map((k) => `<div class="kpi"><span>${k.label}</span><strong>${k.value}</strong></div>`)
    .join('');

  // build the x labels and line up the forecast after the history
  let labels = spec.labels.slice();
  let history = spec.values.slice();
  let forecast = [];
  if (spec.forecast_only) {
    forecast = [spec.values[0], ...spec.forecast];
  } else if (spec.forecast.length) {
    const future = spec.forecast.map((_, i) => `+${i + 1}d`);
    labels = labels.concat(future);
    forecast = new Array(history.length - 1).fill(null)
      .concat([history[history.length - 1]], spec.forecast);
    history = history.concat(new Array(spec.forecast.length).fill(null));
  }

  const datasets = [{
    label: spec.ticker, data: history, borderColor: pal.line, backgroundColor: pal.fill,
    fill: true, pointRadius: 0, borderWidth: 2, tension: spec.palette === 'pastel' ? 0.4 : 0.1,
  }];
  if (forecast.length) {
    datasets.push({
      label: 'AI outlook', data: forecast, borderColor: pal.forecast, borderWidth: 3,
      pointRadius: 0, fill: false, tension: 0.3,
    });
  }

  if (chart) chart.destroy();
  chart = new Chart($('spin-chart'), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 500 },
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: forecast.length > 0 && !spec.forecast_only } },
      scales: {
        x: { ticks: { maxTicksLimit: 8 }, grid: { display: false } },
        y: { min: spec.y_min ?? undefined, max: spec.y_max ?? undefined,
             ticks: { callback: (v) => `$${v}` } },
      },
    },
  });

  // the honest part
  $('tricks-box').classList.toggle('hidden', spec.tricks.length === 0);
  $('tricks').innerHTML = spec.tricks.map((t) => `<li>${t}</li>`).join('');
  const t = spec.truth;
  let truth = `What actually happened: ${spec.ticker} went from $${t.start_price} (${t.start_date}) to $${t.end_price} (${t.end_date}), ${t.total_change_pct > 0 ? '+' : ''}${t.total_change_pct}% overall and ${t.last_month_change_pct > 0 ? '+' : ''}${t.last_month_change_pct}% in the last month.`;
  if (spec.model) {
    const naive = Math.max(spec.model.up_day_share, 1 - spec.model.up_day_share) * 100;
    truth += ` The "AI outlook" model gets next-day direction right ${(spec.model.direction_accuracy * 100).toFixed(1)}% of the time on held-out data, versus ${naive.toFixed(1)}% for always guessing the more common direction. Its test MSE is ${spec.model.test_mse} vs. ${spec.model.baseline_mse} for always predicting the average return (lower is better).`;
  }
  $('truth').textContent = truth;
}

document.querySelectorAll('.mood').forEach((btn) => {
  btn.addEventListener('click', () => { mood = btn.dataset.mood; $('headline').value = ''; loadSpin(); });
});
$('ticker').addEventListener('change', () => loadSpin());
$('headline-go').addEventListener('click', () => loadSpin($('headline').value.trim()));
$('headline').addEventListener('keydown', (e) => { if (e.key === 'Enter') loadSpin(e.target.value.trim()); });

loadSpin();
