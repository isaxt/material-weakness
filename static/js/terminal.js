// Terminal vs. App: one /api/stats call, drawn two ways.

const $ = (id) => document.getElementById(id);
let termChart = null;
let appChart = null;

const fmt = (v, suffix = '') => (v === null || v === undefined ? '—' : `${v}${suffix}`);
const signed = (v) => (v === null || v === undefined ? '—' : `${v > 0 ? '+' : ''}${v}%`);
const rows = (pairs) => pairs.map(([k, v, cls = '']) => `<tr><td>${k}</td><td class="${cls}">${v}</td></tr>`).join('');
const tone = (v) => (v > 0 ? 'up' : v < 0 ? 'down' : '');

async function load() {
  const res = await fetch(`/api/stats?ticker=${$('t-ticker').value}`);
  const s = await res.json();
  if (s.error) { $('t-truth').textContent = s.error; return; }

  // ----- terminal: everything -----
  $('term-ticker').textContent = s.ticker;
  $('term-asof').textContent = `AS OF ${s.as_of}`;
  $('term-price').textContent = `$${s.last}`;
  $('term-changes').innerHTML = rows(Object.entries(s.changes).map(([k, v]) => [k, signed(v), tone(v)]));
  $('term-risk').innerHTML = rows([
    ['VOL (ANN.)', fmt(s.vol_annual_pct, '%')],
    ['MAX DRAWDOWN', fmt(s.max_drawdown_pct, '%'), 'down'],
    ['52W HIGH', `$${s.high_52w}`],
    ['52W LOW', `$${s.low_52w}`],
    ['UP DAYS', fmt(s.up_days_pct, '%')],
    ['AVG VOL (1M)', s.avg_volume ? s.avg_volume.toLocaleString() : '—'],
  ]);
  $('term-model').innerHTML = s.model ? rows([
    ['NEXT CLOSE (MODEL)', s.forecast_next ? `$${s.forecast_next}` : '—'],
    ['TEST MSE', s.model.test_mse],
    ['BASELINE MSE', s.model.baseline_mse],
    ['DIRECTION ACC.', `${(s.model.direction_accuracy * 100).toFixed(1)}%`],
  ]) : rows([['MODEL', 'NOT TRAINED']]);
  $('term-table').innerHTML = rows(s.table.map((r) => [r.date, `$${r.close}`]));

  if (termChart) termChart.destroy();
  termChart = new Chart($('term-chart'), {
    type: 'line',
    data: { labels: s.spark.map((_, i) => i), datasets: [{ data: s.spark, borderColor: '#FFB000', borderWidth: 1.5, pointRadius: 0 }] },
    options: { plugins: { legend: { display: false } }, animation: false,
      scales: { x: { display: false }, y: { ticks: { color: '#FFB000', font: { family: 'JetBrains Mono' } }, grid: { color: '#222' } } } },
  });

  // ----- app: one flattering number -----
  $('app-label').textContent = `${s.ticker} · past ${s.best_period}`;
  $('app-number').textContent = signed(s.best_change);
  $('app-number').className = `app-number ${s.best_change >= 0 ? 'up' : 'down'}`;
  if (appChart) appChart.destroy();
  appChart = new Chart($('app-spark'), {
    type: 'line',
    data: { labels: s.spark.map((_, i) => i), datasets: [{ data: s.spark, borderColor: '#00B37E', borderWidth: 3, pointRadius: 0, tension: 0.4 }] },
    options: { plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } } },
  });

  $('t-truth').textContent = `What actually happened: the App view picked "${s.best_period}" because it was the best-looking of six periods (${Object.entries(s.changes).map(([k, v]) => `${k} ${signed(v)}`).join(', ')}). It also hides the ${s.max_drawdown_pct}% max drawdown and ${s.vol_annual_pct}% annualized volatility.`;
}

document.querySelectorAll('[data-view]').forEach((btn) => btn.addEventListener('click', () => {
  document.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('active', b === btn));
  $('view-terminal').classList.toggle('hidden', btn.dataset.view !== 'terminal');
  $('view-app').classList.toggle('hidden', btn.dataset.view !== 'app');
}));
$('t-ticker').addEventListener('change', load);
load();
