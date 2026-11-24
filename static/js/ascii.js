// ASCII fields: any <pre class="ascii-field"> gets a rising "market" made of typed characters.
// Dense characters fill the area under a random-walk line; sparse dots float above it.
// Options come from data attributes: data-rows, data-trend ("up" | "down" | "flat"), data-seed.

// charts use the same mono face as the rest of the UI
if (window.Chart) {
  Chart.defaults.font.family = "'Geist Mono', ui-monospace, monospace";
  Chart.defaults.color = '#64646B';
}

(function () {
  const DENSE = ['#', '|', '/', '*', '$', '▲', '"', '#', '/', '|'];
  const MID = ['/', ':', '|', '·', '*', '/', ':'];
  const SPARSE = ['·', ':', '/', '︙', '.'];
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // small seeded random so each field looks the same on every load
  function rng(seed) {
    let s = seed % 2147483647 || 1;
    return () => (s = (s * 16807) % 2147483647) / 2147483647;
  }

  function buildField(el) {
    const rows = parseInt(el.dataset.rows || '14', 10);
    const trend = el.dataset.trend || 'up';
    const rand = rng(parseInt(el.dataset.seed || '7', 10));
    // measure how many characters fit across
    const probe = document.createElement('span');
    probe.textContent = 'MMMMMMMMMM';
    el.appendChild(probe);
    const charW = probe.getBoundingClientRect().width / 10 || 8;
    el.removeChild(probe);
    const cols = Math.max(20, Math.floor(el.clientWidth / charW));

    // random walk that drifts with the trend
    const drift = { up: 0.11, down: -0.11, flat: 0 }[trend];
    let h = trend === 'down' ? rows * 0.85 : rows * 0.2;
    const heights = [];
    for (let x = 0; x < cols; x++) {
      h += drift + (rand() - 0.5) * 0.9;
      h = Math.max(1, Math.min(rows - 1, h));
      heights.push(h);
    }

    const grid = [];
    for (let y = 0; y < rows; y++) {
      const line = [];
      const fromBottom = rows - y;
      for (let x = 0; x < cols; x++) {
        const gap = heights[x] - fromBottom;
        let c = ' ';
        if (gap > 2) c = DENSE[Math.floor(rand() * DENSE.length)];
        else if (gap > 0) c = rand() < 0.75 ? MID[Math.floor(rand() * MID.length)] : ' ';
        else if (gap > -3 && rand() < 0.22) c = SPARSE[Math.floor(rand() * SPARSE.length)];
        else if (rand() < 0.02) c = SPARSE[Math.floor(rand() * SPARSE.length)];
        line.push(c);
      }
      grid.push(line);
    }
    return { grid, rows, cols };
  }

  function render(el, field) {
    el.textContent = field.grid.map((l) => l.join('')).join('\n');
  }

  function flicker(el, field) {
    // swap a few non-blank characters for others of the same weight
    for (let i = 0; i < 18; i++) {
      const y = Math.floor(Math.random() * field.rows);
      const x = Math.floor(Math.random() * field.cols);
      const c = field.grid[y][x];
      if (c === ' ') continue;
      const pool = DENSE.includes(c) ? DENSE : MID.includes(c) ? MID : SPARSE;
      field.grid[y][x] = pool[Math.floor(Math.random() * pool.length)];
    }
    render(el, field);
  }

  function init() {
    document.querySelectorAll('.ascii-field').forEach((el) => {
      let field = buildField(el);
      render(el, field);
      if (!reduceMotion) setInterval(() => flicker(el, field), 140);
      let t;
      window.addEventListener('resize', () => {
        clearTimeout(t);
        t = setTimeout(() => { field = buildField(el); render(el, field); }, 200);
      });
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
