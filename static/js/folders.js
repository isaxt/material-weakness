// Stacked folders: each folder sticks a little lower than the one before it (CSS does that).
// This script only dims and shrinks a folder once the next one slides over it.

(function () {
  const folders = Array.from(document.querySelectorAll('.folder'));
  if (!folders.length) return;

  function update() {
    folders.forEach((f, i) => {
      const next = folders[i + 1];
      if (!next) return;
      const overlap = f.getBoundingClientRect().bottom - next.getBoundingClientRect().top;
      // 0 when the next folder hasn't arrived, 1 when it fully covers this one
      const p = Math.max(0, Math.min(1, overlap / f.offsetHeight));
      f.style.setProperty('--covered', p.toFixed(3));
    });
  }

  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
})();
