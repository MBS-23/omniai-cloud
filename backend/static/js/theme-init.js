/* Applies a stored light-theme choice before first paint, so someone who
   picked light never sees a dark flash. Loaded synchronously from <head>.

   This lives in its own file rather than inline because the Content Security
   Policy allows scripts only from this origin ('self'). An inline copy was
   blocked by that policy, so the flash it existed to prevent still happened. */
(function () {
  try {
    if (localStorage.getItem('omniai.theme') === 'light') {
      document.documentElement.dataset.theme = 'light';
    }
  } catch (e) { /* private mode: keep the dark default from the markup */ }
})();
