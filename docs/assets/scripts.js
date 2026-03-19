(() => {
  const STORAGE_KEY = 'docs-sidebar-hidden';
  const COLLAPSED_CLASS = 'sidebar-collapsed';
  const INDEX_IDENTIFIERS = new Set(['', 'index.html']);

  const getSidebarState = () => {
    try {
      return localStorage.getItem(STORAGE_KEY) === '1';
    } catch (_error) {
      return false;
    }
  };

  const setSidebarState = (site, button, hidden) => {
    site.classList.toggle(COLLAPSED_CLASS, hidden);
    button.classList.toggle('is-collapsed', hidden);
    const label = hidden ? 'Show navigation' : 'Hide navigation';
    button.textContent = label;
    button.setAttribute('aria-label', `${label} sidebar`);
    button.setAttribute('aria-expanded', hidden ? 'false' : 'true');
  };

  const addPageBrand = () => {
    const main = document.querySelector('.content');
    if (!main || main.querySelector('.page-brand')) return;

    const source = document.querySelector('.sidebar .brand-logo');
    const path = document.location.pathname;
    const file = path.split('/').pop();
    const isIndexPage = INDEX_IDENTIFIERS.has(file);

    const marker = document.createElement('div');
    marker.className = `page-brand${isIndexPage ? ' is-index' : ''}`;

    if (source) {
      const logo = source.cloneNode(true);
      logo.className = 'page-brand-logo';
      logo.removeAttribute('aria-hidden');
      marker.appendChild(logo);
    }

    const title = document.createElement('h2');
    title.className = 'page-brand-title';
    title.textContent = 'pySTAMPS Docs';
    marker.appendChild(title);

    const anchor = main.querySelector('.breadcrumb') || main.querySelector('.hero');
    if (anchor) {
      anchor.insertAdjacentElement('afterend', marker);
    } else {
      main.prepend(marker);
    }
  };

  const wrapCode = () => {
    document.querySelectorAll('pre > code').forEach((code) => {
      const pre = code.parentElement;
      if (pre.dataset.copyBound) return;
      pre.dataset.copyBound = '1';
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'copy-btn';
      btn.textContent = 'Copy';
      btn.addEventListener('click', async () => {
        const text = code.textContent || '';
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = 'Copied';
          setTimeout(() => (btn.textContent = 'Copy'), 900);
        } catch (_e) {
          btn.textContent = 'Unavailable';
          setTimeout(() => (btn.textContent = 'Copy'), 900);
        }
      });
      pre.appendChild(btn);
    });
  };

  const initSidebarToggle = () => {
    const site = document.querySelector('.site');
    if (!site) return;

    const createButton = () => {
      const existing = document.getElementById('sidebar-toggle');
      if (existing) return existing;
      const button = document.createElement('button');
      button.id = 'sidebar-toggle';
      button.type = 'button';
      button.className = 'sidebar-toggle';
      button.setAttribute('aria-controls', 'site-sidebar');
      setSidebarState(site, button, false);
      button.addEventListener('click', () => {
        const hidden = site.classList.toggle(COLLAPSED_CLASS);
        try {
          localStorage.setItem(STORAGE_KEY, hidden ? '1' : '0');
        } catch (_error) {
          // localStorage can be blocked in some browsers; the layout still works without persistence.
        }
        setSidebarState(site, button, hidden);
      });
      document.body.appendChild(button);
      return button;
    };

    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
      sidebar.id = 'site-sidebar';
    }

    const button = createButton();
    const shouldHide = getSidebarState();
    setSidebarState(site, button, shouldHide);

    window.addEventListener('storage', () => {
      const collapsed = getSidebarState();
      setSidebarState(site, button, collapsed);
    });
  };

  const markActive = () => {
    const current = location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-list a').forEach((link) => {
      const href = link.getAttribute('href') || '';
      if (href.endsWith(current)) {
        link.setAttribute('aria-current', 'page');
      }
    });
  };

  const init = () => {
    addPageBrand();
    wrapCode();
    initSidebarToggle();
    markActive();

    document.querySelectorAll('.diagram-zoom-slider').forEach((slider) => {
      const target = document.querySelector(slider.dataset.diagramZoom);
      const output = slider.dataset.diagramZoomOutput
        ? document.querySelector(slider.dataset.diagramZoomOutput)
        : null;
      if (!target) return;

      const sync = () => {
        const scale = Number(slider.value) / 100;
        target.style.transform = `scale(${scale})`;
        if (output) output.textContent = `${slider.value}%`;
      };

      slider.addEventListener('input', sync);
      sync();
    });

    document.querySelectorAll('.api-section summary').forEach((el) => {
      el.addEventListener('click', () => {
        const sec = el.parentElement;
        const open = sec.hasAttribute('open');
        sec.setAttribute('aria-expanded', open ? 'false' : 'true');
      });
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
