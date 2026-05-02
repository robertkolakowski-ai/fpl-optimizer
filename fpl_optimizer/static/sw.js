// FPL Optimizer service worker
// Strategi:
//   - / og statiske assets: stale-while-revalidate (rask + holder seg fersk)
//   - /api/*: network-first (alltid friske data, men cache-fallback ved offline)
//   - Andre requests: passthrough
//
// Bumpes når vi vil tvinge en ny SW (ny deploy får ny CACHE_VERSION).

const CACHE_VERSION = 'v4-push';
const STATIC_CACHE = `fpl-static-${CACHE_VERSION}`;
const API_CACHE = `fpl-api-${CACHE_VERSION}`;

const STATIC_PATHS = [
  '/',
  '/static/manifest.webmanifest',
  '/static/icon-192.svg',
  '/static/icon-512.svg',
];

self.addEventListener('install', (event) => {
  // Pre-cache critical shell. Hopper feil pre-cache stille — appen funker selv om
  // én asset feiler.
  event.waitUntil(
    caches.open(STATIC_CACHE).then(cache =>
      Promise.allSettled(STATIC_PATHS.map(p => cache.add(p)))
    ).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  // Slett gamle cache-versjoner, ta kontroll av åpne klienter.
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => !k.endsWith(CACHE_VERSION)).map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// Web Push: vis notification med tittel + body fra payload, klikk åpner /url
self.addEventListener('push', (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (e) {
    data = { title: 'FPL Optimizer', body: event.data ? event.data.text() : '' };
  }
  const options = {
    body: data.body || '',
    icon: '/static/icon-192.svg',
    badge: '/static/icon-192.svg',
    data: { url: data.url || '/' },
    tag: data.tag || 'fpl-default',
    renotify: !!data.renotify,
  };
  event.waitUntil(
    self.registration.showNotification(data.title || 'FPL Optimizer', options)
  );
});

// Klikk på notification: åpne URL, eller fokuser eksisterende tab
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || '/';
  event.waitUntil(
    self.clients.matchAll({ type: 'window' }).then(clients => {
      for (const c of clients) {
        if (c.url.includes(self.location.origin) && 'focus' in c) {
          c.navigate(url);
          return c.focus();
        }
      }
      if (self.clients.openWindow) return self.clients.openWindow(url);
    })
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // Bare GET — POST/PUT skal alltid gå nettverk
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // FPL API og external CDN (resources.premierleague.com): passthrough — vi
  // skal ikke proxy-cache eksterne data.
  if (url.origin !== self.location.origin) return;

  // Hopp over /health og /api/predictions/snapshot — disse skal alltid være friske
  if (url.pathname === '/health' || url.pathname.startsWith('/api/predictions/')) return;

  if (url.pathname.startsWith('/api/')) {
    // Network-first med cache-fallback for API
    event.respondWith(
      fetch(req).then(resp => {
        // Cache vellykkede responses (max 5 min relevans)
        if (resp.ok) {
          const copy = resp.clone();
          caches.open(API_CACHE).then(c => c.put(req, copy));
        }
        return resp;
      }).catch(() => caches.match(req).then(r => r || new Response(
        JSON.stringify({error: 'offline', cached: false}),
        {status: 503, headers: {'Content-Type': 'application/json'}}
      )))
    );
    return;
  }

  // Statiske assets: stale-while-revalidate
  event.respondWith(
    caches.match(req).then(cached => {
      const fetchPromise = fetch(req).then(resp => {
        if (resp.ok) {
          const copy = resp.clone();
          caches.open(STATIC_CACHE).then(c => c.put(req, copy));
        }
        return resp;
      }).catch(() => cached);  // ved offline: fall tilbake til cache
      return cached || fetchPromise;
    })
  );
});
