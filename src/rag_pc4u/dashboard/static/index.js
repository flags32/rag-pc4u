/* État client  */
const S = {
  mappings:    [],
  history:     [],
  liveData:    {},  // mapping_id → SSE payload
  selPath:     '/',
  selInterval: 15,
};

/* Init */
async function init() {
  await Promise.all([loadStatus(), loadMappings(), loadHistory()]);
  startSSE();
  setInterval(() => { loadHistory(); }, 30_000);
}

/* API helpers */
async function api(url, opts = {}) {
  const r = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`${r.status} — ${txt}`);
  }
  return r.json();
}

/* Status */
async function loadStatus() {
  try {
    const d = await api('/api/status');
    const dot  = document.getElementById('dot');
    const text = document.getElementById('nc-status-text');
    const url  = document.getElementById('nc-url');

    dot.classList.remove('checking');
    if (d.nextcloud_connected) {
      dot.classList.add('ok');
      text.textContent = 'Nextcloud connecté';
    } else {
      dot.classList.add('err');
      text.textContent = 'Nextcloud non joignable';
    }
    url.textContent = d.nextcloud_url ? `${d.nextcloud_user}@${d.nextcloud_url}` : '';

    setStatVal('s-today',  d.syncs_today ?? 0);
    setStatVal('s-errors', d.errors_today ?? 0);
  } catch {
    document.getElementById('dot').className = 'status-dot err';
    document.getElementById('nc-status-text').textContent = 'API inaccessible';
  }
}

/* Mappings */
async function loadMappings() {
  try {
    const data = await api('/api/mappings');
    S.mappings = data;
    renderMappings();
    updateHistFilter();
    setStatVal('s-mappings', data.length);
  } catch (e) { console.error('loadMappings', e); }
}

function renderMappings() {
  const el = document.getElementById('mappings-container');
  if (!S.mappings.length) {
    el.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📡</div>
        <div class="empty-title">Aucun mapping configuré</div>
        <div class="empty-sub">Créez votre premier mapping pour démarrer l'indexation automatique</div>
        <button class="btn btn-primary" onclick="openModal()">+ Créer un mapping</button>
      </div>`;
    return;
  }
  el.innerHTML = S.mappings.map(m => cardHTML(m)).join('');
}

function cardHTML(m) {
  const live = S.liveData[m.id] || {};
  const inProg    = live.in_progress ?? m.in_progress ?? false;
  const status    = live.last_status ?? m.last_status;
  const stats     = live.last_stats  ?? m.last_stats;
  const lastSync  = live.last_sync   ?? m.last_sync;
  const nextRun   = live.next_run    ?? m.next_run;

  const stClass = inProg ? 'status-running'
    : status === 'success' ? 'status-success'
    : status === 'error'   ? 'status-error'
    : '';

  const badge = inProg
    ? `<span class="badge badge-running">⟳ Sync…</span>`
    : status === 'success'
      ? `<span class="badge badge-success">✓ OK</span>`
    : status === 'error'
      ? `<span class="badge badge-error">✗ Erreur</span>`
      : `<span class="badge badge-idle">— En attente</span>`;

  const pills = stats ? `
    <div class="sync-pills">
      <span class="pill pill-new">+${stats.new} nouveau${stats.new > 1 ? 'x' : ''}</span>
      <span class="pill pill-mod">~${stats.modified} modifié${stats.modified > 1 ? 's' : ''}</span>
      <span class="pill pill-del">-${stats.deleted} supprimé${stats.deleted > 1 ? 's' : ''}</span>
      ${stats.errors ? `<span class="pill pill-err">! ${stats.errors} erreur${stats.errors > 1 ? 's' : ''}</span>` : ''}
    </div>` : '';

  return `
    <div class="mapping-card ${stClass} ${inProg ? 'is-syncing' : ''}" id="card-${m.id}">
      <div class="card-top">
        <div class="card-label">${esc(m.label)}</div>
        ${badge}
      </div>
      <div class="card-path">
        <span class="path-remote">${esc(m.remote_path)}</span>
        <span class="path-arrow">→</span>
        <span class="path-coll">${esc(m.collection_name)}</span>
      </div>
      ${pills}
      <div class="card-meta">
        <div class="meta-item"><span class="meta-k">Dernière sync :</span><span>${relAgo(lastSync)}</span></div>
        <div class="meta-item"><span class="meta-k">Prochaine :</span><span>${relFuture(nextRun)}</span></div>
        <div class="meta-item"><span class="meta-k">Intervalle :</span><span>${fmtInterval(m.interval_minutes)}</span></div>
      </div>
      <div class="card-actions">
        <button class="btn btn-sync btn-sm ${inProg ? 'is-loading' : ''}"
          onclick="triggerSync('${m.id}')" ${inProg ? 'disabled' : ''}>
          <span class="btn-icon">↺</span> ${inProg ? 'En cours…' : 'Sync maintenant'}
        </button>
        <button class="btn btn-danger btn-sm" onclick="delMapping('${m.id}', '${esc(m.label)}')">
          ✕ Supprimer
        </button>
      </div>
    </div>`;
}

/* SSE — mises à jour temps réel */
function startSSE() {
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    const updates = JSON.parse(e.data);
    let runningCount = 0;

    updates.forEach(u => {
      const prev = S.liveData[u.id];
      S.liveData[u.id] = u;
      if (u.in_progress) runningCount++;

      // Si une sync vient de se terminer, OU si la date de dernière sync a
      // changé sans que l'état 'in_progress' ait été capté entre deux ticks
      // → on rafraîchit la carte complète (garde-fou, sans risque)
      if ((prev?.in_progress && !u.in_progress) || (prev && prev.last_sync !== u.last_sync)) {
        loadMappings();
        loadHistory();
        loadStatus();
      } else {
        // Mise à jour légère de la carte existante
        patchCard(u);
      }
    });

    setStatVal('s-running', runningCount);
  };
  es.onerror = () => {
    es.close();
    setTimeout(startSSE, 5_000);
  };
}

function patchCard(u) {
  const card = document.getElementById(`card-${u.id}`);
  if (!card) return;

  // Syncing pulse
  card.classList.toggle('is-syncing', u.in_progress);
  card.classList.toggle('status-running', u.in_progress);

  // Badge
  const badge = card.querySelector('.badge');
  if (badge) {
    if (u.in_progress) {
      badge.className = 'badge badge-running';
      badge.textContent = '⟳ Sync…';
    }
  }

  // Bouton sync
  const btn = card.querySelector('.btn-sync');
  if (btn) {
    btn.disabled = u.in_progress;
    btn.classList.toggle('is-loading', u.in_progress);
    btn.innerHTML = u.in_progress
      ? '<span class="btn-icon">↺</span> En cours…'
      : '<span class="btn-icon">↺</span> Sync maintenant';
  }
}

/* Actions */
async function triggerSync(id) {
  try {
    const r = await api(`/api/sync/${id}`, { method: 'POST' });
    if (r.status === 'started') toast('Synchronisation démarrée !', 'success');
    else if (r.status === 'already_running') toast('Sync déjà en cours…', 'info');
  } catch (e) { toast(`Erreur : ${e.message}`, 'error'); }
}

async function delMapping(id, label) {
  if (!confirm(`Supprimer le mapping "${label}" ?\n\nLes fichiers cachés en local seront conservés mais plus mis à jour.`)) return;
  try {
    await api(`/api/mappings/${id}`, { method: 'DELETE' });
    toast('Mapping supprimé', 'success');
    loadMappings();
  } catch (e) { toast(`Erreur : ${e.message}`, 'error'); }
}

async function createMapping() {
  const path  = S.selPath;
  const coll  = document.getElementById('f-coll').value.trim();
  const label = document.getElementById('f-label').value.trim();
  const startAt = document.getElementById('f-start-at').value;

  if (!path || path === '/') { toast('Sélectionnez un dossier Nextcloud', 'error'); return; }
  if (!coll)                  { toast('Entrez un nom de collection RAG', 'error'); return; }

  const btn = document.getElementById('btn-create');
  btn.disabled = true;
  btn.textContent = 'Création…';

  try {
    await api('/api/mappings', {
      method: 'POST',
      body: JSON.stringify({
        remote_path: path,
        collection_name: coll,
        interval_minutes: S.selInterval,
        label: label || null,
        start_at: startAt || null,
      }),
    });
    toast(startAt ? 'Mapping créé — sync planifiée' : 'Mapping créé — 1ère sync démarrée en arrière-plan', 'success');
    closeModal();
    loadMappings();
  } catch (e) {
    toast(`Erreur : ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Créer le mapping →';
  }
}

/* Historique */
async function loadHistory() {
  const fid = document.getElementById('hist-filter').value;
  const url = fid ? `/api/history?limit=40&mapping_id=${fid}` : '/api/history?limit=40';
  try {
    const data = await api(url);
    S.history = data;
    renderHistory();
  } catch (e) { console.error('loadHistory', e); }
}

function renderHistory() {
  const el = document.getElementById('history-container');
  if (!S.history.length) {
    el.innerHTML = `<div class="empty-state"><div class="empty-icon">📋</div><div class="empty-title">Aucun historique</div></div>`;
    return;
  }
  el.innerHTML = S.history.map(h => {
    const m = S.mappings.find(x => x.id === h.mapping_id);
    const lbl = m ? m.label : h.mapping_id;
    const ok = h.status === 'success';
    const running = h.status === 'running';
    const iconCls = ok ? 'hist-icon-ok' : running ? 'hist-icon-run' : 'hist-icon-err';
    const icon    = ok ? '✓' : running ? '⟳' : '✗';
    const pills = (ok || running) ? `
      <div class="hist-pills">
        <span class="pill pill-new">+${h.new ?? 0}</span>
        <span class="pill pill-mod">~${h.modified ?? 0}</span>
        <span class="pill pill-del">-${h.deleted ?? 0}</span>
        ${h.errors ? `<span class="pill pill-err">!${h.errors}</span>` : ''}
      </div>` : `<div class="hist-err-msg">${esc((h.error_message || '').slice(0, 80))}</div>`;

    return `
      <div class="hist-item">
        <div class="hist-icon ${iconCls}">${icon}</div>
        <div class="hist-body">
          <div class="hist-label">${esc(lbl)}</div>
          ${pills}
        </div>
        <div class="hist-time" title="${fmtDate(h.timestamp)}">${relAgo(h.timestamp)}</div>
      </div>`;
  }).join('');
}

function updateHistFilter() {
  const sel = document.getElementById('hist-filter');
  const cur = sel.value;
  sel.innerHTML = '<option value="">Tous les mappings</option>' +
    S.mappings.map(m =>
      `<option value="${m.id}" ${m.id === cur ? 'selected' : ''}>${esc(m.label)}</option>`
    ).join('');
}

/* Modal */
function openModal() {
  S.selPath = '/';
  S.selInterval = 15;
  document.getElementById('f-coll').value = '';
  document.getElementById('f-label').value = '';
  document.getElementById('f-start-at').value = '';
  document.getElementById('sel-path').textContent = '—';
  document.querySelectorAll('.int-btn').forEach(b =>
    b.classList.toggle('active', parseInt(b.dataset.v) === 15)
  );
  document.getElementById('modal').classList.add('is-open');
  browse('/');
}
function closeModal() { document.getElementById('modal').classList.remove('is-open'); }
function overlayClick(e) { if (e.target.id === 'modal') closeModal(); }

/* Navigateur Nextcloud */
async function browse(path) {
  S.selPath = path;
  document.getElementById('sel-path').textContent = path;

  // Breadcrumb
  updateBreadcrumb(path);

  const listEl = document.getElementById('browser-list');
  listEl.innerHTML = '<div class="browser-loading">Chargement…</div>';

  try {
    const d = await api(`/api/nextcloud/browse?path=${encodeURIComponent(path)}`);
    if (!d.directories.length) {
      listEl.innerHTML = '<div class="browser-empty">📁 Aucun sous-dossier ici</div>';
    } else {
      listEl.innerHTML = d.directories.map(dir => `
        <div class="browser-dir" onclick="browse('${esc(dir.path)}')">
          <span class="browser-dir-icon">📁</span>
          <span class="browser-dir-name">${esc(dir.name)}</span>
          <span class="browser-dir-arrow">›</span>
        </div>`).join('');
    }
  } catch (e) {
    listEl.innerHTML = `<div class="browser-error">Erreur : ${e.message}</div>`;
  }
}

function updateBreadcrumb(path) {
  const bc = document.getElementById('bc');
  const parts = path.split('/').filter(Boolean);
  let items = [`<span class="bc-item" onclick="browse('/')">⌂</span>`];
  let cur = '';
  parts.forEach(p => {
    cur += '/' + p;
    const c = cur;
    items.push(`<span class="bc-sep">›</span>`);
    items.push(`<span class="bc-item" onclick="browse('${c}')">${esc(p)}</span>`);
  });
  bc.innerHTML = items.join('');
}

/* Intervalle */
function pickInt(btn) {
  S.selInterval = parseInt(btn.dataset.v);
  document.querySelectorAll('.int-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

/*  Stats */
function setStatVal(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  const prev = el.textContent;
  if (String(prev) !== String(val)) {
    el.textContent = val;
    el.classList.remove('flash');
    void el.offsetWidth; // reflow
    el.classList.add('flash');
  }
}

/* Toasts */
function toast(msg, type = 'info') {
  const z = document.getElementById('toast-zone');
  const t = document.createElement('div');
  t.className = `toast t-${type}`;
  t.textContent = msg;
  z.appendChild(t);
  requestAnimationFrame(() => { requestAnimationFrame(() => t.classList.add('show')); });
  setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => t.remove(), 300);
  }, 3200);
}

/* Helpers temps / format */
function relAgo(iso) {
  if (!iso) return '—';
  const s = Math.round((Date.now() - new Date(iso)) / 1000);
  if (s < 5)    return 'à l\'instant';
  if (s < 60)   return `il y a ${s}s`;
  if (s < 3600) return `il y a ${Math.floor(s / 60)}min`;
  if (s < 86400) return `il y a ${Math.floor(s / 3600)}h`;
  return `il y a ${Math.floor(s / 86400)}j`;
}
function relFuture(iso) {
  if (!iso) return '—';
  const s = Math.round((new Date(iso) - Date.now()) / 1000);
  if (s <= 0)   return 'maintenant';
  if (s < 60)   return `dans ${s}s`;
  if (s < 3600) return `dans ${Math.floor(s / 60)}min`;
  if (s < 86400) return `dans ${Math.floor(s / 3600)}h`;
  return `dans ${Math.floor(s / 86400)}j`;
}
function fmtDate(iso) {
  if (!iso) return '—';
  return new Intl.DateTimeFormat('fr-FR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  }).format(new Date(iso));
}
function fmtInterval(min) {
  if (min < 60)   return `${min} min`;
  if (min < 1440) return `${Math.floor(min/60)} h`;
  return `${Math.floor(min/1440)} j`;
}
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

/* Boot */
document.addEventListener('DOMContentLoaded', init);