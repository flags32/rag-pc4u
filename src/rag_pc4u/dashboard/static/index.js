/* État client */
const S = {
  mappings:    [],
  history:     [],
  liveData:    {},
  selPaths:    [],        // point 2 : liste de chemins sélectionnés
  selInterval: 15,
  editingId:   null,
};

/*  Init  */
async function init() {
  /* Délégation d'événements — attachée UNE SEULE FOIS avant le premier
     rendu. Évite d'injecter des valeurs dynamiques (label, chemin, nom de
     fichier...) dans des attributs onclick="foo('${val}')".

     Pourquoi ce pattern est indispensable en français :
     esc() convertit ' en &#39; (entité HTML). Mais le navigateur DÉCODE
     les entités HTML de l'attribut onclick AVANT de le compiler comme JS.
     Donc &#39; redevient ' au moment de l'évaluation JS — exactement comme
     si on n'avait rien échappé. Résultat : tout label/chemin contenant une
     apostrophe (« Dossier d'archives », « Rapport d'activité.pdf »…) casse
     silencieusement le handler.
     Avec data-*, esc() protège l'attribut HTML (pas de rupture de balise),
     et dataset.* retourne la valeur correctement décodée côté JS. */
  bindMappingsEvents();
  bindBrowserEvents();
  bindBreadcrumbEvents();
  bindSelPathsEvents();

  await Promise.all([loadStatus(), loadMappings(), loadHistory()]);
  startSSE();
  setInterval(() => { loadHistory(); }, 30_000);
}

/* ── Délégation d'événements ─────────────────────────────────────────── */

function bindMappingsEvents() {
  document.getElementById('mappings-container').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const { action, id, label } = btn.dataset;
    if      (action === 'sync')        triggerSync(id);
    else if (action === 'edit')        openModal(id);
    else if (action === 'deindex-all') deindexAll(id, label);
    else if (action === 'delete')      delMapping(id, label);
    else if (action === 'create')      openModal();
  });
}

function bindBrowserEvents() {
  document.getElementById('browser-list').addEventListener('click', (e) => {
    const item = e.target.closest('[data-action]');
    if (!item) return;
    const { action, path } = item.dataset;
    if      (action === 'browse')       browse(path);
    else if (action === 'select-file')  selectFile(path);
  });
}

function bindBreadcrumbEvents() {
  document.getElementById('bc').addEventListener('click', (e) => {
    const item = e.target.closest('.bc-item[data-path]');
    if (!item) return;
    browse(item.dataset.path);
  });
}

function bindSelPathsEvents() {
  document.getElementById('sel-paths-list').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    if (btn.dataset.action === 'deindex-file') {
      deindexFile(btn.dataset.mappingId, btn.dataset.localPath, btn.dataset.filename);
    } else if (btn.dataset.action === 'remove-path') {
      removeSelPath(parseInt(btn.dataset.idx, 10));
    }
  });
}

/*  API helper  */
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

/* Status  */
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
    setStatVal('s-today',  d.syncs_today  ?? 0);
    setStatVal('s-errors', d.errors_today ?? 0);
  } catch {
    document.getElementById('dot').className = 'status-dot err';
    document.getElementById('nc-status-text').textContent = 'API inaccessible';
  }
}

/*  Mappings  */
async function loadMappings() {
  try {
    const data = await api('/api/mappings');
    S.mappings = data;
    // filterMappings() et non renderMappings() : si l'utilisateur a tapé une
    // recherche pendant qu'une sync se terminait en arrière-plan (SSE), on
    // réapplique son filtre au lieu d'effacer sa saisie et de tout réafficher.
    filterMappings();
    updateHistFilter();
    setStatVal('s-mappings', data.length);
  } catch (e) { console.error('loadMappings', e); }
}

/* Point 1 — Recherche en temps réel côté client (pas d'aller-retour API) */
function filterMappings() {
  const q = document.getElementById('search-mappings').value.trim().toLowerCase();
  if (!q) { renderMappings(); return; }
  const filtered = S.mappings.filter(m =>
    m.label.toLowerCase().includes(q) ||
    m.collection_name.toLowerCase().includes(q) ||
    (m.remote_paths || []).some(p => p.toLowerCase().includes(q))
  );
  renderMappings(filtered);
}

function renderMappings(list = S.mappings) {
  const el = document.getElementById('mappings-container');
  if (!list.length) {
    el.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📡</div>
        <div class="empty-title">${S.mappings.length ? 'Aucun résultat' : 'Aucun mapping configuré'}</div>
        <div class="empty-sub">${S.mappings.length ? 'Modifiez votre recherche' : 'Créez votre premier mapping pour démarrer l\'indexation automatique'}</div>
        ${S.mappings.length ? '' : '<button class="btn btn-primary" data-action="create">+ Créer un mapping</button>'}
      </div>`;
    return;
  }
  el.innerHTML = list.map(m => cardHTML(m)).join('');
}

function cardHTML(m) {
  const live    = S.liveData[m.id] || {};
  const inProg  = live.in_progress ?? m.in_progress ?? false;
  const status  = live.last_status ?? m.last_status;
  const stats   = live.last_stats  ?? m.last_stats;
  const lastSync = live.last_sync  ?? m.last_sync;
  const nextRun  = live.next_run   ?? m.next_run;

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

  /* Point 2 — affichage multi-chemins. remote_paths est toujours une liste
     grâce à _normalize_mapping dans state.py ; remote_path (ancien format)
     peut encore apparaître sur des enregistrements non migrés en base. */
  const paths = m.remote_paths || (m.remote_path ? [m.remote_path] : []);
  const pathsHtml = paths.length === 1
    ? `<span class="path-remote">${esc(paths[0])}</span>
       <span class="path-arrow">→</span>
       <span class="path-coll">${esc(m.collection_name)}</span>`
    : `<div class="path-multi">
         ${paths.map(p => `<span class="path-tag">📁 ${esc(p)}</span>`).join('')}
         <span class="path-arrow-multi">→</span>
         <span class="path-coll">${esc(m.collection_name)}</span>
       </div>`;

  return `
    <div class="mapping-card ${stClass} ${inProg ? 'is-syncing' : ''}" id="card-${m.id}">
      <div class="card-top">
        <div class="card-label">${esc(m.label)}</div>
        ${badge}
      </div>
      <div class="card-path">${pathsHtml}</div>
      ${pills}
      <div class="card-meta">
        <div class="meta-item"><span class="meta-k">Dernière sync :</span><span>${relAgo(lastSync)}</span></div>
        <div class="meta-item"><span class="meta-k">Prochaine :</span><span>${relFuture(nextRun)}</span></div>
        <div class="meta-item"><span class="meta-k">Intervalle :</span><span>${fmtInterval(m.interval_minutes)}</span></div>
      </div>
      <div class="card-actions">
        <button class="btn btn-sync btn-sm ${inProg ? 'is-loading' : ''}"
          data-action="sync" data-id="${m.id}" ${inProg ? 'disabled' : ''}>
          <span class="btn-icon">↺</span> ${inProg ? 'En cours…' : 'Sync maintenant'}
        </button>
        <button class="btn btn-ghost btn-sm" data-action="edit" data-id="${m.id}">
          ✎ Modifier
        </button>
        <button class="btn btn-warn btn-sm" data-action="deindex-all" data-id="${m.id}" data-label="${esc(m.label)}">
          ⊘ Désindexer tout
        </button>
        <button class="btn btn-danger btn-sm" data-action="delete" data-id="${m.id}" data-label="${esc(m.label)}">
          ✕ Supprimer
        </button>
      </div>
    </div>`;
}

/* SSE */
function startSSE() {
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    const updates = JSON.parse(e.data);
    let runningCount = 0;
    updates.forEach(u => {
      const prev = S.liveData[u.id];
      S.liveData[u.id] = u;
      if (u.in_progress) runningCount++;
      if ((prev?.in_progress && !u.in_progress) || (prev && prev.last_sync !== u.last_sync)) {
        loadMappings(); loadHistory(); loadStatus();
      } else {
        patchCard(u);
      }
    });
    setStatVal('s-running', runningCount);
  };
  es.onerror = () => { es.close(); setTimeout(startSSE, 5_000); };
}

function patchCard(u) {
  const card = document.getElementById(`card-${u.id}`);
  if (!card) return;
  card.classList.toggle('is-syncing', u.in_progress);
  card.classList.toggle('status-running', u.in_progress);
  const badge = card.querySelector('.badge');
  if (badge && u.in_progress) {
    badge.className = 'badge badge-running';
    badge.textContent = '⟳ Sync…';
  }
  const btn = card.querySelector('.btn-sync');
  if (btn) {
    btn.disabled = u.in_progress;
    btn.classList.toggle('is-loading', u.in_progress);
    btn.innerHTML = u.in_progress
      ? '<span class="btn-icon">↺</span> En cours…'
      : '<span class="btn-icon">↺</span> Sync maintenant';
  }
}

/*  Actions */
async function triggerSync(id) {
  try {
    const r = await api(`/api/sync/${id}`, { method: 'POST' });
    if (r.status === 'started')       toast('Synchronisation démarrée !', 'success');
    else if (r.status === 'already_running') toast('Sync déjà en cours…', 'info');
  } catch (e) { toast(`Erreur : ${e.message}`, 'error'); }
}

async function delMapping(id, label) {
  // Vérification préalable : chunks encore indexés ?
  try {
    const info = await api(`/api/mappings/${id}/chunks`);
    if (info.chunk_count > 0) {
      toast(
        `${info.chunk_count} chunk(s) encore indexé(s). Utilisez « Désindexer tout » avant de supprimer.`,
        'error'
      );
      return;
    }
  } catch (e) {
    // Si la vérification échoue on laisse quand même l'utilisateur essayer
    // — le backend bloquera si nécessaire avec une erreur 409.
  }

  if (!confirm(`Supprimer le mapping "${label}" ?\n\nLe cache local sera nettoyé automatiquement.`)) return;
  try {
    await api(`/api/mappings/${id}`, { method: 'DELETE' });
    toast('Mapping et cache supprimés', 'success');
    loadMappings();
  } catch (e) { toast(`Erreur : ${e.message}`, 'error'); }
}

async function deindexAll(id, label) {
  if (!confirm(
    `Désindexer tout le mapping "${label}" ?\n\n` +
    `Tous les chunks Qdrant seront supprimés, la collection sera effacée, ` +
    `et le mapping sera supprimé.\n\nCette action est irréversible.`
  )) return;

  try {
    const r = await api(`/api/mappings/${id}/deindex`, { method: 'POST' });
    if (r.status === 'started') {
      toast('Désindexation en cours — le mapping sera supprimé automatiquement', 'success');
      // On recharge après un court délai pour laisser le background task finir
      setTimeout(loadMappings, 2500);
    }
  } catch (e) { toast(`Erreur : ${e.message}`, 'error'); }
}

/* Désindexation d'un fichier individuel depuis le modal d'édition.
   local_path = chemin absolu dans le cache local du container.
   Le cache est dans nextcloud_cache/<collection>/<filename>. */
async function deindexFile(mappingId, localPath, displayName) {
  if (!confirm(`Désindexer le fichier "${displayName}" ?\n\nSes chunks seront supprimés de Qdrant.`)) return;
  try {
    const r = await api(`/api/mappings/${mappingId}/deindex-file`, {
      method: 'POST',
      body: JSON.stringify({ local_path: localPath }),
    });
    toast(`${r.deindexed_chunks} chunk(s) supprimé(s) pour ${displayName}`, 'success');
    // Retire visuellement le chemin de la liste des chemins sélectionnés
    const idx = S.selPaths.findIndex(p => p === localPath || localPath.endsWith(p.split('/').pop()));
    if (idx !== -1) { S.selPaths.splice(idx, 1); renderSelPaths(); }
    loadMappings();
  } catch (e) { toast(`Erreur : ${e.message}`, 'error'); }
}

/* Modal */
async function openModal(mappingId = null) {
  S.editingId  = mappingId;
  S.selPaths   = [];
  S.selInterval = 15;

  const titleEl = document.getElementById('modal-title');
  const btn     = document.getElementById('btn-create');

  if (mappingId) {
    const m = S.mappings.find(x => x.id === mappingId);
    if (!m) { toast('Mapping introuvable', 'error'); return; }

    titleEl.textContent = 'Modifier le mapping';
    btn.textContent     = 'Enregistrer →';
    S.selInterval = m.interval_minutes;

    /* Point 2 — rétrocompatibilité : remote_paths liste OU remote_path string */
    S.selPaths = [...(m.remote_paths || (m.remote_path ? [m.remote_path] : []))];

    document.getElementById('f-coll').value    = m.collection_name;
    document.getElementById('f-label').value   = m.label || '';
    document.getElementById('f-start-at').value = m.start_at || '';

    /* Navigateur : ouvre sur le dossier parent du premier chemin */
    const firstPath = S.selPaths[0] || '/';
    const parent = firstPath.split('/').slice(0, -1).join('/') || '/';
    await browse(parent);

  } else {
    titleEl.textContent = 'Nouveau mapping Nextcloud → RAG';
    btn.textContent     = 'Créer le mapping →';
    document.getElementById('f-coll').value     = '';
    document.getElementById('f-label').value    = '';
    document.getElementById('f-start-at').value = '';
    browse('/');
  }

  /* Sync de l'affichage des chemins et des boutons d'intervalle */
  renderSelPaths();
  document.querySelectorAll('.int-btn').forEach(b =>
    b.classList.toggle('active', parseInt(b.dataset.v) === S.selInterval)
  );

  document.getElementById('modal').classList.add('is-open');
}

function closeModal() {
  document.getElementById('modal').classList.remove('is-open');
  S.editingId = null;
  S.selPaths  = [];
}
function overlayClick(e) { if (e.target.id === 'modal') closeModal(); }

/* ── Gestion multi-chemins (point 2) */

/* Ajoute le chemin actuellement affiché dans le navigateur à la liste */
function addCurrentPath() {
  const pathEl = document.getElementById('sel-path');
  const path = pathEl.textContent.trim();
  if (!path || path === '—' || path === '/') {
    toast('Naviguez jusqu\'à un dossier ou fichier à ajouter', 'info');
    return;
  }
  if (S.selPaths.includes(path)) {
    toast('Ce chemin est déjà dans la liste', 'info');
    return;
  }
  S.selPaths.push(path);
  renderSelPaths();
}

/* Construit le local_path attendu par l'API deindex-file pour un chemin
   sélectionné, en mode édition. Partagé entre renderSelPaths() et
   removeSelPath() pour rester cohérent. */
function localPathForSelPath(p) {
  if (!S.editingId) return null;
  const mapping = S.mappings.find(x => x.id === S.editingId);
  if (!mapping) return null;
  const fileName = p.split('/').pop();
  return `nextcloud_cache/${mapping.collection_name}/${fileName}`;
}

/* Supprime un chemin de la liste.
   En mode édition, on ne retire JAMAIS le chemin localement avant d'avoir
   la confirmation du backend qu'il n'y a plus de chunks indexés pour ce
   fichier — sinon l'UI affiche un faux "désindexé" alors que les chunks
   sont toujours dans Qdrant (point 1 du rapport de bugs). */
async function removeSelPath(idx) {
  const path = S.selPaths[idx];
  if (path === undefined) return;

  if (!S.editingId) {
    // Mode création : aucun fichier n'a encore été indexé, retrait local sûr.
    S.selPaths.splice(idx, 1);
    renderSelPaths();
    return;
  }

  const localPath = localPathForSelPath(path);
  if (!localPath) {
    toast('Impossible de déterminer le chemin local pour ce fichier', 'error');
    return;
  }

  if (!confirm(
    `Retirer "${path}" du mapping ?\n\n` +
    `Si des chunks sont encore indexés pour ce fichier dans Qdrant, ils seront supprimés avant le retrait.`
  )) return;

  try {
    const r = await api(`/api/mappings/${S.editingId}/deindex-file`, {
      method: 'POST',
      body: JSON.stringify({ local_path: localPath }),
    });

    // On ne met à jour S.selPaths qu'après confirmation explicite du
    // backend — on ne fait jamais confiance à l'état local seul.
    const remaining = (typeof r.remaining_chunks === 'number') ? r.remaining_chunks : 0;
    if (remaining > 0) {
      toast(`${remaining} chunk(s) encore présent(s) — retrait bloqué`, 'error');
      return;
    }

    toast(`${r.deindexed_chunks} chunk(s) supprimé(s) pour ${path.split('/').pop()}`, 'success');
    S.selPaths.splice(idx, 1);
    renderSelPaths();
    loadMappings();
  } catch (e) {
    toast(`Erreur : ${e.message}`, 'error');
  }
}

/* Affiche la liste des chemins sélectionnés sous le navigateur.
   En mode édition (S.editingId), chaque chemin a un bouton "Désindexer"
   et le ✕ est bloqué tant que le fichier n'a pas été désindexé. */
function renderSelPaths() {
  const el = document.getElementById('sel-paths-list');
  if (!el) return;
  if (!S.selPaths.length) {
    el.innerHTML = '<div class="sel-paths-empty">Aucun chemin sélectionné — naviguez et cliquez « + Ajouter »</div>';
    return;
  }

  const isEdit = !!S.editingId;
  const mapping = isEdit ? S.mappings.find(x => x.id === S.editingId) : null;
  const cacheBase = mapping ? `/api/cache/${mapping.collection_name}/` : null;

  el.innerHTML = S.selPaths.map((p, i) => {
    const fileName = p.split('/').pop();
    /* En mode édition on construit le local_path pour l'API deindex-file.
       Le cache local est nextcloud_cache/<collection>/<filename>. */
    const localPath = isEdit ? localPathForSelPath(p) : null;

    const deindexBtn = isEdit
      ? `<button class="sel-path-deindex" title="Désindexer ce fichier"
           data-action="deindex-file"
           data-mapping-id="${esc(S.editingId)}"
           data-local-path="${esc(localPath || p)}"
           data-filename="${esc(fileName)}">
           ⊘ Désindexer
         </button>`
      : '';

    /* ✕ : en mode édition, le clic appelle désormais le backend pour
       désindexer le fichier avant de le retirer de la liste — la
       suppression locale seule (sans confirmation backend) est interdite. */
    const removeTitle = isEdit
      ? 'Désindexer puis retirer de la liste'
      : 'Retirer ce chemin';

    return `
      <div class="sel-path-tag">
        <span class="sel-path-icon">📁</span>
        <span class="sel-path-text">${esc(p)}</span>
        ${deindexBtn}
        <button class="sel-path-remove" data-action="remove-path" data-idx="${i}" title="${removeTitle}">✕</button>
      </div>`;
  }).join('');
}

/*  Navigateur Nextcloud */
async function browse(path) {
  /* On ne met plus à jour S.selPaths ici — le navigateur est juste un
     outil de navigation. Le chemin devient sélectionné seulement via
     addCurrentPath() ou un clic sur un fichier individuel. */
  document.getElementById('sel-path').textContent = path;
  updateBreadcrumb(path);

  const listEl = document.getElementById('browser-list');
  listEl.innerHTML = '<div class="browser-loading">Chargement…</div>';

  try {
    const d = await api(`/api/nextcloud/browse?path=${encodeURIComponent(path)}`);

    /* Point 2 — affiche aussi les fichiers individuels pour pouvoir les sélectionner */
    const dirs = (d.directories || []).map(dir => `
      <div class="browser-dir" data-action="browse" data-path="${esc(dir.path)}">
        <span class="browser-dir-icon">📁</span>
        <span class="browser-dir-name">${esc(dir.name)}</span>
        <span class="browser-dir-arrow">›</span>
      </div>`).join('');

    const files = (d.files || []).map(f => `
      <div class="browser-file" data-action="select-file" data-path="${esc(path + (path.endsWith('/') ? '' : '/') + f.name)}">
        <span class="browser-dir-icon">📄</span>
        <span class="browser-dir-name">${esc(f.name)}</span>
        <span class="browser-file-size">${fmtSize(f.size)}</span>
      </div>`).join('');

    if (!dirs && !files) {
      listEl.innerHTML = '<div class="browser-empty">📁 Dossier vide</div>';
    } else {
      listEl.innerHTML = dirs + files;
    }
  } catch (e) {
    listEl.innerHTML = `<div class="browser-error">Erreur : ${e.message}</div>`;
  }
}

/* Sélectionne un fichier individuel directement depuis le navigateur */
function selectFile(filePath) {
  document.getElementById('sel-path').textContent = filePath;
}

function updateBreadcrumb(path) {
  const bc = document.getElementById('bc');
  const parts = path.split('/').filter(Boolean);
  let items = [`<span class="bc-item" data-path="/">⌂</span>`];
  let cur = '';
  parts.forEach(p => {
    cur += '/' + p;
    const c = cur;
    items.push(`<span class="bc-sep">›</span>`);
    items.push(`<span class="bc-item" data-path="${c}">${esc(p)}</span>`);
  });
  bc.innerHTML = items.join('');
}

/* Création / Modification */
async function createMapping() {
  const coll    = document.getElementById('f-coll').value.trim();
  const label   = document.getElementById('f-label').value.trim();
  const startAt = document.getElementById('f-start-at').value;
  const isEdit  = !!S.editingId;

  if (!S.selPaths.length) { toast('Ajoutez au moins un chemin Nextcloud', 'error'); return; }
  if (!coll)               { toast('Entrez un nom de collection RAG', 'error');       return; }

  const btn = document.getElementById('btn-create');
  btn.disabled    = true;
  btn.textContent = isEdit ? 'Enregistrement…' : 'Création…';

  const payload = {
    remote_paths:     S.selPaths,
    collection_name:  coll,
    interval_minutes: S.selInterval,
    label:            label || null,
    start_at:         startAt || null,
  };

  try {
    if (isEdit) {
      await api(`/api/mappings/${S.editingId}`, {
        method: 'PUT',
        body: JSON.stringify(payload),
      });
      toast('Mapping mis à jour', 'success');
    } else {
      await api('/api/mappings', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      toast(startAt ? 'Mapping créé — sync planifiée' : 'Mapping créé — 1ère sync en cours', 'success');
    }
    closeModal();
    loadMappings();
  } catch (e) {
    toast(`Erreur : ${e.message}`, 'error');
  } finally {
    btn.disabled    = false;
    btn.textContent = isEdit ? 'Enregistrer →' : 'Créer le mapping →';
  }
}

/*  Historique */
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
    const ok      = h.status === 'success';
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

/* Intervalle (point 6 — hebdomadaire) */
function pickInt(btn) {
  S.selInterval = parseInt(btn.dataset.v);
  document.querySelectorAll('.int-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

/* Stats  */
function setStatVal(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  if (String(el.textContent) !== String(val)) {
    el.textContent = val;
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');
  }
}

/* Toasts */
function toast(msg, type = 'info') {
  const z = document.getElementById('toast-zone');
  const t = document.createElement('div');
  t.className  = `toast t-${type}`;
  t.textContent = msg;
  z.appendChild(t);
  requestAnimationFrame(() => { requestAnimationFrame(() => t.classList.add('show')); });
  setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => t.remove(), 300);
  }, 3200);
}

/* Helpers */
function relAgo(iso) {
  if (!iso) return '—';
  const s = Math.round((Date.now() - new Date(iso)) / 1000);
  if (s < 5)     return 'à l\'instant';
  if (s < 60)    return `il y a ${s}s`;
  if (s < 3600)  return `il y a ${Math.floor(s / 60)}min`;
  if (s < 86400) return `il y a ${Math.floor(s / 3600)}h`;
  return `il y a ${Math.floor(s / 86400)}j`;
}
function relFuture(iso) {
  if (!iso) return '—';
  const s = Math.round((new Date(iso) - Date.now()) / 1000);
  if (s <= 0)    return 'maintenant';
  if (s < 60)    return `dans ${s}s`;
  if (s < 3600)  return `dans ${Math.floor(s / 60)}min`;
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
  if (min < 60)    return `${min} min`;
  if (min < 1440)  return `${Math.floor(min / 60)} h`;
  if (min < 10080) return `${Math.floor(min / 1440)} j`;
  return `${Math.floor(min / 10080)} sem`;
}
function fmtSize(bytes) {
  if (!bytes) return '';
  if (bytes < 1024)       return `${bytes} o`;
  if (bytes < 1048576)    return `${(bytes / 1024).toFixed(0)} Ko`;
  return `${(bytes / 1048576).toFixed(1)} Mo`;
}
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

/* Boot */
document.addEventListener('DOMContentLoaded', init);