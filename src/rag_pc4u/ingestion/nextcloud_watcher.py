"""
Connecteur Nextcloud / WebDAV pour l'ingestion RAG.

Utilise les ETag WebDAV pour détecter les changements sans télécharger
tous les fichiers à chaque poll.

Workflow :
  1. PROPFIND → liste les fichiers distants avec leur ETag
  2. Comparaison avec l'état précédent (JSON local)
  3. Téléchargement des fichiers nouveaux / modifiés dans un cache local
  4. Suppression des fichiers locaux orphelins
  5. Déclenchement de run_folder_ingestion() sur le cache local
  6. Sauvegarde du nouvel état
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import unquote
from xml.etree import ElementTree as ET

import requests
import structlog

from rag_pc4u.core.config import settings
from rag_pc4u.core.tz_utils import now_paris_naive
from rag_pc4u.ingestion.run import run_folder_ingestion, IngestionPendingFilesError

logger = structlog.get_logger(__name__)

DAV = "{DAV:}"

ALLOWED_EXTENSIONS = {
     "", ".txt", ".md", ".pdf", ".csv", ".docx", ".pptx", ".xlsx", ".html",
            # Images
            ".jpg", ".jpeg", ".png", ".tiff",
            # Données structurées
            ".json", ".xml",
            # Audio
            ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".flac",
            # Vidéo
            ".mp4", ".mov"
}


class NextcloudWatcher:
    """
    Surveille un dossier Nextcloud et déclenche l'ingestion RAG
    sur les fichiers nouveaux / modifiés / supprimés.

    Une instance unique est partagée par le scheduler et l'API.
    La méthode sync() est protégée par un verrou par (remote_path, collection).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        host = (host or settings.nextcloud_url).strip()
        if not host.startswith("http"):
            host = f"http://{host}"
        self.host = host.rstrip("/")
        self.user = user or settings.nextcloud_user
        self.password = password or settings.nextcloud_password

        # WebDAV root propre à cet utilisateur
        self.webdav_root = f"{self.host}/remote.php/dav/files/{self.user}"

        # Session HTTP persistante avec auth de base
        self.session = requests.Session()
        self.session.auth = (self.user, self.password)
        self.session.headers.update({"Content-Type": "application/xml; charset=utf-8"})

        # Cache local : un sous-dossier par collection
        self.cache_base = Path(__file__).parent / "nextcloud_cache"
        self.cache_base.mkdir(parents=True, exist_ok=True)

        # État WebDAV (ETags) — séparé de l'état d'ingestion géré par run.py
        self.state_base = Path(__file__).parent / "fichier_injecter"
        self.state_base.mkdir(parents=True, exist_ok=True)

        # Verrous par mapping pour éviter les syncs parallèles sur le même dossier
        self._locks: dict[str, threading.Lock] = {}
        self._locks_meta = threading.Lock()

        logger.info(
            "nextcloud_watcher.initialized",
            host=self.host,
            user=self.user,
            webdav_root=self.webdav_root,
        )

    # Connexion

    def test_connection(self) -> bool:
        """Vérifie que Nextcloud est joignable et que les credentials sont valides."""
        try:
            resp = self.session.request(
                "PROPFIND",
                self.webdav_root + "/",
                headers={"Depth": "0"},
                data="""<?xml version="1.0"?><D:propfind xmlns:D="DAV:">
                    <D:prop><D:resourcetype/></D:prop></D:propfind>""",
                timeout=10,
            )
            return resp.status_code in (200, 207)
        except requests.RequestException as e:
            logger.warning("nextcloud.connection_test_failed", error=str(e))
            return False

    # WebDAV : PROPFIND

    def _propfind(self, remote_path: str, depth: int = 1) -> list[dict]:
        """
        Envoie une requête PROPFIND et parse la réponse XML multistatus.
        Retourne une liste de dicts :
          {href, is_dir, etag, modified, size}
        """
        url = f"{self.webdav_root}/{remote_path.lstrip('/')}"
        body = """<?xml version="1.0" encoding="utf-8"?>
        <D:propfind xmlns:D="DAV:">
            <D:prop>
                <D:resourcetype/>
                <D:getetag/>
                <D:getlastmodified/>
                <D:getcontentlength/>
            </D:prop>
        </D:propfind>"""

        resp = self.session.request(
            "PROPFIND",
            url,
            headers={"Depth": str(depth)},
            data=body,
            timeout=30,
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        items: list[dict] = []

        for response in root.findall(f"{DAV}response"):
            href = response.findtext(f"{DAV}href", "")
            propstat = response.find(f"{DAV}propstat")
            if propstat is None:
                continue
            prop = propstat.find(f"{DAV}prop")
            if prop is None:
                continue

            resourcetype = prop.find(f"{DAV}resourcetype")
            is_dir = (
                resourcetype is not None
                and resourcetype.find(f"{DAV}collection") is not None
            )

            raw_etag = prop.findtext(f"{DAV}getetag") or ""
            items.append(
                {
                    "href": href,
                    "is_dir": is_dir,
                    "etag": raw_etag.strip('"').strip("'"),
                    "modified": prop.findtext(f"{DAV}getlastmodified") or "",
                    "size": int(prop.findtext(f"{DAV}getcontentlength") or 0),
                }
            )

        return items

    # Listing

    def list_remote_files(self, remote_path: str) -> list[dict]:
        """
        Liste les fichiers indexables d'un dossier Nextcloud.
        Filtre par extension et ignore les dossiers.
        Le 1er item PROPFIND est le dossier lui-même — on le saute.
        """
        items = self._propfind(remote_path, depth=1)
        files = []
        for item in items[1:]:
            if item["is_dir"]:
                continue
            ext = Path(item["href"]).suffix.lower()
            if ext in ALLOWED_EXTENSIONS:
                files.append(
                    {
                        "href": item["href"],
                        # unquote() décode les caractères URL-encodés dans le
                        # href (%C3%A9 → é, etc.). Le href brut est conservé
                        # intact pour les requêtes WebDAV ; seul le nom affiché
                        # et utilisé comme nom de fichier local est décodé.
                        "name": unquote(Path(item["href"]).name),
                        "etag": item["etag"],
                        "modified": item["modified"],
                        "size": item["size"],
                    }
                )

        logger.info(
            "nextcloud.files_listed",
            remote_path=remote_path,
            count=len(files),
        )
        return files

    def _webdav_rel(self, href: str) -> str:
        """
        Chemin d'un href WebDAV relatif à la racine de l'utilisateur, forme
        BRUTE (encodée) — réutilisable telle quelle pour un PROPFIND de
        sous-dossier (les segments %XX restent valides dans une URL).
        """
        webdav_prefix = f"/remote.php/dav/files/{self.user}"
        rel = href[len(webdav_prefix):] if href.startswith(webdav_prefix) else href
        return rel.strip("/")

    def list_remote_files_recursive(self, remote_path: str) -> list[dict]:
        """
        Liste RÉCURSIVEMENT les fichiers indexables d'un dossier Nextcloud
        et de tous ses sous-dossiers.

        On descend niveau par niveau (PROPFIND Depth:1) au lieu d'utiliser
        Depth:infinity, souvent désactivé côté serveur Nextcloud. Un ensemble
        `visited` protège contre d'éventuelles boucles.

        C'est cette variante récursive — et non list_remote_files (depth=1) —
        qui alimente la synchro : sans elle, les fichiers rangés dans des
        sous-dossiers ne seraient jamais indexés, et leur suppression (ou
        celle du sous-dossier entier) ne pourrait donc pas être propagée.
        """
        files: list[dict] = []
        stack = [remote_path]
        visited: set[str] = set()

        while stack:
            current = stack.pop()
            key = current.strip("/")
            if key in visited:
                continue
            visited.add(key)

            items = self._propfind(current, depth=1)
            # items[0] est le dossier courant lui-même — on le saute.
            for item in items[1:]:
                href = item["href"]
                if item["is_dir"]:
                    sub_rel = self._webdav_rel(href)
                    if sub_rel:
                        stack.append(sub_rel)
                    continue
                ext = Path(unquote(href)).suffix.lower()
                if ext in ALLOWED_EXTENSIONS:
                    files.append(
                        {
                            "href": href,
                            "name": Path(unquote(href)).name,
                            "etag": item["etag"],
                            "modified": item["modified"],
                            "size": item["size"],
                        }
                    )

        logger.info(
            "nextcloud.files_listed_recursive",
            remote_path=remote_path,
            count=len(files),
        )
        return files

    def resolve_remote_path(self, remote_path: str) -> list[dict]:
        """
        Point 2 — Résout un chemin Nextcloud en liste de fichiers indexables,
        que ce chemin pointe vers un DOSSIER ou un FICHIER INDIVIDUEL.

        - Dossier  → liste tous les fichiers indexables (depth=1, non récursif).
        - Fichier  → retourne ce fichier seul (extension non filtrée : un
          fichier choisi explicitement par l'utilisateur est toujours inclus).

        Lève FileNotFoundError si le chemin n'existe pas côté Nextcloud
        (que ce soit parce que le serveur répond 404, ou parce que le
        PROPFIND aboutit mais renvoie une liste vide).
        """
        try:
            probe = self._propfind(remote_path, depth=0)
        except requests.exceptions.HTTPError as e:
            # Un chemin supprimé côté Nextcloud répond généralement 404 au
            # PROPFIND — sans cette conversion explicite, raise_for_status()
            # lève une HTTPError « brute » qui finissait dans le except
            # Exception générique de _do_sync (traitée comme une simple
            # erreur réseau), au lieu du except FileNotFoundError dédié au
            # nettoyage du cache local. Résultat : les fichiers supprimés
            # côté Nextcloud n'étaient jamais retirés du cache, donc jamais
            # désindexés de Qdrant.
            status = e.response.status_code if e.response is not None else None
            if status == 404:
                raise FileNotFoundError(
                    f"Chemin Nextcloud introuvable : {remote_path}"
                ) from e
            raise

        if not probe:
            raise FileNotFoundError(f"Chemin Nextcloud introuvable : {remote_path}")

        target = probe[0]
        if target["is_dir"]:
            # Récursif : inclut les fichiers des sous-dossiers, indispensable
            # pour que la suppression d'un sous-dossier soit détectée et
            # propagée au cache local et à Qdrant.
            return self.list_remote_files_recursive(remote_path)

        return [{
            "href": target["href"],
            "name": Path(unquote(target["href"])).name,
            "etag": target["etag"],
            "modified": target["modified"],
            "size": target["size"],
        }]

    def list_remote_dirs(self, remote_path: str) -> list[dict]:
        """
        Liste les sous-dossiers d'un chemin pour le navigateur du dashboard.
        Retourne des chemins relatifs à la racine WebDAV de l'utilisateur.
        """
        items = self._propfind(remote_path, depth=1)
        result = []
        webdav_prefix = f"/remote.php/dav/files/{self.user}"

        for item in items[1:]:
            if not item["is_dir"]:
                continue
            href = item["href"]
            # Chemin relatif à la racine de l'utilisateur
            rel = href[len(webdav_prefix):] if href.startswith(webdav_prefix) else href
            rel = rel.rstrip("/") or "/"
            result.append(
                {
                    "path": rel,
                    "name": unquote(Path(href.rstrip("/")).name),
                    "type": "directory",
                }
            )
        return result

    # Téléchargement

    def download_file(self, href: str, local_path: Path) -> bool:
        """
        Télécharge un fichier depuis son href WebDAV absolu.
        Le href commence par /remote.php/dav/... → on préfixe par self.host.
        """
        url = f"{self.host}{href}"
        try:
            resp = self.session.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65_536):
                    fh.write(chunk)
            logger.info("nextcloud.downloaded", href=href, local=str(local_path))
            return True
        except Exception as e:
            logger.error("nextcloud.download_failed", href=href, error=str(e))
            return False

    # État WebDAV

    def _state_file(self, remote_path: str) -> Path:
        safe = remote_path.replace("/", "_").replace(":", "_").replace(" ", "_").strip("_")
        return self.state_base / f".nextcloud_etag_{safe}.json"

    def _load_state(self, remote_path: str) -> dict:
        f = self._state_file(remote_path)
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                logger.warning("nextcloud.state_load_failed", path=str(f))
        return {}

    def _save_state(self, remote_path: str, state: dict) -> None:
        self._state_file(remote_path).write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _get_lock(self, key: str) -> threading.Lock:
        with self._locks_meta:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    # Chemins de cache

    def _rel_to_remote(self, remote_path: str, href: str) -> str:
        """
        Chemin de cache local (relatif à local_dir) pour un fichier, calculé
        depuis son href WebDAV et le chemin racine du mapping.

        - Fichier directement sous remote_path → nom de fichier seul
          (cache « à plat », identique à l'ancien comportement).
        - Fichier dans un sous-dossier         → "sous/dossier/fichier",
          ce qui reproduit l'arborescence Nextcloud dans le cache et évite
          les collisions de noms entre sous-dossiers.

        Le calcul se fait en forme DÉCODÉE (unquote) : les fichiers avec
        espaces/accents sont stockés avec un nom lisible plutôt que %XX.
        """
        decoded = unquote(href)
        base = f"/remote.php/dav/files/{self.user}/{remote_path.strip('/')}/"
        if decoded.startswith(base):
            rel = decoded[len(base):].strip("/")
            if rel:
                return rel
        # remote_path pointe sur le fichier lui-même, ou href inattendu →
        # repli sur le nom de fichier seul.
        return Path(decoded).name

    def _remove_legacy_flat_copy(
        self, local_dir: Path, remote_path: str, href: str, new_local_path: Path
    ) -> None:
        """
        Migration : l'ancien schéma de cache stockait les fichiers À PLAT sous
        leur nom ENCODÉ (ex. « caf%C3%A9.pdf »). Le nouveau schéma utilise un
        nom DÉCODÉ (et imbriqué pour les sous-dossiers). Avant de télécharger
        un fichier sous son nouveau chemin, on retire une éventuelle ancienne
        copie encodée pour éviter un doublon (ancien fichier orphelin + copie
        neuve) — les chunks Qdrant de l'orphelin sont ensuite purgés par
        run_folder_ingestion, qui ne le retrouvera plus sur le disque.

        À n'appeler QUE quand un téléchargement va réellement avoir lieu :
        sur un fichier inchangé, l'ancienne copie EST la copie de travail, la
        supprimer entraînerait une perte de données.

        Restreint aux fichiers de PREMIER NIVEAU (« rel » sans « / ») : c'est
        le seul cas que l'ancien schéma non-récursif pouvait produire. On évite
        ainsi de supprimer par erreur un fichier de même nom rangé ailleurs.
        """
        rel = self._rel_to_remote(remote_path, href)
        if "/" in rel:
            return  # sous-dossier : jamais mis en cache par l'ancien schéma

        legacy = local_dir / Path(href).name  # ancien nom : basename ENCODÉ
        if legacy != new_local_path and legacy.exists():
            try:
                legacy.unlink()
                logger.info("nextcloud.legacy_cache_copy_removed", path=str(legacy))
            except OSError as e:
                logger.warning(
                    "nextcloud.legacy_cache_removal_failed",
                    path=str(legacy),
                    error=str(e),
                )

    def _cleanup_empty_dirs(self, local_dir: Path, removed_file: Path) -> None:
        """
        Après suppression d'un fichier, remonte l'arborescence et retire les
        sous-dossiers devenus vides, en s'arrêtant à local_dir (jamais
        supprimé). Un sous-dossier supprimé côté Nextcloud ne laisse ainsi
        pas de dossier fantôme dans le cache.
        """
        parent = removed_file.parent
        while parent != local_dir and local_dir in parent.parents:
            try:
                next(parent.iterdir())
                break  # dossier non vide → on arrête de remonter
            except StopIteration:
                try:
                    parent.rmdir()
                    logger.info("nextcloud.empty_dir_removed", path=str(parent))
                except OSError:
                    break
                parent = parent.parent
            except FileNotFoundError:
                break

    # Réconciliation stats / résultat réel de l'ingestion RAG

    def _reconcile_pending(
        self,
        stats: dict,
        pending_files: list[str],
        downloaded_this_cycle: dict[str, str],
    ) -> None:
        """
        CORRECTIF : stats["new"]/["modified"] étaient incrémentés au moment
        du TÉLÉCHARGEMENT (download_file réussi), pas au moment où le fichier
        est réellement écrit dans Qdrant. Un fichier peut très bien se
        télécharger correctement puis échouer plus loin, lors de l'ingestion
        RAG proprement dite (run_folder_ingestion / _worker_job), et se
        retrouver dans pending_files sans que stats["new"] ne soit jamais
        décrémenté ni stats["errors"] incrémenté — le dashboard affichait
        alors un succès ("new") pour un fichier qui n'a en réalité jamais été
        indexé.

        Pour chaque fichier présent dans pending_files, si on l'avait compté
        "new" ou "modified" à l'étape 2, on retire ce compte et on l'ajoute à
        "errors" à la place. Un fichier de pending_files qui n'a PAS été
        téléchargé ce cycle-ci (kind is None) est un échec reporté d'un
        cycle précédent, relancé via previous_pending_files (cf. _do_sync) :
        on l'ajoute quand même à "errors" pour que ce compteur reflète le
        nombre réel de fichiers actuellement en échec, pas seulement ceux
        fraîchement téléchargés.
        """
        for pending_path in pending_files:
            kind = downloaded_this_cycle.get(pending_path)
            if kind in ("new", "modified"):
                stats[kind] -= 1
            stats["errors"] += 1

    # Sync principale

    def sync(
        self,
        remote_paths: list[str],
        collection_name: str,
        previous_pending_files: Optional[list[str]] = None,
    ) -> dict:
        """
        Point 2 — Synchronise une liste de chemins Nextcloud (dossiers ET/OU
        fichiers individuels) vers une collection RAG en un seul cycle.

        Protégé par un verrou par (remote_paths, collection).

        CORRECTIF : previous_pending_files (fichiers restés en échec après
        le cycle précédent, cf. mapping["pending_files"] dans state.py)
        permet de forcer une nouvelle tentative de run_folder_ingestion même
        si rien n'a changé côté Nextcloud à ce cycle — sinon un fichier qui
        échoue à l'ingestion RAG reste bloqué indéfiniment tant qu'aucun
        AUTRE fichier du mapping ne change (cf. _do_sync, étape 4).
        """
        lock_key = f"{'|'.join(sorted(remote_paths))}::{collection_name}"
        with self._get_lock(lock_key):
            return self._do_sync(remote_paths, collection_name, previous_pending_files or [])

    def _do_sync(
        self,
        remote_paths: list[str],
        collection_name: str,
        previous_pending_files: list[str] = None,
    ) -> dict:
        previous_pending_files = previous_pending_files or []
        started_at = now_paris_naive()
        stats: dict = {
            "remote_paths": remote_paths,
            "collection": collection_name,
            "new": 0,
            "modified": 0,
            "deleted": 0,
            "errors": 0,
            "started_at": started_at.isoformat(),
            "finished_at": None,
            "status": "running",
            "pending_files": [],
        }

        logger.info(
            "nextcloud.sync_start",
            remote_paths=remote_paths,
            collection=collection_name,
        )

        # Cache local partagé par tous les chemins du mapping — un seul appel
        # run_folder_ingestion à la fin pour ne pas fragmenter le cycle
        # delete-puis-réindex entre plusieurs chemins d'un même mapping.
        local_dir = self.cache_base / collection_name
        local_dir.mkdir(parents=True, exist_ok=True)

        had_list_error = False
        # Fichiers effectivement écrits dans le cache local à ce cycle (clé =
        # chemin local absolu, valeur = "new"/"modified") — sert à réconcilier
        # stats avec pending_files une fois l'ingestion RAG tentée (étape 4).
        downloaded_this_cycle: dict[str, str] = {}

        for remote_path in remote_paths:
            previous_state = self._load_state(remote_path)
            new_state: dict[str, str] = {}

            # 1. Résolution du chemin (dossier OU fichier individuel — point 2)
            try:
                remote_files = self.resolve_remote_path(remote_path)
            except FileNotFoundError:
                # Le chemin lui-même n'existe plus donc tous ses fichiers sont supprimés
                logger.info("nextcloud.path_deleted", path=remote_path)
                for href in previous_state:
                    local_path = local_dir / self._rel_to_remote(remote_path, href)
                    if local_path.exists():
                        local_path.unlink()
                        self._cleanup_empty_dirs(local_dir, local_path)
                    stats["deleted"] += 1
                self._save_state(remote_path, {})  # vider l'état
                continue
            except Exception as e:
                logger.error("nextcloud.list_failed", path=remote_path, error=str(e))
                stats["errors"] += 1
                had_list_error = True
                # On continue avec les autres chemins du mapping plutôt que
                # d'abandonner toute la sync à cause d'un seul chemin en erreur.
                continue

            current_hrefs = {f["href"] for f in remote_files}

            # 2. Nouveaux / modifiés
            for file_info in remote_files:
                href = file_info["href"]
                etag = file_info["etag"] or file_info["modified"]
                # Chemin imbriqué (arborescence des sous-dossiers préservée).
                # download_file crée les dossiers parents au besoin.
                local_path = local_dir / self._rel_to_remote(remote_path, href)
                prev_etag = previous_state.get(href)

                if prev_etag is None:
                    logger.info("nextcloud.new_file", href=href)
                    self._remove_legacy_flat_copy(local_dir, remote_path, href, local_path)
                    ok = self.download_file(href, local_path)
                    stats["new" if ok else "errors"] += 1
                    # CORRECTIF : on ne mémorise l'ETag que si le téléchargement
                    # a réussi. Avant ce correctif, new_state[href] = etag était
                    # écrit inconditionnellement plus haut, AVANT même la
                    # tentative de download : un échec de download_file() était
                    # quand même persisté dans le mapping ETag, donc au cycle
                    # suivant prev_etag == etag et le fichier n'était plus
                    # jamais retenté alors qu'il n'avait jamais atterri dans
                    # le cache local.
                    if ok:
                        new_state[href] = etag
                        downloaded_this_cycle[str(local_path.resolve())] = "new"
                elif prev_etag != etag:
                    logger.info("nextcloud.modified_file", href=href)
                    self._remove_legacy_flat_copy(local_dir, remote_path, href, local_path)
                    ok = self.download_file(href, local_path)
                    stats["modified" if ok else "errors"] += 1
                    # CORRECTIF : en cas d'échec on conserve l'ANCIEN etag
                    # (prev_etag) plutôt que le nouveau, pour que le fichier
                    # reste détecté comme "modifié" — et donc retenté — au
                    # prochain cycle au lieu d'être considéré à jour.
                    new_state[href] = etag if ok else prev_etag
                    if ok:
                        downloaded_this_cycle[str(local_path.resolve())] = "modified"
                else:
                    new_state[href] = etag

            # 3. Fichiers supprimés distants → supprimer du cache local
            for href in previous_state:
                if href not in current_hrefs:
                    logger.info("nextcloud.deleted_file", href=href)
                    local_path = local_dir / self._rel_to_remote(remote_path, href)
                    if local_path.exists():
                        local_path.unlink()
                        self._cleanup_empty_dirs(local_dir, local_path)
                    stats["deleted"] += 1

            # État sauvegardé par chemin individuel — plus robuste si la liste
            # de chemins d'un mapping change entre deux cycles.
            self._save_state(remote_path, new_state)

        # 4. Ingestion unique sur l'ensemble du cache fusionné
        # CORRECTIF : on force aussi le passage si previous_pending_files
        # n'est pas vide — sinon un fichier resté en échec la fois d'avant
        # ne serait jamais retenté tant qu'aucun new/modified/deleted ne
        # survient par ailleurs dans le mapping (cf. docstring de sync()).
        if stats["new"] or stats["modified"] or stats["deleted"] or previous_pending_files:
            try:
                ingestion_result = run_folder_ingestion(str(local_dir), collection_name)
                stats["pending_files"] = ingestion_result.get("fichiers_en_attente", [])
                # CORRECTIF : réconcilie new/modified/errors avec le résultat
                # réel de l'ingestion (cf. _reconcile_pending ci-dessus).
                self._reconcile_pending(stats, stats["pending_files"], downloaded_this_cycle)
            except IngestionPendingFilesError as e:
                logger.error(
                    "nextcloud.ingestion_failed",
                    error=str(e),
                    pending_files=e.pending_files,
                )
                # CORRECTIF : idem, même quand run_folder_ingestion lève
                # IngestionPendingFilesError plutôt que de retourner un dict.
                self._reconcile_pending(stats, e.pending_files, downloaded_this_cycle)
                stats.update(
                    status="error",
                    error_message=str(e),
                    finished_at=now_paris_naive().isoformat(),
                    pending_files=e.pending_files,
                )
                return stats
            except Exception as e:
                logger.error("nextcloud.ingestion_failed", error=str(e))
                stats.update(
                    status="error",
                    error_message=str(e),
                    finished_at=now_paris_naive().isoformat(),
                )
                return stats
        else:
            logger.info("nextcloud.no_changes", collection=collection_name)

        if had_list_error:
            stats.update(
                status="error",
                error_message="Un ou plusieurs chemins Nextcloud du mapping sont inaccessibles.",
                finished_at=now_paris_naive().isoformat(),
            )
            return stats

        stats.update(status="success", finished_at=now_paris_naive().isoformat())
        logger.info(
            "nextcloud.sync_done",
            **{k: v for k, v in stats.items() if k != "started_at"},
        )
        return stats