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
from xml.etree import ElementTree as ET

import requests
import structlog

from rag_pc4u.core.config import settings
from rag_pc4u.ingestion.run import run_folder_ingestion

logger = structlog.get_logger(__name__)

DAV = "{DAV:}"

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ""}


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
                        "name": Path(item["href"]).name,
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
                    "name": Path(href.rstrip("/")).name,
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

    # Sync principale

    def sync(self, remote_path: str, collection_name: str) -> dict:
        """
        Synchronise un dossier Nextcloud vers une collection RAG.

        Protégé par un verrou par (remote_path, collection) pour empêcher
        deux syncs simultanées sur le même mapping.

        Returns:
            dict de stats : new, modified, deleted, errors, status, ...
        """
        lock_key = f"{remote_path}::{collection_name}"
        with self._get_lock(lock_key):
            return self._do_sync(remote_path, collection_name)

    def _do_sync(self, remote_path: str, collection_name: str) -> dict:
        started_at = datetime.now()
        stats: dict = {
            "remote_path": remote_path,
            "collection": collection_name,
            "new": 0,
            "modified": 0,
            "deleted": 0,
            "errors": 0,
            "started_at": started_at.isoformat(),
            "finished_at": None,
            "status": "running",
        }

        logger.info(
            "nextcloud.sync_start",
            remote_path=remote_path,
            collection=collection_name,
        )

        # Cache local dédié à cette collection
        local_dir = self.cache_base / collection_name
        local_dir.mkdir(parents=True, exist_ok=True)

        # État précédent {href: etag}
        previous_state = self._load_state(remote_path)
        new_state: dict[str, str] = {}

        # 1. Liste des fichiers distants
        try:
            remote_files = self.list_remote_files(remote_path)
        except Exception as e:
            logger.error("nextcloud.list_failed", error=str(e))
            stats.update(status="error", error_message=str(e),
                         finished_at=datetime.now().isoformat())
            return stats

        current_hrefs = {f["href"] for f in remote_files}

        # 2. Nouveaux / modifiés
        for file_info in remote_files:
            href = file_info["href"]
            etag = file_info["etag"] or file_info["modified"]
            new_state[href] = etag
            local_path = local_dir / file_info["name"]
            prev_etag = previous_state.get(href)

            if prev_etag is None:
                logger.info("nextcloud.new_file", href=href)
                ok = self.download_file(href, local_path)
                stats["new" if ok else "errors"] += 1
            elif prev_etag != etag:
                logger.info("nextcloud.modified_file", href=href)
                ok = self.download_file(href, local_path)
                stats["modified" if ok else "errors"] += 1
            # sinon : inchangé → pas de téléchargement

        # 3. Fichiers supprimés distants → supprimer du cache local
        for href in previous_state:
            if href not in current_hrefs:
                logger.info("nextcloud.deleted_file", href=href)
                local_path = local_dir / Path(href).name
                if local_path.exists():
                    local_path.unlink()
                stats["deleted"] += 1

        # 4. Ingestion incrémentale si des changements ont eu lieu
        if stats["new"] or stats["modified"] or stats["deleted"]:
            try:
                run_folder_ingestion(str(local_dir), collection_name)
            except Exception as e:
                logger.error("nextcloud.ingestion_failed", error=str(e))
                stats.update(status="error", error_message=str(e),
                             finished_at=datetime.now().isoformat())
                return stats
        else:
            logger.info("nextcloud.no_changes", collection=collection_name)

        # 5. Sauvegarde de l'état WebDAV
        self._save_state(remote_path, new_state)

        stats.update(status="success", finished_at=datetime.now().isoformat())
        logger.info("nextcloud.sync_done", **{k: v for k, v in stats.items() if k != "started_at"})
        return stats