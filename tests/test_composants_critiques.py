import tempfile
import os
from pathlib import Path
from rag_pc4u.ingestion.sources import LocalDirectoryScanner


class TestLocalDirectoryScanner:
    def test_scan_trouve_fichiers_valides(self, tmp_path):
        (tmp_path / "doc.txt").write_text("contenu")
        (tmp_path / "notice.md").write_text("# titre")
        (tmp_path / "image.jpg").write_bytes(b"\xff\xd8")  # doit être ignoré

        scanner = LocalDirectoryScanner(allowed_extensions=[".txt", ".md"])
        result = scanner.run(directory_path=str(tmp_path))

        paths = result["paths"]
        assert len(paths) == 2
        extensions = {p.suffix for p in paths}
        assert extensions == {".txt", ".md"}

    def test_scan_repertoire_inexistant(self):
        scanner = LocalDirectoryScanner()
        result = scanner.run(directory_path="/chemin/qui/nexiste/pas")
        assert result["paths"] == []

    def test_scan_sous_repertoires(self, tmp_path):
        sub = tmp_path / "sous_dossier"
        sub.mkdir()
        (sub / "fichier.txt").write_text("contenu")
        (tmp_path / "racine.txt").write_text("contenu")

        scanner = LocalDirectoryScanner(allowed_extensions=[".txt"])
        result = scanner.run(directory_path=str(tmp_path))
        assert len(result["paths"]) == 2
