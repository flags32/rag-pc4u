"""
Utilitaire de timezone partagé — garantit que toutes les heures affichées
ou persistées dans le dashboard sont en Europe/Paris, indépendamment de
la timezone système du conteneur Docker (souvent UTC par défaut).

Sans ce module, datetime.now() retourne l'heure du conteneur (UTC si TZ
n'est pas réglé), ce qui crée un décalage de 1h ou 2h (selon CET/CEST)
entre l'heure planifiée par l'utilisateur et l'heure réellement
enregistrée dans dashboard_state.json (last_sync, started_at, etc.).

Le fait de régler TZ=Europe/Paris dans docker-compose.yml résout déjà
le problème côté conteneur. Ce module garantit le même résultat même si
TZ n'est pas réglé (autre machine, autre environnement, oubli futur).

Usage :
    from rag_pc4u.core.tz_utils import now_paris
    horodatage = now_paris().isoformat()
"""

from datetime import datetime

import pytz

PARIS_TZ = pytz.timezone("Europe/Paris")


def now_paris() -> datetime:
    """Retourne l'heure actuelle, toujours en Europe/Paris (aware)."""
    return datetime.now(PARIS_TZ)


def now_paris_naive() -> datetime:
    """
    Variante sans tzinfo, pour les endroits où le code existant attend
    un datetime naive (ex: comparaisons avec d'anciennes valeurs déjà
    sauvegardées sans timezone). Garde la VALEUR horaire de Paris, juste
    sans l'objet tzinfo attaché.
    """
    return now_paris().replace(tzinfo=None)