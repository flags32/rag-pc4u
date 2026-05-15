from rag_pc4u.core.tenancy import CLIENT_ID_FIELD, filter_for


def test_filter_for_sans_extra_retourne_un_filtre_plat() -> None:
    assert filter_for("acme") == {
        "field": CLIENT_ID_FIELD,
        "operator": "==",
        "value": "acme",
    }


def test_filter_for_avec_condition_simple_combine_en_and() -> None:
    extra = {"field": "meta.doc_type", "operator": "==", "value": "facture"}

    result = filter_for("acme", extra)

    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": CLIENT_ID_FIELD, "operator": "==", "value": "acme"},
            {"field": "meta.doc_type", "operator": "==", "value": "facture"},
        ],
    }


def test_filter_for_avec_extra_and_aplatit_les_conditions() -> None:
    extra = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.doc_type", "operator": "==", "value": "facture"},
            {"field": "meta.year", "operator": ">=", "value": 2024},
        ],
    }

    result = filter_for("acme", extra)

    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": CLIENT_ID_FIELD, "operator": "==", "value": "acme"},
            {"field": "meta.doc_type", "operator": "==", "value": "facture"},
            {"field": "meta.year", "operator": ">=", "value": 2024},
        ],
    }


def test_filter_for_avec_extra_or_ne_aplatit_pas() -> None:
    """Un OR doit rester imbriqué — sinon on changerait la sémantique."""
    extra = {
        "operator": "OR",
        "conditions": [
            {"field": "meta.doc_type", "operator": "==", "value": "facture"},
            {"field": "meta.doc_type", "operator": "==", "value": "devis"},
        ],
    }

    result = filter_for("acme", extra)

    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": CLIENT_ID_FIELD, "operator": "==", "value": "acme"},
            extra,
        ],
    }


def test_filter_for_preserve_le_client_id_meme_avec_un_extra_qui_filtre_client_id() -> None:
    """Garde-fou : si un appelant tente de filtrer sur client_id via extra,
    le client_id passé en argument reste celui qui s'applique en premier."""
    extra = {"field": CLIENT_ID_FIELD, "operator": "==", "value": "autre_client"}

    result = filter_for("acme", extra)

    assert result["conditions"][0] == {
        "field": CLIENT_ID_FIELD,
        "operator": "==",
        "value": "acme",
    }
