"""Conservative mappings from Folkets lexikon paradigms to named forms.

Folkets paradigms are positional rather than labelled.  The mappings below are
based on the ordering used by the published Swedish-English XML.  Unknown or
ambiguous positions are deliberately omitted so that later review/enrichment
can handle them instead of displaying a guessed form.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _field(entry: object, name: str, default: Any) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _forms(entry: object) -> list[str]:
    raw = _field(entry, "paradigm", _field(entry, "forms", ()))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    result: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            value = item.get("value", item.get("word", ""))
        else:
            value = item
        value = str(value or "").strip()
        if value:
            # Duplicate values can carry positional meaning (veta has vet as
            # both imperative and present), so never deduplicate paradigms.
            result.append(value)
    return result


def _usage(entry: object) -> str:
    raw = _field(entry, "usage", _field(entry, "use", ()))
    if isinstance(raw, str):
        return raw.casefold()
    if isinstance(raw, Sequence):
        return " ".join(str(item) for item in raw).casefold()
    return ""


def _same(left: str, right: str) -> bool:
    return (
        left.replace("|", "").strip().casefold()
        == right.replace("|", "").strip().casefold()
    )


def extract_inflections(entry: object, expected_pos: str) -> dict[str, str]:
    """Return only forms whose observed Folkets shape can be labelled safely."""
    forms = _forms(entry)
    headword = str(_field(entry, "headword", _field(entry, "word", "")) or "").strip()
    usage = _usage(entry)
    result: dict[str, str] = {}

    if expected_pos == "noun":
        # Most records are [definite, plural, definite plural]. A few expanded
        # records prepend the lemma; plural-only headwords prepend singular.
        if len(forms) == 4 and _same(forms[0], headword):
            forms = forms[1:]
        elif len(forms) == 4 and "plural" in usage:
            result["singular"] = forms[0]
            forms = forms[1:]
        elif len(forms) > 3:
            return result
        if forms:
            result["definite"] = forms[0]
        if len(forms) >= 2:
            result["plural"] = forms[1]
        if len(forms) == 3:
            result["definite_plural"] = forms[2]
        return result

    if expected_pos == "adjective":
        # Some expanded records prepend the lemma (betagen, betaget, betagna).
        if len(forms) == 3 and _same(forms[0], headword):
            forms = forms[1:]
        # A five-item adjective record in Folkets is a misclassified/non-adjective
        # paradigm; omit it rather than shifting labels onto verb forms.
        if len(forms) > 4:
            return result
        # Invariable positives (for example bra) list comparison forms only.
        if "oböjligt" in usage and len(forms) >= 2:
            result["comparative"] = forms[0]
            result["superlative"] = forms[1]
            return result
        if len(forms) == 4:
            result.update(
                neuter=forms[0],
                plural=forms[1],
                comparative=forms[2],
                superlative=forms[3],
            )
        elif len(forms) >= 2:
            result.update(neuter=forms[0], plural=forms[1])
        elif len(forms) == 1:
            if "superlativ saknas" in usage:
                result["comparative"] = forms[0]
            elif "komparativ" in usage:
                result["superlative"] = forms[0]
            else:
                # One changed form is normally the plural/definite form; its
                # neuter form is identical to the headword and is not repeated.
                result["plural"] = forms[0]
        return result

    if expected_pos == "verb":
        # Expanded six-item records can wrap the normal five-item paradigm with
        # a duplicate present form (omsätter ... omsätter).
        if (
            len(forms) == 6
            and _same(forms[0], forms[-1])
            and _same(forms[-2], headword)
        ):
            forms = forms[1:]

        # Rare alternate records put infinitive first.
        if (
            len(forms) in {4, 5}
            and _same(forms[0], headword)
            and not _same(forms[-2], headword)
        ):
            if len(forms) == 4:  # infinitive, present, preterite, supine
                return {"present": forms[1], "past": forms[2], "supine": forms[3]}
            if _same(
                forms[1], headword
            ):  # bry: infinitive, imperative, present, past, supine
                return {
                    "imperative": forms[1],
                    "present": forms[2],
                    "past": forms[3],
                    "supine": forms[4],
                }
            # åta: infinitive, present, preterite, supine, imperative
            return {
                "present": forms[1],
                "past": forms[2],
                "supine": forms[3],
                "imperative": forms[4],
            }

        if len(forms) == 5:
            # Standard: preterite, supine, imperative, infinitive, present.
            result.update(past=forms[0], supine=forms[1], present=forms[4])
            if _same(forms[3], headword):
                result["imperative"] = forms[2]
            elif _same(forms[2], headword) and _same(forms[3], forms[4]):
                result["imperative"] = forms[3]  # veta anomaly
            return result

        if len(forms) == 4:
            if _same(forms[-1], headword):
                # Present, preterite, supine, infinitive (utdela).
                return {"present": forms[0], "past": forms[1], "supine": forms[2]}
            # Standard short: preterite, supine, infinitive, present.
            return {"past": forms[0], "supine": forms[1], "present": forms[3]}

        if len(forms) == 3:
            if _same(forms[1], headword):
                return {"past": forms[0], "present": forms[2]}
            if _same(forms[-1], headword):
                return {"past": forms[0], "supine": forms[1], "present": headword}
            if headword.casefold() in {"ska", "skall"}:
                return {"past": forms[0], "supine": forms[1], "present": headword}
            return {"past": forms[0]}

        if len(forms) == 2:
            result.update(past=forms[0], supine=forms[1])
            if _same(forms[0], headword):
                result["present"] = headword  # måste
            return result
        if len(forms) == 1:
            return {"past": forms[0]}

        # Longer mixed/alternative paradigms are not positionally reliable. The
        # final form is only accepted as present when the infinitive appears
        # immediately before it; every other label is left for review.
        if len(forms) > 5 and _same(forms[-2], headword):
            return {"present": forms[-1]}
        return result

    return result
