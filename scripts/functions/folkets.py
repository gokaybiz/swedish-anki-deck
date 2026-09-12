"""Streaming reader and conservative adapter for Folkets lexikon XML."""

from __future__ import annotations

import html
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from functions.inflection import extract_inflections

POS_CLASSES = {
    "noun": {"nn"},
    "verb": {"vb"},
    "adjective": {"jj"},
    "adverb": {"ab"},
    "pronoun": {"pn"},
    "determiner": {"pn"},
    "preposition": {"pp"},
    "conjunction": {"kn", "sn"},
    "interjection": {"in"},
    "particle": {"ab", "pp"},
    "numeral": {"rg"},
    # Older Folkets records also use pn for some proper names.
    "proper noun": {"pm", "pn"},
}
PLURAL_LINK = re.compile(r"plural\s+till\s+[\"“]?([^\"”]+)", re.IGNORECASE)


def source_text(value: object) -> str:
    """Decode Folkets attributes, some of which contain double-escaped HTML."""
    return unicodedata.normalize("NFC", html.unescape(str(value or ""))).strip()


def normalize(value: str) -> str:
    # Folkets uses | as an internal compound/morpheme boundary, not spelling.
    return source_text(value).replace("|", "").casefold()


@dataclass(frozen=True)
class FolketsRecord:
    headword: str
    word_class: str
    translations: tuple[str, ...]
    source_definitions: tuple[str, ...]
    definition_translations: tuple[str, ...]
    examples: tuple[tuple[str, str], ...]
    paradigm: tuple[str, ...]
    usage: tuple[str, ...]
    pronunciations: tuple[str, ...]
    variants: tuple[str, ...]
    constructions: tuple[str, ...]
    translation_comments: tuple[str, ...]
    explanations: tuple[str, ...]
    synonyms: tuple[str, ...]
    compounds: tuple[str, ...]
    derivations: tuple[str, ...]
    idioms: tuple[str, ...]
    source_order: int


class FolketsLexicon:
    """An in-memory headword index built with bounded-memory XML parsing."""

    def __init__(self, records: list[FolketsRecord], metadata: dict[str, str]):
        self.records = records
        self.metadata = metadata
        self._by_headword: dict[str, list[FolketsRecord]] = defaultdict(list)
        self._by_paradigm: dict[str, list[FolketsRecord]] = defaultdict(list)
        self._plural_for: dict[str, list[FolketsRecord]] = defaultdict(list)
        for record in records:
            self._by_headword[normalize(record.headword)].append(record)
            for form in record.paradigm:
                self._by_paradigm[normalize(form)].append(record)
            for marker in record.usage:
                match = PLURAL_LINK.search(marker)
                if match:
                    self._plural_for[normalize(match.group(1))].append(record)

    @classmethod
    def from_file(cls, path: Path) -> FolketsLexicon:
        if not path.is_file():
            raise FileNotFoundError(path)
        records: list[FolketsRecord] = []
        metadata: dict[str, str] = {}
        order = 0
        for event, element in ET.iterparse(path, events=("start", "end")):
            if event == "start" and element.tag == "dictionary" and not metadata:
                metadata = dict(element.attrib)
            if event != "end" or element.tag != "word":
                continue
            order += 1
            headword = source_text(element.get("value"))
            word_class = source_text(element.get("class"))
            if headword and word_class:
                translations = tuple(
                    value
                    for child in element.findall("translation")
                    if (value := source_text(child.get("value")))
                )
                source_definitions = tuple(
                    value
                    for definition in element.findall("definition")
                    if (value := source_text(definition.get("value")))
                )
                definition_translations = tuple(
                    value
                    for definition in element.findall("definition")
                    for child in definition.findall("translation")
                    if (value := source_text(child.get("value")))
                )
                examples: list[tuple[str, str]] = []
                for example in element.findall("example"):
                    source = source_text(example.get("value"))
                    target_node = example.find("translation")
                    target = (
                        source_text(target_node.get("value"))
                        if target_node is not None
                        else ""
                    )
                    if source:
                        examples.append((source, target))
                paradigm_node = element.find("paradigm")
                paradigm = tuple(
                    value
                    for child in (
                        paradigm_node.findall("inflection")
                        if paradigm_node is not None
                        else []
                    )
                    if (
                        value := source_text(child.get("value"))
                        .replace("|", "")
                        .rstrip("!")
                    )
                )
                usage = cls._values(element, "use")
                records.append(
                    FolketsRecord(
                        headword=headword,
                        word_class=word_class,
                        translations=translations,
                        source_definitions=source_definitions,
                        definition_translations=definition_translations,
                        examples=tuple(examples),
                        paradigm=paradigm,
                        usage=usage,
                        pronunciations=cls._values(element, "phonetic"),
                        variants=cls._values(element, "variant"),
                        constructions=cls._values(element, "grammar"),
                        translation_comments=tuple(
                            value
                            for child in (
                                list(element.findall("translation"))
                                + [
                                    translation
                                    for definition in element.findall("definition")
                                    for translation in definition.findall("translation")
                                ]
                            )
                            if (value := source_text(child.get("comment")))
                        ),
                        explanations=cls._values(element, "explanation"),
                        synonyms=cls._values(element, "synonym"),
                        compounds=cls._values(element, "compound"),
                        derivations=cls._values(element, "derivation"),
                        idioms=cls._values(element, "idiom"),
                        source_order=order,
                    )
                )
            element.clear()
        if not metadata or metadata.get("source-language") != "sv":
            raise ValueError(f"{path} is not a Swedish-source Folkets dictionary")
        return cls(records, metadata)

    @staticmethod
    def _values(element: ET.Element, tag: str) -> tuple[str, ...]:
        return tuple(
            value
            for child in element.findall(tag)
            if (value := source_text(child.get("value")))
        )

    def lookup(self, headword: str, expected_pos: str) -> list[FolketsRecord]:
        classes = POS_CLASSES.get(expected_pos, set())
        return [
            record
            for record in self._by_headword.get(normalize(headword), [])
            if record.word_class in classes
        ]

    def definitions(
        self, records: list[FolketsRecord], limit: int = 2
    ) -> list[dict[str, object]]:
        # Repeated glosses across sense records are a useful weak signal for the
        # common reading of an otherwise indistinguishable homograph.
        grouped: dict[
            tuple[str, tuple[str, ...]],
            list[tuple[int, str, str, str, tuple[str, ...]]],
        ] = defaultdict(list)
        broad_gloss_counts: Counter[str] = Counter()
        broad_best_candidates: dict[str, list[tuple[bool, bool, int]]] = defaultdict(
            list
        )
        for record in records:
            glosses = record.translations or record.definition_translations
            if glosses:
                gloss = "; ".join(dict.fromkeys(glosses[:3])).casefold()
                broad_gloss_counts[gloss] += 1
                example, translation = next(
                    ((source, target) for source, target in record.examples if target),
                    record.examples[0] if record.examples else ("", ""),
                )
                broad_best_candidates[gloss].append(
                    (not bool(translation), not bool(example), record.source_order)
                )
        broad_best_orders = {
            gloss: min(candidates)[2]
            for gloss, candidates in broad_best_candidates.items()
        }
        for record in records:
            glosses = record.translations or record.definition_translations
            if not glosses:
                continue
            definition = "; ".join(dict.fromkeys(glosses[:3]))
            example, translation = next(
                ((source, target) for source, target in record.examples if target),
                record.examples[0] if record.examples else ("", ""),
            )
            sense_evidence = record.definition_translations or (
                record.source_definitions
                if broad_gloss_counts[definition.casefold()] > 1
                else ()
            )
            evidence_identity = tuple(
                normalize(value)
                for value in (
                    *record.source_definitions,
                    *record.definition_translations,
                )
            )
            grouped[(definition.casefold(), evidence_identity)].append(
                (
                    record.source_order,
                    definition,
                    example,
                    translation,
                    sense_evidence,
                )
            )

        ranked: list[tuple[int, int, int, str, str, str, tuple[str, ...]]] = []
        for candidates in grouped.values():
            best = min(
                candidates,
                key=lambda item: (not bool(item[3]), not bool(item[2]), item[0]),
            )
            ranked.append(
                (
                    -len(candidates),
                    -int(bool(best[3])),
                    best[0],
                    best[1],
                    best[2],
                    best[3],
                    best[4],
                )
            )
        ranked.sort()
        # Preserve broad-gloss diversity before spending the two-sense budget
        # on source-distinguished readings of the same English gloss. This adds
        # physical/figurative splits such as läger without displacing an
        # already retained, differently translated sense elsewhere.
        readings_by_gloss: dict[
            str, list[tuple[int, int, int, str, str, str, tuple[str, ...]]]
        ] = defaultdict(list)
        for candidate in ranked:
            readings_by_gloss[candidate[3].casefold()].append(candidate)
        broad_readings: list[tuple[int, int, int, str, str, str, tuple[str, ...]]] = []
        additional_readings: list[
            tuple[int, int, int, str, str, str, tuple[str, ...]]
        ] = []
        for gloss, candidates in readings_by_gloss.items():
            primary = min(
                candidates,
                key=lambda candidate: (
                    candidate[2] != broad_best_orders[gloss],
                    candidate,
                ),
            )
            broad_readings.append(primary)
            additional_readings.extend(
                candidate for candidate in candidates if candidate is not primary
            )
        broad_readings.sort(
            key=lambda candidate: (
                -broad_gloss_counts[candidate[3].casefold()],
                candidate[1],
                candidate[2],
            )
        )
        additional_readings.sort()
        selected = (broad_readings + additional_readings)[:limit]

        result: list[dict[str, object]] = []
        for (
            _,
            _,
            _,
            definition,
            example,
            translation,
            sense_glosses,
        ) in selected:
            item: dict[str, object] = {
                "definition": definition,
                "example": example,
                "example_translation": translation,
            }
            if sense_glosses:
                item["source_hints"] = {
                    "sense_glosses": list(dict.fromkeys(sense_glosses))
                }
            result.append(item)
        return result

    def lexical_info(self, records: list[FolketsRecord]) -> dict[str, list[str]]:
        """Preserve useful non-gloss Folkets fields without flattening them."""
        attributes = {
            "pronunciations": "pronunciations",
            "variants": "variants",
            "constructions": "constructions",
            "usage_labels": "usage",
            "translation_comments": "translation_comments",
            "explanations": "explanations",
            "synonyms": "synonyms",
            "compounds": "compounds",
            "derivations": "derivations",
            "idioms": "idioms",
        }
        result: dict[str, list[str]] = {}
        for output_key, attribute in attributes.items():
            values = list(
                dict.fromkeys(
                    value
                    for record in records
                    for value in getattr(record, attribute)
                    if value
                )
            )
            if values:
                result[output_key] = values
        return result

    def inflections(
        self, headword: str, expected_pos: str, records: list[FolketsRecord]
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        """Merge agreeing paradigms and report tied conflicts instead of guessing."""
        values: dict[str, list[str]] = defaultdict(list)
        for record in records:
            for key, value in extract_inflections(record, expected_pos).items():
                values[key].append(value)

        result: dict[str, str] = {}
        conflicts: dict[str, list[str]] = {}
        for key, candidates in values.items():
            counts = Counter(candidates)
            ranked = counts.most_common()
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                result[key] = ranked[0][0]
            else:
                conflicts[key] = sorted(counts)

        if expected_pos == "adjective":
            # Folkets stores exceptional plural forms such as små as separate
            # records with an explicit "plural till ..." usage marker.
            linked_plurals = self._plural_for.get(normalize(headword), [])
            if linked_plurals:
                if "plural" in result:
                    result["definite"] = result.pop("plural")
                plural_values = list(
                    dict.fromkeys(record.headword for record in linked_plurals)
                )
                if len(plural_values) == 1:
                    result["plural"] = plural_values[0]
                else:
                    conflicts["plural"] = plural_values

            # Resolve a comparison chain for a headword that is itself a form,
            # e.g. många -> fler -> flest.  This supplements, never overwrites,
            # exact-entry data.
            reverse = self._by_paradigm.get(normalize(headword), [])
            for source in reverse:
                if (
                    source.word_class == "jj"
                    and "komparativ" in " ".join(source.usage).casefold()
                ):
                    result.setdefault("comparative", source.headword)
                    result.setdefault("superlative", headword)
                elif source.word_class == "pn" and len(source.paradigm) >= 2:
                    if normalize(source.paradigm[-1]) == normalize(headword):
                        result.setdefault("positive", source.headword)
                        result.setdefault("comparative", source.paradigm[-2])
                        result.setdefault("superlative", headword)
                    elif normalize(source.paradigm[-2]) == normalize(headword):
                        result.setdefault("positive", source.headword)
                        result.setdefault("comparative", headword)
                        result.setdefault("superlative", source.paradigm[-1])

        return result, conflicts
