"""Shared lexical quality contract for both LLM enrichment backends."""

from __future__ import annotations

import re
import sys
from typing import Any

from functions.cefr import CEFR_LEVELS

MAX_NOTE_CHARS = 180
MAX_DEFINITIONS = 2

INFLECTIONS_BY_POS = {
    "noun": ("plural",),
    "verb": ("present", "past", "supine", "imperative", "participle"),
    "adjective": ("plural", "comparative", "superlative"),
}

ENRICHMENT_QUALITY_POLICY = f"""Gemensam kvalitetsstandard för svenskt Anki-innehåll:

Betydelser:
* Skriv korta, naturliga engelska glosor för den svenska uppslagsformen och exakt den aktuella betydelsen.
* Respektera hela uppslagsformen, inklusive artikel, partikel, reflexivt pronomen och preposition. Tolka inte ett lexikaliserat flerordsuttryck som summan av de enskilda orden.
* Ge en andra betydelse endast när den är vanlig, modern och tydligt åtskild. Lägg inte till en sällsynt, ålderdomlig, regional eller fackspråklig betydelse bara för att nå två.

Exempel:
* Skriv naturlig, idiomatisk, modern standardsvenska som tydligt visar den angivna betydelsen.
* Prioritera korta, återanvändbara meningar för en vuxen på SFI D/SVA-nivå: vardagsliv, arbete, studier och samhällsliv i Sverige.
* Föredra vanliga situationer, kollokationer och konstruktioner framför abstrakta, litterära eller konstruerade exempel.
* Behåll hela partikelverb, reflexiva former, flerordsuttryck och styrda prepositioner. Använd uppslagsformen exakt när det är grammatiskt möjligt; gör annars minsta nödvändiga böjning.
* Översätt hela meningen troget och naturligt till engelska. Översätt den aktuella betydelsen, inte en annan möjlig betydelse av ordet. Bevara särskilt singular/plural, bestämdhet, tempus, person och negation; välj engelska ordformer efter den faktiska svenska meningen, inte efter en vanemässig kollokation.
* Variera subjekt, tempus och situation mellan posterna. Börja inte alla meningar med ”Jag”.

Valfri användningsnot (fältet note):
* Noten ska vara på engelska, höra till en enda betydelse och ge information som en kort glosa och exempelmening inte förmedlar tillräckligt säkert.
* Skriv en note endast när den hjälper eleven att undvika ett sannolikt missförstånd eller faktiskt använda uttrycket rätt. Tillåtna skäl är: (1) fast eller lexikaliserat uttryck vars betydelse inte är kompositionell, (2) viktig stilnivå, artighet, attityd, laddning, regionalitet eller ålderdomlighet, (3) nödvändig svensk kulturell eller institutionell referens, (4) obligatorisk eller starkt föredragen konstruktion, kollokation, partikel eller preposition, eller (5) kort kontrast mot en falsk vän eller lättförväxlad form där förväxlingen kvarstår även efter glosan och exempelmeningen.
* Exempelmeningen är den viktigaste avgränsaren. Om glosan och exempelmeningen redan gör betydelsen entydig för en engelsktalande elev ska du returnera null. Att ett engelskt ord råkar se likadant ut är i sig ingen anledning till en note; jämför glosan med exemplet och fråga dig om eleven fortfarande kan välja fel betydelse. Typiska fall där null räcker: glass (glosan visar ”ice cream” och exemplet gör betydelsen entydig), prick (pricken i fjärran), slut (filmens slut), fart (bilens fart).
* Kärnbetydelsen måste stå i meaning/definition. Använd aldrig noten för att reparera en felaktig eller vag glosa. Noten måste precisera det aktuella svenska ordet, den aktuella betydelsen eller en konstruktion som faktiskt visas i exemplet.
* En bred engelsk glosa är inte belägg för en snävare faktauppgift. Påståenden om till exempel släktskapsrelation, institution, register eller kulturell avgränsning måste uttryckligen stödjas av source_hints; annars ska note vara null.
* Upprepa inte betydelsen, översättningen, ordklassen eller böjningslistan. Undvik etymologi, kuriosa, ovanliga sidobetydelser och generiska kommentarer som ”used in Swedish”.
* Använd vanlig text utan Markdown, HTML, minnesregler eller instruktioner till eleven.
* Skriv normalt högst 25 ord och alltid högst {MAX_NOTE_CHARS} tecken i en enda kort mening. Returnera null när ingen note är klart motiverad. En tom sträng är inte tillåten.

Böjning:
* Använd endast moderna standardsvenska former. Substantivets plural är obestämd nominativ plural. Verbformerna är aktiv indikativ presens och preteritum, supinum, imperativ och presens particip. Adjektivets plural är positiv obestämd plural; komparativ och superlativ anges endast när adjektivet är graderbart.
* Använd null endast när den efterfrågade formen verkligen saknas. Hitta inte på ålderdomliga eller mycket ovanliga former.

source_hints, när det finns, är oföränderlig lokal källkontext från Folkets eller Kelly (varianter, konstruktioner, bruk, kommentarer, förklaringar, synonymer, avledningar, idiom eller Kellys uppslagsordsanteckningar och användningsmönster). sense_glosses är en kort, betydelsespecifik ordboksförklaring: använd den för att avgränsa den bredare glosan och motsäg den inte. Kellys usage pattern är inte en färdig exempelmening: använd det bara som betydelse- eller konstruktionsstöd och skriv en fullständig mening när sentence efterfrågas. Använd bara relevanta ledtrådar för den aktuella betydelsen. I konstruktioner står & för uppslagsformen, till exempel betyder "A & på x" att verbet styr prepositionen på. Kopiera inte ogenomskinlig ordboksnotation ordagrant till elevtext och hitta inte på information när ledtrådarna är tvetydiga.

Befintliga värden är källdata: ersätt, radera eller omformulera dem aldrig. Returnera endast efterfrågade värden. Kontrollera tyst betydelse, idiomatik, grammatik, partikel/preposition, register och praktisk användbarhet innan du svarar.
"""


# Source markers that justify spending a standalone usage-note request.
_NOTE_SIGNAL_KEYS = (
    "idioms",
    "usage_labels",
    "translation_comments",
    "explanations",
    "kelly_hints",
)
# Folkets valency notation: x/y/A/B stand for argument slots, and & for the
# headword. These tokens are structural placeholders rather than linguistic
# information, so a pattern made only of them (plain transitivity) is not worth
# a usage note. Complement-type labels such as INF and SATS are deliberately
# absent: they record a real complementation choice.
_STRUCTURAL_TOKENS = frozenset(
    {
        "x",
        "y",
        "z",
        "a",
        "b",
        "c",
        "n",
        "v",
        "adj",
        "obj",
        "subj",
        "plats",
        "tid",
        "riktning",
        "att",
    }
)


def informative_construction(construction: str) -> bool:
    """True when a Folkets valency pattern carries more than plain transitivity.

    ``A & x`` only says the verb is transitive, which never needs a usage note.
    ``A & på x``, ``A & sig + INF``, or ``A & x/att + SATS`` encode a required
    preposition, particle, reflexive, or complement type, which is exactly the
    kind of construction a note should surface.
    """
    tokens = [token for token in re.split(r"[^A-Za-zÅÄÖåäö]+", construction) if token]
    return any(token.casefold() not in _STRUCTURAL_TOKENS for token in tokens)


def note_warrants_assessment(entry: dict[str, Any]) -> bool:
    """Select senses where a usage note is plausible from source evidence.

    Most sanctioned note reasons leave a Folkets trace: idioms
    (non-compositional), usage labels (register or plural-only), translation
    comments (nuanced sense), explanations (cultural reference), and informative
    constructions (required preposition, particle, or reflexive). This is only
    a cost-saving heuristic: source metadata is incomplete and cannot replace a
    model judgment over the full word, gloss, and example context.
    """
    lexical_info = entry.get("lexical_info")
    if not isinstance(lexical_info, dict):
        return False
    if any(lexical_info.get(key) for key in _NOTE_SIGNAL_KEYS):
        return True
    constructions = lexical_info.get("constructions")
    return isinstance(constructions, list) and any(
        informative_construction(str(value))
        for value in constructions
        if str(value).strip()
    )


def learner_order(entry: dict[str, Any]) -> tuple[int, int]:
    """Order work by reviewed learner level, then the original Kelly rank."""
    learner_level = entry.get("learner_level") or {}
    level = str(learner_level.get("cefr") or "").upper()
    if level not in CEFR_LEVELS:
        level = str(
            entry.get("sources", {}).get("frequency", {}).get("cefr_band") or ""
        ).upper()
    level_index = CEFR_LEVELS.index(level) if level in CEFR_LEVELS else len(CEFR_LEVELS)
    rank = entry.get("frequency_rank")
    return level_index, rank if isinstance(rank, int) and rank > 0 else sys.maxsize


def model_source_hints(
    entry: dict[str, Any], *, purpose: str = "examples"
) -> dict[str, list[str]]:
    """Select purpose-specific source hints without wasting prompt tokens."""
    lexical_info = entry.get("lexical_info")
    if not isinstance(lexical_info, dict):
        return {}
    limits = {
        "variants": 3,
        "constructions": 3,
        "usage_labels": 3,
        "translation_comments": 3,
        "kelly_hints": 3,
    }
    if purpose == "definitions":
        # Long Swedish prose and near-synonyms only help sense disambiguation.
        limits.update({"explanations": 1, "synonyms": 3, "derivations": 3})
    elif purpose == "notes":
        # These are strong note signals, but they are entry-level rather than
        # sense-aligned. Keep the sample small and let the model reject hints
        # that do not apply to the requested meaning.
        limits.update({"idioms": 2, "explanations": 1})
    result: dict[str, list[str]] = {}
    for key, limit in limits.items():
        values = lexical_info.get(key)
        if not isinstance(values, list):
            continue
        selected = [
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        ][:limit]
        if selected:
            result[key] = selected
    return result


def model_sense_hints(definition: object) -> dict[str, list[str]]:
    """Return one short, sense-aligned source gloss for model grounding."""
    if not isinstance(definition, dict):
        return {}
    source_hints = definition.get("source_hints")
    if not isinstance(source_hints, dict):
        return {}
    values = source_hints.get("sense_glosses")
    if not isinstance(values, list):
        return {}
    selected = [
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip() and len(value.strip()) <= 180
    ][:1]
    return {"sense_glosses": selected} if selected else {}


def normalize_optional_note(value: object, label: str = "note") -> str | None:
    """Validate and normalize a model-generated optional usage note."""
    if value is None:
        return None
    if not isinstance(value, str):
        # Model-output validation uses one ValueError contract for retries.
        raise ValueError(f"{label} must be a string or null")  # noqa: TRY004
    note = value.strip()
    if not note:
        raise ValueError(f"{label} must be null rather than an empty string")
    if "\n" in note:
        raise ValueError(f"{label} must be a single line")
    if len(note) > MAX_NOTE_CHARS:
        raise ValueError(f"{label} exceeds {MAX_NOTE_CHARS} characters")
    return note
