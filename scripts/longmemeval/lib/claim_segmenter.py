"""Adaptive claim segmentation for LME ingest.

Two-stage detector that identifies independently durable claims in conversation
turns, avoiding the 8x episode inflation of blind sentence splitting while
preserving extraction quality for knowledge-update scenarios.

Stage A: cheap deterministic gate (regex + heuristics)
Stage B: targeted LLM claim extraction (only for escalated messages)

See: .agent/plans/menhir-adaptive-claim-segmentation-plan.md
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SegmentationMode(str, Enum):
    """Outcome of the segmentation decision."""
    CONTEXT_ONLY = "context_only"       # store as context, skip graph extraction
    EXTRACT_WHOLE = "extract_whole"     # extract the full message as one episode
    SEGMENT_CLAIMS = "segment_claims"   # extract individual claim spans
    SKIP = "skip"                       # do not ingest at all


@dataclass
class ClaimSegment:
    """A durable claim extracted from a longer message."""
    text: str
    subject: str = ""
    claim_type: str = ""        # state_update, new_fact, correction, etc.
    durable: bool = True
    char_start: int = 0
    char_end: int = 0
    segment_index: int = 0


@dataclass
class ClaimRiskSignals:
    """Signals detected by the Stage A deterministic gate."""
    correction_markers: list[str] = field(default_factory=list)
    state_change_verbs: list[str] = field(default_factory=list)
    sentence_count: int = 0
    char_length: int = 0
    has_multiple_subjects: bool = False
    has_topic_shift: bool = False
    has_buried_update: bool = False
    has_multiple_numbers: bool = False
    has_multiple_locations: bool = False

    @property
    def high_confidence_single_claim(self) -> bool:
        """Message likely contains at most one durable claim."""
        return (
            self.sentence_count <= 2
            and not self.correction_markers
            and not self.state_change_verbs
            and not self.has_multiple_subjects
        )

    @property
    def escalate(self) -> bool:
        """Any strong signal suggesting the message needs claim extraction."""
        return bool(
            self.correction_markers
            or self.state_change_verbs
            or self.has_multiple_subjects
            or self.has_topic_shift
            or self.has_buried_update
            or (self.has_multiple_numbers and self.char_length > 200)
            or (self.has_multiple_locations and self.char_length > 200)
        )


# ---------------------------------------------------------------------------
# Stage A: Deterministic gate
# ---------------------------------------------------------------------------

# Correction / update markers — words that signal a fact is being revised.
_CORRECTION_MARKERS = re.compile(
    r"\b(?:actually|instead|no longer|not anymore|now (?:that|I|we|she|he|they|it)|"
    r"used to|again|turns out|changed|anymore|switched|rather than)\b",
    re.IGNORECASE,
)

# State-change verbs — actions that create or revise durable facts.
_STATE_CHANGE_VERBS = re.compile(
    r"\b(?:moved|changed|started|stopped|switched|left|quit|joined|"
    r"bought|sold|married|divorced|graduated|retired|promoted|transferred|"
    r"adopted|resigned|relocated|enrolled|dropped)\b",
    re.IGNORECASE,
)

# Thanks/gratitude, used to demote "again" from correction marker to pleasantry.
_THANKS = re.compile(r"\bthank", re.IGNORECASE)

# Possessive assertions — "my rent is 2400", "my mom uses the same app".
# These carry knowledge updates that no first-person verb pattern catches: the subject is a
# possessed noun, not "I". The verb list is closed on purpose -- matching any inflected word
# after "my ..." would fire on "my question about ...".
_POSSESSIVE_FACT = re.compile(
    r"\bmy\s+(?:[A-Za-z][\w'-]*\s+){0,2}"
    r"(?:is|are|was|were|has|have|had|became|becomes|will\s+be|"
    r"uses|used|likes|liked|prefers|preferred|works|worked|lives|lived|"
    r"goes|went|owns|owned|needs|needed|wants|wanted|takes|took|"
    r"makes|made|says|said|keeps|kept|runs|ran|costs|cost|"
    r"starts|started|stops|stopped|moves|moved|switched|switches)\b",
    re.IGNORECASE,
)

# State-change verbs in their base form, for frames that carry tense elsewhere.
_STATE_CHANGE_BASE = (
    r"(?:move|change|start|stop|switch|leave|quit|join|buy|sell|marry|divorce|"
    r"graduate|retire|transfer|adopt|resign|relocate|enroll|drop)"
)

# Presuppositional wh-questions — "Why did I move to Seattle?" is a question about a move that
# the speaker is asserting happened. The fact is in the presupposition, so the turn is durable
# even though every sentence ends in "?".
#
# Deliberately wh-only. A yes/no question ("Did I move to Seattle?") presupposes nothing -- it
# may be genuinely asking -- so it stays conservative and is not treated as durable.
_PRESUPPOSITIONAL_CHANGE = re.compile(
    r"\b(?:why|when|where|how)\s+(?:did|do|does|has|have|had)\s+"
    r"(?:i|we|he|she|they|my\s+[A-Za-z][\w'-]*)\s+"
    r"(?:[A-Za-z][\w'-]*\s+){0,2}" + _STATE_CHANGE_BASE + r"\w*\b",
    re.IGNORECASE,
)

# Habits and rates — "three times a week", "every other day", "daily". A frequency is a durable
# personal fact and is often the entire content of a knowledge-update turn, with the count
# spelled as a word so no digit pattern sees it.
_FREQUENCY_FACT = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|once|twice|a|an|\d+)\s+"
    r"times?\s+(?:a|an|per|each|every)\s+"
    r"(?:day|week|month|year|morning|evening|night|session|visit)\b"
    r"|\b(?:every|each)\s+(?:other\s+)?(?:day|week|month|year|morning|evening|night)\b"
    r"|\b(?:daily|weekly|monthly|yearly|nightly|hourly)\b",
    re.IGNORECASE,
)

# Quantities — currency, percentages, and counted units. A knowledge update is
# very often just a new number ("my rent is $2,400 now").
_QUANTITY_FACT = re.compile(
    r"(?:\$\s*\d|\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|dollars?|k\b|hours?|minutes?|"
    r"days?|weeks?|months?|years?|miles?|km|kg|lbs?|pounds?)\b)",
    re.IGNORECASE,
)

# Sentence boundary (same as the existing splitter).
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Multiple distinct numbers (dates, amounts, counts).
_NUMBER_PATTERN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:\s*(?:hours?|minutes?|times?|days?|weeks?|months?|years?))?")

# Location-like proper nouns (capitalized multi-word sequences after prepositions).
_LOCATION_PREP = re.compile(
    r"\b(?:in|to|from|at|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
)

# Subject-predicate detection: crude heuristic — count distinct capitalized names
# or first-person references that appear as sentence subjects.
_SUBJECT_PATTERN = re.compile(
    r"(?:^|[.!?]\s+)(?:My\s+(?:friend|sister|brother|mom|dad|wife|husband|partner)\s+)?"
    r"([A-Z][a-z]{2,})",
)
_UNBOUND_SEGMENT_START = re.compile(
    r"^(?:[Hh]e|[Ss]he|[Ii]t|[Tt]hey)\b"
)
_FIRST_PERSON = re.compile(r"\b[Ii]\b")

# Common sentence-starting words that are not proper-noun subjects.
# Shared between detect_claim_risk_signals and _has_extractable_content.
_COMMON_SENTENCE_STARTERS = frozenset({
    "The", "This", "That", "These", "Those", "What", "Where", "When",
    "How", "Why", "Which", "Who", "Here", "There", "Some", "Many",
    "Most", "Any", "Each", "Every", "Both", "All", "One", "Two",
    "Can", "Could", "Would", "Should", "Will", "May", "Might",
    "For", "And", "But", "Also", "Yes", "Yeah", "Sure", "Thanks",
    "Great", "Good", "Got", "Nice", "Sounds", "Let",
    # Fronted auxiliaries. A yes/no question opens with one of these, and reading it as a
    # proper-noun subject made "Did I mention that already?" look like a durable named fact.
    "Did", "Does", "Was", "Were", "Are", "Has", "Have", "Had", "Shall", "Must",
    # Other common non-noun openers.
    "Okay", "Well", "Just", "Please", "Actually", "Maybe", "Perhaps", "Right",
})


def detect_claim_risk_signals(text: str) -> ClaimRiskSignals:
    """Stage A: analyze a message for signals that suggest multiple durable claims."""
    signals = ClaimRiskSignals()
    signals.char_length = len(text)

    # Sentence count
    sentences = _SENTENCE_BOUNDARY.split(text)
    signals.sentence_count = len([s for s in sentences if s.strip()])

    # Correction markers
    signals.correction_markers = [m.group() for m in _CORRECTION_MARKERS.finditer(text)]

    # State-change verbs
    signals.state_change_verbs = [m.group() for m in _STATE_CHANGE_VERBS.finditer(text)]

    # Multiple numbers
    numbers = _NUMBER_PATTERN.findall(text)
    signals.has_multiple_numbers = len(set(numbers)) >= 3

    # Multiple locations
    locations = _LOCATION_PREP.findall(text)
    signals.has_multiple_locations = len(set(locations)) >= 2

    # Multiple subjects — distinct named entities appearing as sentence subjects
    named_subjects = set(_SUBJECT_PATTERN.findall(text))
    # Filter out common non-name words that happen to be capitalized at sentence start
    named_subjects -= _COMMON_SENTENCE_STARTERS
    signals.has_multiple_subjects = len(named_subjects) >= 2

    # Buried update: long message with a correction marker or state-change verb
    # but most of the text is non-factual filler
    if (signals.char_length > 300
            and (signals.correction_markers or signals.state_change_verbs)
            and signals.sentence_count >= 3):
        signals.has_buried_update = True

    # Topic shift: check if distinct locations or named subjects appear in
    # different sentences (crude proxy for semantic distance)
    if signals.sentence_count >= 3:
        sentence_subjects = []
        for sent in sentences:
            subjs = set(_SUBJECT_PATTERN.findall(sent)) - _COMMON_SENTENCE_STARTERS
            locs = set(_LOCATION_PREP.findall(sent))
            sentence_subjects.append(subjs | locs)
        # If non-overlapping subject sets across sentences, likely topic shift
        for i in range(len(sentence_subjects) - 1):
            if sentence_subjects[i] and sentence_subjects[i + 1]:
                if not sentence_subjects[i] & sentence_subjects[i + 1]:
                    signals.has_topic_shift = True
                    break

    return signals


def _has_correction_signal(text: str) -> bool:
    """Correction/update marker that is not a social pleasantry.

    ``again`` is a correction marker in "I moved again" but pure phatic filler in
    "Thanks again for your help!", so it is ignored when the text also thanks.
    """
    for match in _CORRECTION_MARKERS.finditer(text):
        if match.group().lower() == "again" and _THANKS.search(text):
            continue
        return True
    return False


def has_durable_signals(text: str) -> bool:
    """Whether ``text`` carries any signal of a durable, extractable fact.

    This is the single durable-content detector shared by
    :func:`_has_extractable_content` and :func:`is_purely_interrogative`, so a
    turn can never be judged durable by one and phatic by the other.  It looks at
    the whole string — a durable fact is just as durable inside a question
    ("Since I moved to Portland, what should I see?") as inside a statement.

    A miss here silently drops a knowledge update from graph extraction, so the
    signal set deliberately covers the shapes LongMemEval knowledge-update items
    use: first-person durable verbs in declarative clauses, state-change verbs
    (inflected or presupposed by a wh-question), correction markers, named
    sentence subjects, possessive assertions ("my mom uses the same app"),
    frequencies ("three times a week"), and quantities ("$2,400", "three years").
    """
    return bool(
        _has_first_person_durable_assertion(text)
        or _STATE_CHANGE_VERBS.search(text)
        or _PRESUPPOSITIONAL_CHANGE.search(text)
        or _has_correction_signal(text)
        or _POSSESSIVE_FACT.search(text)
        or _FREQUENCY_FACT.search(text)
        or _QUANTITY_FACT.search(text)
        or (set(_SUBJECT_PATTERN.findall(text)) - _COMMON_SENTENCE_STARTERS)
    )


# Vocabulary of fact-free/phatic language. A declarative sentence built ONLY from these words asserts
# nothing durable. Kept deliberately small: this list is a denylist, and anything it does not
# recognise is treated as content.
_PHATIC_TOKENS = frozenset({
    "a", "about", "absolutely", "again", "agreed", "alright", "all", "also", "am", "amazing",
    "and", "any", "anyway", "appreciate", "are", "as", "awesome", "be", "been", "brilliant",
    "but", "cheers", "cool", "definitely", "do", "does", "enough", "exactly", "excellent",
    "fair", "fantastic", "fine", "for", "get", "glad", "good", "got", "great", "haha", "have",
    "hear", "hello", "help", "helpful", "hey", "hi", "hmm", "how", "i", "idea", "if", "in",
    "indeed", "interesting", "is", "it", "just", "know", "let", "like", "lol", "lot", "love",
    "makes", "many", "me", "much", "my", "neat", "nice", "no", "noted", "of", "ofcourse", "oh",
    "ok", "okay", "one", "perfect", "please", "point", "really", "right", "same", "see",
    "sense", "so", "sorry", "sound", "sounds", "sure", "team", "thank", "thanks", "that",
    "the", "then", "there", "these", "think", "this", "those", "thx", "to", "too", "true",
    "try", "understood", "very", "well", "what", "when", "will", "wonderful", "work", "works",
    "wow", "yeah", "yep", "yes", "you", "your", "yup", "that's",
})

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_ANY_DIGIT = re.compile(r"\d")

# The longest a sentence can be and still be *provably* reactive commentary. Past this, the
# claim "there is nothing durable in here" stops being something a heuristic can support.
_MAX_PHATIC_WORDS = 8


def is_phatic_sentence(sentence: str) -> bool:
    """Whether a declarative sentence is *provably* fact-free/phatic.

    This is a denylist, and that direction is the whole design. The previous version asked the
    opposite question -- "can I recognise a durable fact in here?" -- which requires an
    exhaustive allowlist of every shape a fact can take. It cannot exist: an audit of the frozen
    78-item fixture found 28 user turns carrying answers that no pattern matched, so they were
    routed to CONTEXT_ONLY and silently never extracted. A missing fact looks exactly like a
    model that had nothing to say, and nothing in the pipeline reports it.

    So an unrecognised statement fails OPEN, to extraction. If extraction then collapses to zero
    edges the episode FAILS and the strict gate stops the build -- loud, attributable, and
    fixable. Silent omission is the worse of the two failures.
    """
    stripped = sentence.strip()
    if not stripped:
        return True
    # A durable signal settles it: never phatic.
    if has_durable_signals(stripped):
        return False
    words = _WORD.findall(stripped)
    if not words:
        return True
    # A number is a value, and a value is not fact-free ("I've completed 30 videos").
    if _ANY_DIGIT.search(stripped):
        return False
    # A named thing anywhere but the sentence opener is content, whatever surrounds it
    # ("my Canon EOS 80D on five trips"). "I" is not a name.
    if any(word[0].isupper() and word != "I" for word in words[1:]):
        return False
    lowered = [word.lower() for word in words]
    if all(word in _PHATIC_TOKENS for word in lowered):
        return True
    # Reactive commentary about the topic ("I think that sounds great"), capped short so a fact
    # cannot ride along inside it.
    return bool(_FIRST_PERSON_REACTIVE.search(stripped)) and len(words) <= _MAX_PHATIC_WORDS


# Clause boundaries. A sentence that ends in "?" routinely opens with a declarative clause --
# "I'm planning to go to the mall this weekend, can you remind me what's near H&M?" -- and the
# statement half is the knowledge update. Sentence-level analysis cannot see it.
_CLAUSE_BOUNDARY = re.compile(r"\s*(?:[,;:]|--|—)\s*|\s+(?:and|but|so|then|though|although)\s+")

# A clause that is asking, not telling. Wh-word or inverted auxiliary at the clause opening.
_QUESTION_CLAUSE = re.compile(
    r"^(?:and|but|so|or|also)?\s*"
    r"(?:what|when|where|why|how|which|who|whom|whose|"
    r"can|could|would|should|will|shall|may|might|must|"
    r"do|does|did|is|are|was|were|have|has|had|any|anyone|anybody)\b",
    re.IGNORECASE,
)

# First-person or possessive subject. Deliberately broad -- "did the user talk about themselves"
# is a question a short closed list can answer, unlike "is this a durable fact".
_FIRST_PERSON_SUBJECT = re.compile(
    r"\b(?:I|I'm|I've|I'll|I'd|we|we're|we've|we'll|we'd|my|mine|our|ours)\b",
    re.IGNORECASE,
)

# The narrow exception: first-person framing that reacts to the conversation instead of
# reporting anything about the speaker's life. Anchored at the clause opening, so "I think that
# sounds great" is reactive while "I'm thinking of switching plans" is a plan.
_REACTIVE_CLAUSE = re.compile(
    r"^(?:and|but|so|also|well|ok|okay|yeah|yes|sure)?\s*"
    r"(?:I\s+(?:agree|see|understand|appreciate)\b(?:\s+(?:that|this|it))?\s*[.!]?$"
    r"|I\s+(?:think|thought|suppose|guess|bet|mean|imagine|assume|figured|believe)\s+"
    r"(?:that|this|it|so|the\s+(?:idea|plan|approach)|.*\bsounds?\b)"
    r"|I\s+was\s+wondering\b"
    r"|I'?m\s+(?:curious|wondering|not\s+sure|glad|sorry|excited\s+to\s+hear)\b"
    r"|I'?d\s+(?:love|like)\s+to\s+(?:know|hear)\b"
    r"|I\s+see\b)",
    re.IGNORECASE,
)


def _clauses(sentence: str) -> list[str]:
    return [clause.strip() for clause in _CLAUSE_BOUNDARY.split(sentence) if clause.strip()]


def _has_first_person_durable_assertion(text: str) -> bool:
    """Whether a declarative clause contains a first-person durable verb.

    The durable-verb regex is intentionally broad, but applying it to a complete
    question mistakes embedded phrases such as "the pair I need?" for knowledge
    updates. Check the same clause boundaries used by
    :func:`has_first_person_declarative` and ignore clauses that ask rather than
    tell.
    """
    for sentence in _SENTENCE_BOUNDARY.split(text):
        for clause in _clauses(sentence):
            if clause.rstrip().endswith("?") or _QUESTION_CLAUSE.match(clause):
                continue
            if _FIRST_PERSON_DURABLE.search(clause):
                return True
    return False


def has_first_person_declarative(text: str) -> bool:
    """Whether any clause is the speaker telling us something about themselves.

    This is the load-bearing fail-open rule. A first-person or possessive declarative clause --
    "I'm planning a trip", "I've been experimenting with noodles", "I'm feeling overwhelmed",
    "I've been using my Starbucks Rewards app" -- is a statement about the user's own life. None
    of those match any fact template, and under a template-based rule all of them were discarded.
    Whether such a clause is *durable* is a judgement no regex should be making, so it is handed
    to extraction, which is the component that actually reads.

    Only two things are excluded: clauses that ask rather than tell, and the narrow set of
    first-person framings that react to the conversation ("I think that sounds great").
    """
    for sentence in _SENTENCE_BOUNDARY.split(text):
        for clause in _clauses(sentence):
            if clause.rstrip().endswith("?") or _QUESTION_CLAUSE.match(clause):
                continue
            if not _FIRST_PERSON_SUBJECT.search(clause):
                continue
            if _REACTIVE_CLAUSE.match(clause) and is_phatic_sentence(clause):
                continue
            return True
    return False


def is_context_only_user_turn(text: str) -> bool:
    """Whether a user turn can be proven to be fact-free.

    Three ways to fail that proof, in order of how much they catch:

    1. a durable signal anywhere (a presupposition can span the sentence split);
    2. any first-person/possessive declarative clause -- the speaker talking about themselves;
    3. any declarative sentence that is not provably phatic.

    Only a turn that survives all three is evidence-only. In practice what remains is fact-free
    question frames ("What are some good resources for X?") and acknowledgments ("Thanks
    again!"), which is exactly the population that extracts to ``{"name":"user"}`` with zero
    edges.
    """
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not sentences:
        return True
    if has_durable_signals(text):
        return False
    if has_first_person_declarative(text):
        return False
    declaratives = [s for s in sentences if not s.rstrip().endswith("?")]
    if not declaratives:
        # Nothing but questions, and nothing durable presupposed by them.
        return is_purely_interrogative(text)
    return all(is_phatic_sentence(sentence) for sentence in declaratives)


def is_short_and_coherent(text: str, max_chars: int = 300, max_sentences: int = 3) -> bool:
    """Quick check: message is short enough to always extract as a whole."""
    if len(text) > max_chars:
        return False
    sentences = _SENTENCE_BOUNDARY.split(text)
    return len([s for s in sentences if s.strip()]) <= max_sentences


# First-person durable-fact patterns — statements that assert personal state,
# preferences, plans, or experiences (as opposed to reactions about the topic).
_FIRST_PERSON_DURABLE = re.compile(
    r"\bI\s+(?:live|work|have|own|prefer|like|love|want|need|plan|moved|"
    r"started|stopped|quit|joined|left|bought|sold|got|adopted|enrolled|"
    r"graduated|retired|completed|wrote|finished|built|created|made|"
    r"recently|just|finally|decided|chose|picked|applied|signed up|"
    r"scheduled|booked|reserved|ordered)\b",
    re.IGNORECASE,
)

# Reactive / non-durable first-person patterns — opinions about the conversation
# topic rather than personal facts.
_FIRST_PERSON_REACTIVE = re.compile(
    r"\bI\s+(?:think|agree|see|understand|suppose|imagine|guess|feel like|"
    r"like (?:the |that |this |it)|wonder|hope|bet|mean|"
    r"appreciate|enjoy (?:the |that |this ))\b",
    re.IGNORECASE,
)


def is_purely_interrogative(text: str) -> bool:
    """Check if a user message is purely asking questions or reacting without personal facts.

    Returns True for turns like "What do you think about X?" or "I think that sounds
    great. What kind of elements were you thinking of?" — turns where the user is
    engaging with the topic but not stating any durable personal fact.

    Durable signals are scanned across *every* sentence, questions included. A
    question is a perfectly ordinary carrier for a knowledge update — "Since I
    moved to Portland, which neighborhoods should I check out?" states the move —
    so an all-questions turn is only interrogative when nothing in it is durable.
    """
    sentences = _SENTENCE_BOUNDARY.split(text)
    real = [s.strip() for s in sentences if s.strip()]
    if not real:
        return True

    # Must have at least one question
    if not any(s.rstrip().endswith("?") for s in real):
        return False

    # Any durable fact anywhere — including inside a question — makes this turn
    # worth extracting on its own rather than being recorded as context-only evidence.
    return not has_durable_signals(text)


# ---------------------------------------------------------------------------
# Assistant turn classification
# ---------------------------------------------------------------------------

# Patterns that suggest an assistant message contains durable content (not generic advice).
#
# Every alternative here must bind the claim to a *user-owned entity*, because the failure
# mode measured on the KU fixture was advice prose matching on incidental vocabulary. The
# earlier version of this pattern ended with a bare alternation of past participles --
# (confirmed|completed|finished|done|approved|accepted|rejected|denied|cancelled) -- which
# matched 51 of 58 assistant turns on nothing more than a word appearing anywhere in the
# message: "tasks that need to be completed", "you've already done 2,000 miles on your
# current wheels", "they'll be more likely to be accepted by the thrift store". Unanchored
# participles have no subject, so they cannot establish who did what to whose records. That
# alternation is deleted rather than narrowed; there is no defensible true-positive form of
# it that the anchored alternatives below do not already cover.
#
# Measured on fixtures/longmemeval/knowledge_update_subset.json (915 assistant turns):
# 58 matches before, all false positives; 0 after. The KU corpus has no tool-using
# assistant, so the transactional forms kept below have no validated true positive *here* --
# they are retained because removing a general-purpose pattern for not firing on one
# conversational fixture is overfitting, which this file's plan explicitly warned against.
_ASSISTANT_DURABLE_PATTERNS = re.compile(
    # A completed action on something the user owns. The trailing possessive is load-bearing:
    # without it, "I've created a suggested schedule" and "I've created a 20-mile route" match,
    # which is the assistant generating advice content, not recording a fact about the user.
    r"\bI(?:'ve| have) (?:scheduled|booked|reserved|ordered|created|set up|"
    r"configured|deployed|committed|pushed|merged|installed) your\b"
    r"|"
    # A user record reported in a terminal state. The state word is load-bearing: a bare
    # copula matched instructional prose -- "Ensure your profile is complete, including a
    # profile picture" and "make sure your account is set to a Business or Creator profile".
    #
    # `completed` is listed but bare `complete` is not, and the distinction is the difference
    # between an action that finished and a description of fullness. "Ensure your profile is
    # complete, including a profile picture" is advice about filling in fields and was still
    # matching after the first pass at this pattern. The cost is missing "your booking is
    # complete"; precision is the stated priority here, so that recall loss is accepted.
    r"\b(?:your|the) (?:appointment|reservation|booking|order|subscription|"
    r"membership|account|profile|settings?) (?:is|are|has been|was) "
    r"(?:confirmed|booked|scheduled|reserved|cancelled|canceled|active|ready|"
    r"processed|completed|approved|rejected|denied|suspended|closed)\b"
    r"|"
    # An explicit commitment to remember. No match on the KU fixture; kept as a general
    # agentic form rather than pruned for fixture silence.
    r"\bI(?:'ll| will) (?:remember|note|keep track|make a note)\b",
    re.IGNORECASE,
)


def is_memory_bearing_assistant_output(text: str) -> bool:
    """Check if an assistant message contains durable-content classes worth extracting."""
    return bool(_ASSISTANT_DURABLE_PATTERNS.search(text))


# ---------------------------------------------------------------------------
# Top-level decision function
# ---------------------------------------------------------------------------

def segmentation_mode(
    role: str,
    content: str,
    *,
    max_short_chars: int = 300,
    max_short_sentences: int = 3,
) -> SegmentationMode:
    """Determine how a conversation turn should be processed for graph extraction.

    Returns one of:
        CONTEXT_ONLY   — store as context, skip extraction
        EXTRACT_WHOLE  — extract the full message as one episode
        SEGMENT_CLAIMS — run Stage B to extract individual claim spans
        SKIP           — do not ingest
    """
    if not content or not content.strip():
        return SegmentationMode.SKIP

    # Assistant messages: context-only by default unless they contain durable content
    if role.strip().lower() == "assistant":
        if is_memory_bearing_assistant_output(content):
            return SegmentationMode.EXTRACT_WHOLE
        return SegmentationMode.CONTEXT_ONLY

    # User messages: route to CONTEXT_ONLY only when the turn can be proven fact-free/phatic.
    # Such a turn extracts to just {"name":"user"} with zero edges, which fails the episode; it
    # is instead recorded as evidence and never sent to extraction. Anything not provably phatic
    # is extracted, because a fact dropped here is dropped silently.
    if is_context_only_user_turn(content):
        return SegmentationMode.CONTEXT_ONLY

    if is_short_and_coherent(content, max_short_chars, max_short_sentences):
        return SegmentationMode.EXTRACT_WHOLE

    signals = detect_claim_risk_signals(content)

    if signals.high_confidence_single_claim:
        return SegmentationMode.EXTRACT_WHOLE

    if signals.escalate:
        return SegmentationMode.SEGMENT_CLAIMS

    return SegmentationMode.EXTRACT_WHOLE


# ---------------------------------------------------------------------------
# Stage B: LLM-based claim segmenter
# ---------------------------------------------------------------------------

_SEGMENTER_PROMPT = """\
You are a memory claim extractor. Given a conversation message, identify independently \
durable factual claims — facts that should be remembered long-term.

Rules:
- Extract ONLY factual claims about people, places, events, preferences, states, or changes.
- Do NOT extract questions, requests, opinions about hypotheticals, or generic advice.
- Do NOT extract travel planning, weather discussion, or other transient conversation.
- Each claim should be a self-contained statement that makes sense without the surrounding context.
- If a claim updates or contradicts a previous fact, mark claim_type as "state_update".
- If no durable claims exist, return an empty segments array.
- Preserve the original wording as closely as possible.

Message role: {role}
Message:
{content}

Respond with ONLY a valid JSON object:
{{"segments": [{{"text": "...", "subject": "...", "claim_type": "new_fact|state_update|preference|event", "durable": true}}]}}"""


async def extract_claim_segments(
    content: str,
    role: str = "user",
    *,
    llm_client: Any = None,
    model: str | None = None,
) -> list[ClaimSegment]:
    """Stage B: use a small LLM call to extract durable claim spans.

    If no LLM client is provided, falls back to a simple heuristic extraction
    that pulls sentences containing correction markers or state-change verbs.
    """
    if llm_client is not None:
        return await _llm_extract_claims(content, role, llm_client, model)
    return _heuristic_extract_claims(content, role)


def _heuristic_extract_claims(content: str, role: str) -> list[ClaimSegment]:
    """Fallback: extract sentences that contain state-change or correction signals."""
    sentences = _SENTENCE_BOUNDARY.split(content)
    segments = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue
        has_correction = bool(_CORRECTION_MARKERS.search(sent))
        has_state_change = bool(_STATE_CHANGE_VERBS.search(sent))
        if has_correction or has_state_change:
            # A standalone claim beginning with an anaphoric subject loses its antecedent.
            # The full source turn is already ingested by the adaptive caller, so keep the
            # fact there instead of submitting a duplicate that cannot be grounded alone.
            if _UNBOUND_SEGMENT_START.match(sent):
                continue
            claim_type = "state_update" if has_correction else "new_fact"
            # Try to identify the subject
            subj_match = _SUBJECT_PATTERN.search(sent)
            subject = subj_match.group(1) if subj_match else ""
            start = content.find(sent)
            segments.append(ClaimSegment(
                text=sent,
                subject=subject,
                claim_type=claim_type,
                durable=True,
                char_start=start if start >= 0 else 0,
                char_end=(start + len(sent)) if start >= 0 else len(sent),
                segment_index=i,
            ))
    return segments


async def _llm_extract_claims(
    content: str,
    role: str,
    llm_client: Any,
    model: str | None,
) -> list[ClaimSegment]:
    """Use an OpenAI-compatible client to extract claims."""
    import json as _json

    prompt = _SEGMENTER_PROMPT.format(role=role, content=content)
    _model = model or os.getenv("LME_EXTRACT_MODEL", "gpt-4o-mini")

    try:
        resp = await llm_client.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = _json.loads(raw)
        segments = []
        for i, seg in enumerate(data.get("segments", [])):
            text = seg.get("text", "").strip()
            if not text:
                continue
            start = content.find(text)
            segments.append(ClaimSegment(
                text=text,
                subject=seg.get("subject", ""),
                claim_type=seg.get("claim_type", "new_fact"),
                durable=seg.get("durable", True),
                char_start=start if start >= 0 else 0,
                char_end=(start + len(text)) if start >= 0 else len(text),
                segment_index=i,
            ))
        return segments
    except Exception as exc:
        logger.warning("LLM claim extraction failed: %s; falling back to heuristic", exc)
        return _heuristic_extract_claims(content, role)
