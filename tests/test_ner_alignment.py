"""Offsets stay source-relative even when Obeliks discards control characters."""

from types import SimpleNamespace

import pytest

from semora.ner.adapters import align_classla_tokens


def document(*words):
    class Token:
        def __init__(self, text):
            self.text = text

        @property
        def end_char(self):
            return self._end_char

    tokens = [Token(word) for word in words]
    return SimpleNamespace(iter_tokens=lambda: iter(tokens))


def test_discarded_controls_preserve_original_offsets():
    gap = ''.join(chr(code) for code in [*range(32), *range(127, 160)])
    text = gap + 'Ana' + gap + '!' + gap
    doc = document('Ana', '!')
    align_classla_tokens(doc, text)
    tokens = list(doc.iter_tokens())
    assert [(t._start_char, t._end_char) for t in tokens] == [
        (len(gap), len(gap) + 3), (2 * len(gap) + 3, 2 * len(gap) + 4),
    ]
    assert all(text[t._start_char:t._end_char] == t.text for t in tokens)
    align_classla_tokens(document(), gap)


@pytest.mark.parametrize('missing', ['word', '!', '-', '\u200b'])
def test_does_not_allow_arbitrary_omitted_text(missing):
    with pytest.raises(ValueError, match='Cannot align'):
        align_classla_tokens(document('Ana', 'Novak'), 'Ana\x01' + missing + ' Novak')
    with pytest.raises(ValueError, match='omitted text'):
        align_classla_tokens(document('Ana'), 'Ana\x01' + missing)


def test_real_obeliks_discarded_controls():
    pytest.importorskip('classla')
    from classla.models.common.doc import Document
    from classla.utils.obeliks import ObeliksTrainer

    text = '-\b\x01\v\f\b\n \x0e\x0f\x10\x01\n\nAna Novak!\x01'
    raw, sentences, metadata = ObeliksTrainer.tokenize(text)
    doc = Document(sentences, raw, metasentences=metadata)
    align_classla_tokens(doc, text)
    tokens = list(doc.iter_tokens())
    assert [t.text for t in tokens] == ['-', 'Ana', 'Novak', '!']
    assert all(text[t.start_char:t.end_char] == t.text for t in tokens)


def test_controls_inside_token_preserve_span_and_do_not_match_later_occurrence():
    text = 'J\x03. Novak J.'
    doc = document('J.', 'Novak', 'J.')
    align_classla_tokens(doc, text)
    assert [(t._start_char, t._end_char) for t in doc.iter_tokens()] == [(0, 3), (4, 9), (10, 12)]


@pytest.mark.parametrize('gap', [' ', '\n', 'x', '!', '\u200b'])
def test_does_not_merge_across_whitespace_or_visible_text(gap):
    with pytest.raises(ValueError, match='Cannot align'):
        align_classla_tokens(document('J.'), 'J' + gap + '.')


def test_real_obeliks_entity_span_includes_internal_controls():
    pytest.importorskip('classla')
    from classla.models.common.doc import Document
    from classla.utils.obeliks import ObeliksTrainer

    from semora.ner.base import Entity, validate

    text = 'J\x03. Novak'
    raw, sentences, metadata = ObeliksTrainer.tokenize(text)
    doc = Document(sentences, raw, metasentences=metadata)
    align_classla_tokens(doc, text)
    tokens = list(doc.iter_tokens())
    assert [t.text for t in tokens] == ['J.', 'Novak']
    for token, label in zip(tokens, ['B-PER', 'I-PER'], strict=True):
        token.ner = label
    entities = [Entity(e.start_char, e.end_char, e.type, e.text) for e in doc.build_ents()]
    assert entities == [Entity(0, len(text), 'PER', text)]
    validate(text, entities)


@pytest.mark.parametrize(('text', 'words', 'spans'), [
    ('\xad -', ['-', '-'], [(0, 1), (2, 3)]),
    ('Ana\xadMarija', ['Ana-Marija'], [(0, 10)]),
    ('A\x03\xadB', ['A-B'], [(0, 4)]),
])
def test_soft_hyphen_normalization_keeps_original_offsets(text, words, spans):
    doc = document(*words)
    align_classla_tokens(doc, text)
    assert [(t._start_char, t._end_char) for t in doc.iter_tokens()] == spans


@pytest.mark.parametrize('char', ['\u2013', '\u2014', '\u2212'])
def test_other_dash_characters_are_not_silently_normalized(char):
    with pytest.raises(ValueError, match='Cannot align'):
        align_classla_tokens(document('-'), char)


def test_real_obeliks_soft_hyphen_entity_preserves_original_text():
    pytest.importorskip('classla')
    from classla.models.common.doc import Document
    from classla.utils.obeliks import ObeliksTrainer

    from semora.ner.base import Entity, validate

    text = 'Ana\xadMarija'
    raw, sentences, metadata = ObeliksTrainer.tokenize(text)
    doc = Document(sentences, raw, metasentences=metadata)
    align_classla_tokens(doc, text)
    tokens = list(doc.iter_tokens())
    assert '-' in ''.join(t.text for t in tokens)
    for i, token in enumerate(tokens):
        token.ner = 'B-PER' if i == 0 else 'I-PER'
    entities = [Entity(e.start_char, e.end_char, e.type, e.text) for e in doc.build_ents()]
    assert entities == [Entity(0, len(text), 'PER', text)]
    validate(text, entities)
