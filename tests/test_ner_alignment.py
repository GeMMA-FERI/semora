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
