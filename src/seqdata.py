from dataclasses import InitVar, dataclass, field
from functools import singledispatch
from typing import Iterator, Optional, Union

import numpy as np
from cogent3 import get_moltype
from cogent3.core.alphabet import CharAlphabet
from cogent3.core.moltype import MolType
from cogent3.core.sequence import SeqView

T = Union[str, bytes, np.ndarray]


@singledispatch
def seq_index(seq: T, alphabet: CharAlphabet):
    raise NotImplementedError(
        f"{seq_index.__name__} not implemented for type {type(seq)}"
    )


@seq_index.register
def _(seq: str | bytes, alphabet: CharAlphabet) -> np.ndarray:
    return alphabet.to_indices(seq)


@seq_index.register
def _(seq: np.ndarray, alphabet: CharAlphabet) -> np.ndarray:
    # TODO: should return itself
    return alphabet.from_indices(seq)


@dataclass(slots=True)
class SeqData:
    _data: dict[str, np.ndarray] = field(init=False)
    _moltype: MolType = field(init=False)
    _name_order: tuple[str] = field(init=False)
    _alpha: CharAlphabet = field(init=False)
    data: InitVar[dict[str, str]]
    moltype: InitVar[str | None] = "dna"
    name_order: InitVar[tuple[str] | None] = None

    def __post_init__(self, data, moltype, name_order):
        self._moltype = get_moltype(moltype)
        self._alpha = self._moltype.alphabets.degen_gapped
        self._name_order = name_order or tuple(data.keys())
        # When SeqData is initialised, sequence strings are converted to moltype alphabet indicies
        self._data = {k: seq_index(v, self._alpha) for k, v in data.items()}

    def get_seq_array(self, *, seqid: str, start: int = None, stop: int = None):
        # TODO: Let get_seq_str deal with indices
        pass

    def get_seq_str(
        self, *, seqid: str, start: int = None, stop: int = None
    ) -> tuple[str]:
        return self._alpha.from_indices(self._data[seqid][start:stop])

    def iter_seqs_str(self, *, name_order: list[str] = None) -> Iterator:
        name_order = name_order or self._name_order
        yield from (self.get_seq_str(seqid=n) for n in name_order)

    def iter_names(self, *, name_order: list[str] = None) -> Iterator:
        yield from iter(name_order or self._name_order)

    def iter_seqview_seqs_str(self, *, name_order: list[str] = None) -> Iterator:
        name_order = name_order or self._name_order
        yield from (str(SeqView(seq=self._data[n])) for n in name_order)

    def get_view(self, seqid: str):
        seq_len = len(self._data[seqid])
        return SeqDataView(self, seqid=seqid, seq_len=seq_len)


class SeqDataView(SeqView):
    # self.seq: SeqData

    def _checked_seq_len(self, seq, seq_len) -> int:
        assert seq_len is not None
        return seq_len

    @property
    def value(self) -> str:
        raw = self.seq.get_seq_str(
            seqid=self.seqid, start=self.parent_start, stop=self.parent_stop
        )
        return raw if self.step == 1 else raw[:: self.step]
