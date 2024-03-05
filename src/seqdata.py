from dataclasses import InitVar, dataclass, field
from typing import Iterator, Optional

import numpy as np
from cogent3 import get_moltype
from cogent3.core.sequence import SeqView


def seq_to_index_array(seq: str, moltype: str = "dna") -> np.array:
    alpha = get_moltype(moltype).alphabets.degen_gapped
    indices = alpha.to_indices(seq)
    return np.array(indices)


@dataclass(slots=True)
class SeqData:
    _data: dict[str, str] = field(init=False)
    _moltype: "MolType" = field(init=False)
    _name_order: tuple[str] = field(init=False)
    data: InitVar[dict[str, str]]
    moltype: InitVar[str | None] = "dna"
    name_order: InitVar[tuple[str] | None] = None

    def __post_init__(self, data, moltype, name_order):
        # Write function (external to class) takes single value and alpha, returns numpy array.
        # Must have identical signatures
        self._moltype = get_moltype(moltype)
        # Reference to degen alpha
        # dict comp on every value, dict of numpy arrays
        self._data = data
        self._name_order = name_order or tuple(data.keys())

    def get_seq_array(self, *, seqid: str, start: int = None, stop: int = None):
        # Let get_seq_str deal with indices
        pass

    def get_seq_str(
        self, *, seqid: str, start: int = None, stop: int = None
    ) -> tuple[str]:
        # Add idx-alpha conversion
        return self._data[seqid][start:stop]

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
