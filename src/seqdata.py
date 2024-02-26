from dataclasses import InitVar, dataclass, field
from typing import Iterator, Optional

from cogent3 import get_moltype
from cogent3.core.sequence import SeqView


@dataclass(slots=True)
class SeqData:
    _data: dict[str, str] = field(init=False)
    _moltype: "MolType" = field(init=False)
    _name_order: tuple[str] = field(init=False)
    data: InitVar[dict[str, str]]
    moltype: InitVar[str | None] = "dna"
    name_order: InitVar[tuple[str] | None] = None

    def __post_init__(self, data, moltype, name_order):
        self._data = data
        self._name_order = name_order or tuple(data.keys())
        self._moltype = get_moltype(moltype)

    def get_seq_str(
        self, *, name: str, start: int = None, end: int = None
    ) -> tuple[str]:
        return self._data[name][start:end]

    def iter_seqs_str(self, *, name_order: list[str] = None) -> Iterator:
        name_order = name_order or self._name_order
        yield from (self.get_seq_str(name=n) for n in name_order)

    def iter_names(self, *, name_order: list[str] = None) -> Iterator:
        yield from iter(name_order or self._name_order)

    def iter_seqview_seqs_str(self, *, name_order: list[str] = None) -> Iterator:
        name_order = name_order or self._name_order
        yield from (str(SeqView(seq=self._data[n])) for n in name_order)


class SeqDataView(SeqView):
    # Don't use dataclasses due to avoid inheritance issues

    def __init__(self, seq, *args, **kwargs):
        super().__init__(seq, *args, **kwargs)
