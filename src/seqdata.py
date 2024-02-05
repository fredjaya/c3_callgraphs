from dataclasses import InitVar, dataclass, field

from cogent3 import get_moltype
from numpy import array


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
        self._name_order = name_order or tuple(data)
        self._moltype = get_moltype(moltype)

    def get_seq_str(self, name: str, start: int = 0, end: int = None) -> tuple[str]:
        if not end:
            end = len(self._data[name])
        return self._data[name][start:end]

    def iter_seqs_str(self, name_order: list[str] = None) -> tuple[str]:
        name_order = name_order or self._name_order
        yield from (self._data[n] for n in name_order)
