from dataclasses import dataclass
from numpy import array
@dataclass(slots=True)
class SeqData:
    _data: dict[str,str]
    _moltype: "MolType" = "dna"
    _name_order: tuple[str] = None

    def get_seq_str(self, name: str, start: int = 0 , end: int = None) -> tuple[str]:
        if not end:
            end = len(self._data[name])
        return self._data[name][start:end]