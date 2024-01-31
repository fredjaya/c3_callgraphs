from dataclasses import dataclass
from typing import Tuple, List
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
    
    def iter_seqs_str(self, name_order: List[str]) -> tuple[str]:
        if not name_order:
            raise TypeError("`name_order` must be provided.")
        if not isinstance(name_order, list):
            raise TypeError("`name_order` must be a list.")
        return tuple(self._data[name] for name in name_order)

