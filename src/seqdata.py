from dataclasses import InitVar, dataclass, field
from functools import singledispatch, singledispatchmethod
from typing import Iterator, Optional, Self, Union

import numpy as numpy
from cogent3 import get_moltype
from cogent3.core.alphabet import CharAlphabet
from cogent3.core.moltype import MolType
from cogent3.core.sequence import SeqView
import _convert as convert

T = Union[str, bytes, numpy.ndarray]


@singledispatch
def seq_index(seq: T, alphabet: CharAlphabet) -> numpy.ndarray:
    raise NotImplementedError(
        f"{seq_index.__name__} not implemented for type {type(seq)}"
    )


@seq_index.register
def _(seq: str | bytes, alphabet: CharAlphabet) -> numpy.ndarray:
    return alphabet.to_indices(seq)


@seq_index.register
def _(seq: numpy.ndarray, alphabet: CharAlphabet) -> numpy.ndarray:
    return seq.astype(alphabet.array_type)


@singledispatch
def process_name_order(
    correct_names: Union[dict, tuple, list], name_order: tuple
) -> tuple:
    """dict (data) for constructor; tuple for SeqData instance"""
    raise NotImplementedError(
        f"process_name_order not implemented for type {type(correct_names)}"
    )


@process_name_order.register
def _(correct_names: dict, name_order: tuple) -> tuple:
    keys = correct_names.keys()
    if name_order is None:
        return tuple(keys)
    if set(name_order) == set(keys):
        return name_order
    raise ValueError("name_order does not match dictionary keys")


@process_name_order.register
def _(correct_names: tuple | list, name_order: tuple) -> tuple:
    if name_order is None:
        return correct_names
    if set(name_order) <= set(correct_names):
        return tuple(name_order)
    raise ValueError("some names do not match")


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

    @property
    def array_value(self) -> numpy.ndarray:
        raw = self.seq.get_seq_array(
            seqid=self.seqid, start=self.parent_start, stop=self.parent_stop
        )
        return raw if self.step == 1 else raw[:: self.step]

    @property
    def bytes_value(self) -> bytes:
        raw = self.seq.get_seq_bytes(
            seqid=self.seqid, start=self.parent_start, stop=self.parent_stop
        )
        return raw if self.step == 1 else raw[:: self.step]

    def __array__(self):
        return self.array_value

    def __bytes__(self):
        return self.bytes_value


@dataclass
class SeqData:
    # Separate AlignedSeqData for alignments (need to store gaps somehow)
    # ABC and interface
    _data: dict[str, numpy.ndarray] = field(init=False)
    _moltype: MolType = field(init=False)
    _name_order: tuple[str] = field(init=False)
    _alpha: CharAlphabet = field(init=False)
    data: InitVar[dict[str, str]]
    moltype: InitVar[str | None] = "dna"
    name_order: InitVar[tuple[str] | None] = None

    def __post_init__(self, data, moltype, name_order):
        self._moltype = get_moltype(moltype)
        self._alpha = self._moltype.alphabets.degen_gapped
        self._name_order = process_name_order(data, name_order)
        # When SeqData is initialised, sequence strings are converted to moltype alphabet indicies
        self._data = {k: seq_index(v, self._alpha) for k, v in data.items()}

    @singledispatchmethod
    def __getitem__(self, value: str | int) -> SeqDataView:
        raise NotImplementedError(f"__getitem__ not implemented for {type(value)}")

    @__getitem__.register(str)
    def _(self, value: str) -> SeqDataView:
        return self.get_seq_view(seqid=value)

    @__getitem__.register(int)
    def _(self, value: int) -> SeqDataView:
        seqid = self._name_order[value]
        return self.get_seq_view(seqid=seqid)

    def get_seq_array(
        self, *, seqid: str, start: int = None, stop: int = None
    ) -> numpy.ndarray:
        return self._data[seqid][start:stop]

    def get_seq_str(self, *, seqid: str, start: int = None, stop: int = None) -> str:
        return self._alpha.from_indices(
            self.get_seq_array(seqid=seqid, start=start, stop=stop)
        )

    def get_seq_bytes(
        self, *, seqid: str, start: int = None, stop: int = None
    ) -> bytes:
        return self.get_seq_str(seqid=seqid, start=start, stop=stop).encode("utf8")

    def iter_names(self, *, name_order: tuple[str] = None) -> Iterator:
        yield from process_name_order(self._name_order, name_order)

    def get_seq_view(self, seqid: str) -> SeqDataView:
        seq_len = len(self._data[seqid])
        return SeqDataView(self, seqid=seqid, seq_len=seq_len)

    def iter_seq_view(self, *, name_order: tuple[str] = None) -> Iterator:
        # Should this output SeqView or SeqDataView?
        seqids = process_name_order(self._name_order, name_order)
        for seqid in seqids:
            yield self.get_seq_view(seqid=seqid)


def aligned_to_seq_gaps(seq: str, name: str, moltype: MolType, alphabet: CharAlphabet) -> tuple[numpy.ndarray]:
    seq = moltype.make_seq(seq=seq, name=name)
    m, s = seq.parse_out_gaps()
    # Assuming the maximum integer is < 2^31
    gaps = numpy.array(m.get_gap_coordinates(), dtype=numpy.int32), s
    seq = seq_index(str(s), alphabet)
    return seq, gaps


@dataclass
class AlignedData:
    # Look out for any overlaps with SeqData
    # Check: Made seqs and gaps optional for classmethod to work?
    seqs: Optional[dict[str, numpy.ndarray]] = None 
    gaps: Optional[dict[str, numpy.ndarray]] = None 
    _moltype: MolType = field(init=False)
    _name_order: tuple[str] = field(init=False)
    _alpha: CharAlphabet = field(init=False)
    moltype: InitVar[str | None] = "dna"
    name_order: InitVar[tuple[str] | None] = None

    def __post_init__(self, moltype, name_order):
        self._moltype = get_moltype(moltype)
        self._alpha = self._moltype.alphabets.degen_gapped
        self._name_order = process_name_order(self.seqs, name_order)
        # Check: it works, but I don't know why. 
        
    @classmethod
    def from_strings(cls, data: dict[str, str], moltype: str="dna", name_order: Optional[tuple[str]] = None) -> Self:
        """
        Convert dict of [seq_names, seqs] to two dicts for seqs and gaps  
        """
        seq_lengths = {len(v) for v in data.values()}
        if len(seq_lengths) != 1:
            raise ValueError("All sequence lengths must be the same.")
        
        moltype = get_moltype(moltype)
        alpha = moltype.alphabets.degen_gapped

        seqs = {}
        gaps = {}
        for name, seq in data.items():
            seqs[name], gaps[name] = aligned_to_seq_gaps(seq=seq, name=name, moltype=moltype, alphabet=alpha)
        
        name_order = process_name_order(seqs, name_order)

        return cls(seqs=seqs, gaps=gaps, moltype=moltype, name_order=name_order)
    
    def get_seq_array(
        self, *, seqid: str, start: int = None, stop: int = None
    ) -> numpy.ndarray:
        return self.seqs[seqid][start:stop]

    def get_seq_str(self, *, seqid: str, start: int = None, stop: int = None) -> str:
        return self._alpha.from_indices(
            self.get_seq_array(seqid=seqid, start=start, stop=stop)
        )

    def get_seq_bytes(
        self, *, seqid: str, start: int = None, stop: int = None
    ) -> bytes:
        return self.get_seq_str(seqid=seqid, start=start, stop=stop).encode("utf8")
    

class AlignedDataView(SeqDataView):
    # methods for outputting different data types will need to be overridden

    def value():
        pass
    
    pass

    # TODO: 2.
    # def value() 
        # if not sliced return original string
    # _convert.gap_coords_to_seq and vice versa
    # raw_seqs and raw_gaps from AlignedData
    # Method on AlignedData - get seq and get gaps
    # pass seqid only, for now 

    # TODO: 3. def array_value() -> index and gaps
    # TODO: 4. def bytes_value() -> index and gaps


# TODO: Look at GapPosition tests