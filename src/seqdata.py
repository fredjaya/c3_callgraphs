from dataclasses import InitVar, dataclass, field
from functools import singledispatch, singledispatchmethod
from typing import Iterator, Optional, Self, Union

import numpy as numpy
from cogent3 import get_moltype
from cogent3.core.alphabet import CharAlphabet
from cogent3.core.moltype import MolType
from cogent3.core.sequence import SeqView
from ensembl_lite._aligndb import GapPositions

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
    """
    A view class for SeqData, providing properties for different
    representations.

    self.seq is a SeqData() instance, but other properties are a reference to a single
    seqid only.

    Example
    -------
    data = {"seq1": "ACGT", "seq2": "GTTTGCA"}
    sd = SeqData(data=data)
    sdv = sd.get_seq_view(seqid="seq1")
    """

    def _checked_seq_len(self, seq, seq_len) -> int:
        assert seq_len is not None
        return seq_len

    # inherits __str__
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


def seq_to_gap_coords(parent_seq: str) -> tuple[str, numpy.ndarray]:
    """
    Parameters
    ----------
    parent_seq : str
        Sequence with gaps

    Returns
    -------
    ungapped : str

    gap_coords : GapPositions
    """
    parent_len = len(parent_seq)
    # seq with only gaps
    if parent_seq == "-" * parent_len:
        return None, numpy.array([0, len(parent_seq)])

    # Remove gaps from seq
    ungapped = parent_seq.replace("-", "")

    # seq with no gaps
    if len(ungapped) == parent_len:
        return parent_seq, None

    # iterate over positions
    gap_start = False
    coords = []
    for pos, i in enumerate(parent_seq):
        if i == "-":
            if gap_start is False:
                # record start pos of gap
                gap_start = pos
            if pos == parent_len - 1:
                # seq ends with a gap
                gap_len = pos - gap_start + 1
                coords.append([gap_start, gap_len])
        elif gap_start is not False:
            # End gap segment
            gap_len = pos - gap_start
            coords.append([gap_start, gap_len])
            gap_start = False  # reset

    return ungapped, GapPositions(numpy.array(coords), parent_len)


class AlignedDataView(SeqDataView):
    """
    Example
    -------
    data = {"seq1": "ACGT", "seq2": "GTTTGCA"}
    ad = AlignedData.from_strings(data)
    adv = sd.get_aligned_view(seqid="seq1")
    """

    # methods for outputting different data types will need to be overridden
    @property
    def value(self) -> str:
        coords, ungapped = self.seq.get_gaps(seqid=self.seqid)
        aligned = gap_coords_to_seq(coords, ungapped)
        str_aligned = str(aligned)
        raw_seq = str_aligned[self.parent_start : self.parent_stop]
        # raw_seq = self.seq.get_seq_str(
        #    seqid=self.seqid, start=self.parent_start, stop=self.parent_stop
        # )

        return raw_seq if self.step == 1 else raw_seq[:: self.step]

    # def value()
    # if not sliced return original string
    # _convert.gap_coords_to_seq and vice versa
    # raw_seqs and raw_gaps from AlignedData
    # Method on AlignedData - get seq and get gaps
    # pass seqid only, for now

    # TODO: 3. def array_value() -> index and gaps
    # TODO: 4. def bytes_value() -> index and gaps


@dataclass
class AlignedData:
    # Look out for any overlaps with SeqData
    # Check: Made seqs and gaps optional for classmethod to work?
    seqs: Optional[dict[str, numpy.ndarray]] = None
    gaps: Optional[dict[str, numpy.ndarray]] = None
    _moltype: MolType = field(init=False)
    _name_order: tuple[str] = field(init=False)
    _alpha: CharAlphabet = field(init=False)
    align_len: int = 0
    moltype: InitVar[str | None] = "dna"
    name_order: InitVar[tuple[str] | None] = None

    def __post_init__(self, moltype, name_order):
        self._moltype = get_moltype(moltype)
        self._alpha = self._moltype.alphabets.degen_gapped
        self._name_order = process_name_order(self.seqs, name_order)
        # Check: it works, but I don't know why.

    @classmethod
    def from_strings(
        cls,
        data: dict[str, str],
        moltype: str = "dna",
        name_order: Optional[tuple[str]] = None,
    ) -> Self:
        """
        Convert dict of {seq_name: seq, ...} to two dicts for seqs and gaps
        """
        seq_lengths = {len(v) for v in data.values()}
        if len(seq_lengths) != 1:
            raise ValueError("All sequence lengths must be the same.")

        align_len = seq_lengths.pop()
        moltype = get_moltype(moltype)
        alpha = moltype.alphabets.degen_gapped

        seqs = {}
        gaps = {}
        for name, seq in data.items():
            seqs[name], gaps[name] = aligned_to_seq_gaps(seq)

        name_order = process_name_order(seqs, name_order)

        return cls(
            seqs=seqs,
            gaps=gaps,
            moltype=moltype,
            name_order=name_order,
            align_len=align_len,
        )

    def get_aligned_view(self, seqid: str) -> AlignedDataView:
        # Need to revisit what the variable is called i.e. parent_length
        return AlignedDataView(self, seqid=seqid, seq_len=self.align_len)

    def get_seq_array(
        self, *, seqid: str, start: int = None, stop: int = None
    ) -> numpy.ndarray:
        # TODO: Should return seq interleaved with gaps
        # return: [2, 1, 3, 0,  -, -]
        # same for bytes and str
        # TODO: GapPosition() should have a method that
        # Must convert seq to alignment coordinates
        ungapped = self.seqs[seqid]
        gap_pos = self.gaps[seqid]
        return self.seqs[seqid][start:stop]

    def get_seq_str(self, *, seqid: str, start: int = None, stop: int = None) -> str:
        return self._alpha.from_indices(
            self.get_seq_array(seqid=seqid, start=start, stop=stop)
        )

    def get_seq_bytes(
        self, *, seqid: str, start: int = None, stop: int = None
    ) -> bytes:
        return self.get_seq_str(seqid=seqid, start=start, stop=stop).encode("utf8")

    def get_gaps(self, seqid: str) -> numpy.ndarray:
        return self.gaps[seqid]
