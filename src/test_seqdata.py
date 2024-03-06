import numpy as np
import pytest
from cogent3 import get_moltype

from seqdata import SeqData, SeqDataView, seq_index


@pytest.fixture
def seq1():
    return "ACTG"


@pytest.fixture
def simple_dict():
    return dict(seq1="ACGT", seq2="GTTTGCA")


@pytest.fixture
def sd_demo(simple_dict: dict[str, str]):
    return SeqData(data=simple_dict)


@pytest.fixture
def int_arr():
    return np.arange(17, dtype=np.uint8)


def test_seqdata_default_attributes(sd_demo: SeqData):
    assert sd_demo._name_order == ("seq1", "seq2")
    assert sd_demo._moltype.label == "dna"


def test_seqdata_seq_if_str(seq1: str):
    with pytest.raises(AttributeError):
        SeqData(seq1)
    # assert SeqData(data)._name_order != ("A", "C", "T", "G")


def test_seqdata_get_view(sd_demo: SeqData):
    got = sd_demo.get_view("seq1")
    expect = f"SeqDataView(seq={sd_demo}, start=0, stop=4, step=1, offset=0, seqid='seq1', seq_len=4)"
    assert repr(got) == expect

    got = sd_demo.get_view("seq2")
    expect = f"SeqDataView(seq={sd_demo}, start=0, stop=7, step=1, offset=0, seqid='seq2', seq_len=7)"
    assert repr(got) == expect


def test_get_seq_str(sd_demo: SeqData):
    got = sd_demo.get_seq_str(seqid="seq1", start=1, stop=4)
    assert got == "CGT"

    got = sd_demo.get_seq_str(seqid="seq1")
    assert got == "ACGT"

    with pytest.raises(TypeError):
        sd_demo.get_seq_str()


def test_iter_seqs_str(sd_demo: SeqData):
    got = sd_demo.iter_seqs_str(name_order=["seq2"])
    assert tuple(got) == ("GTTTGCA",)

    got = sd_demo.iter_seqs_str(name_order=["seq2", "seq1"])
    assert tuple(got) == ("GTTTGCA", "ACGT")

    got = sd_demo.iter_seqs_str()
    assert tuple(got) == ("ACGT", "GTTTGCA")


def test_iter_names(sd_demo: SeqData):
    got = sd_demo.iter_names()
    assert tuple(got) == ("seq1", "seq2")

    got = sd_demo.iter_names(name_order=["seq2", "seq1"])
    assert tuple(got) == ("seq2", "seq1")

    got = sd_demo.iter_names(name_order=["seq2"])
    assert tuple(got) == ("seq2",)


def test_get_iter_seqview_seqs_str(sd_demo: SeqData):
    got = sd_demo.iter_seqview_seqs_str(name_order=["seq2"])
    assert tuple(got) == ("GTTTGCA",)

    got = sd_demo.iter_seqview_seqs_str(name_order=["seq2", "seq1"])
    assert tuple(got) == ("GTTTGCA", "ACGT")

    got = sd_demo.iter_seqview_seqs_str()
    assert tuple(got) == ("ACGT", "GTTTGCA")


def test_seqdataview_returns_self(sd_demo: SeqData):
    sdv = sd_demo.get_view("seq1")
    assert isinstance(sdv, SeqDataView)


@pytest.mark.parametrize(
    "index",
    [
        slice(1, 3, None),
        slice(None, 2, None),
        slice(3, None, None),
        slice(0, 10, None),
        slice(None, None, 2),
        slice(None),
    ],
)
def test_seqdataview_slice_returns_self(seq1: str, index: slice):
    sdv = SeqDataView(seq1, seqid="seq1", seq_len=len(seq1))
    got = sdv[index]
    assert isinstance(got, SeqDataView)


@pytest.mark.parametrize("start", (None, 0, 1, 4, -1, -4))
@pytest.mark.parametrize("stop", (None, 0, 1, 4, -1, -4))
@pytest.mark.parametrize("step", (None, 1, 2, 3, -1, -2, -3))
@pytest.mark.parametrize("seq", ("seq1", "seq2"))
def test_seqdataview_value(sd_demo: SeqData, seq: str, start, stop, step):
    sdv = sd_demo.get_view(seq)
    expect = sd_demo._data[seq][start:stop:step]
    sdv2 = sdv[start:stop:step]
    got = sdv2.value
    assert got == expect


@pytest.mark.parametrize(
    "seq, moltype_name",
    [("TCAG-NRYWSKMBDHV?", "dna"), ("UCAG-NRYWSKMBDHV?", "rna")],
)
def test_seq_index_str(seq, moltype_name, int_arr):
    alpha = get_moltype(moltype_name).alphabets.degen_gapped
    got = seq_index(seq, alpha)
    assert np.array_equal(got, int_arr)


@pytest.mark.parametrize(
    "seq, moltype_name",
    [("TCAG-NRYWSKMBDHV?", "dna"), ("UCAG-NRYWSKMBDHV?", "rna")],
)
def test_seq_index_idx(seq, moltype_name, int_arr):
    alpha = get_moltype(moltype_name).alphabets.degen_gapped
    got = seq_index(int_arr, alpha)
    assert np.array_equal(got, seq)


@pytest.mark.parametrize(
    "seq, moltype_name",
    [(b"TCAG-NRYWSKMBDHV?", "dna"), (b"UCAG-NRYWSKMBDHV?", "rna")],
)
def test_seq_index_bytes(seq, moltype_name, int_arr):
    alpha = get_moltype(moltype_name).alphabets.degen_gapped
    got = seq_index(seq, alpha)
    assert np.array_equal(got, int_arr)
