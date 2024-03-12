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


@pytest.mark.parametrize(
    "bad_names", [("bad"), ("bad",), ("bad2", "bad1"), ("seq1",), "seq1"]
)
def test_name_order_init(simple_dict, bad_names):
    sd = SeqData(simple_dict)
    assert sd._name_order == ("seq1", "seq2")
    sd = SeqData(simple_dict, name_order=("seq2", "seq1"))
    assert sd._name_order == ("seq2", "seq1")

    with pytest.raises(AssertionError):
        SeqData(simple_dict, name_order=bad_names)


def test_seqdata_get_seq_view(sd_demo: SeqData):
    got = sd_demo.get_seq_view("seq1")
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
    # TODO: You can input name_order=["rubbish", "doesn't exist"] and outputs
    # those even though it doesn't exist
    got = sd_demo.iter_names()
    assert tuple(got) == ("seq1", "seq2")

    got = sd_demo.iter_names(name_order=["seq2", "seq1"])
    assert tuple(got) == ("seq2", "seq1")

    got = sd_demo.iter_names(name_order=["seq2"])
    assert tuple(got) == ("seq2",)


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
def test_seqdataview_value(simple_dict: dict, seq: str, start, stop, step):
    expect = simple_dict[seq][start:stop:step]
    sd = SeqData(data=simple_dict)
    # Get SeqDataView on seq
    sdv = sd.get_view(seqid=seq)
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
    [(b"TCAG-NRYWSKMBDHV?", "dna"), (b"UCAG-NRYWSKMBDHV?", "rna")],
)
def test_seq_index_bytes(seq, moltype_name, int_arr):
    alpha = get_moltype(moltype_name).alphabets.degen_gapped
    got = seq_index(seq, alpha)
    assert np.array_equal(got, int_arr)


@pytest.mark.parametrize("moltype_name", ("dna", "rna"))
def test_seq_index_arr(moltype_name, int_arr):
    # Should return the same thing
    alpha = get_moltype(moltype_name).alphabets.degen_gapped
    got = seq_index(int_arr, alpha)
    assert np.array_equal(got, int_arr)


@pytest.mark.parametrize("seq", ("seq1", "seq2"))
def test_get_seq_array(simple_dict, seq):
    expect = np.array(simple_dict[seq])
    sd = SeqData(data=simple_dict)
    got = sd.get_seq_array(seqid=seq)


@pytest.mark.parametrize("seq", ("seq1", "seq2"))
def test_getitem_str(sd_demo, seq):
    got = sd_demo[seq]
    assert got.seq == sd_demo
    assert got.seqid == seq


@pytest.mark.parametrize("idx", (0, 1))
def test_getitem_int(simple_dict, idx):
    sd = SeqData(simple_dict)
    got = sd[idx]
    assert got.seq == sd
    assert got.seqid == list(simple_dict)[idx]
