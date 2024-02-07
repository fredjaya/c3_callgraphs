import pytest
from cogent3 import get_moltype

from seqdata import SeqData, SeqDataView


@pytest.fixture
def smalldemo():
    return SeqData(data=dict(seq1="ACGT", seq2="GTTTGCA"))


def test_default():
    seqdata = SeqData(data=dict(seq1="ACGT"))
    assert seqdata._name_order == ("seq1",)
    assert seqdata._moltype.label == "dna"


def test_get_seq_str(smalldemo: SeqData):
    got = smalldemo.get_seq_str(name="seq1", start=1, end=4)
    assert got == "CGT"

    got = smalldemo.get_seq_str(name="seq1")
    assert got == "ACGT"

    with pytest.raises(TypeError):
        smalldemo.get_seq_str()


def test_iter_seqs_str(smalldemo: SeqData):
    got = smalldemo.iter_seqs_str(name_order=["seq2"])
    assert tuple(got) == ("GTTTGCA",)

    got = smalldemo.iter_seqs_str(name_order=["seq2", "seq1"])
    assert tuple(got) == ("GTTTGCA", "ACGT")

    got = smalldemo.iter_seqs_str()
    assert tuple(got) == ("ACGT", "GTTTGCA")


def test_iter_names(smalldemo: SeqData):
    got = smalldemo.iter_names()
    assert tuple(got) == ("seq1", "seq2")

    got = smalldemo.iter_names(name_order=["seq2", "seq1"])
    assert tuple(got) == ("seq2", "seq1")

    got = smalldemo.iter_names(name_order=["seq2"])
    assert tuple(got) == ("seq2",)


def test_get_iter_seqview_seqs_str(smalldemo: SeqData):
    got = smalldemo.iter_seqview_seqs_str(name_order=["seq2"])
    assert tuple(got) == ("GTTTGCA",)

    got = smalldemo.iter_seqview_seqs_str(name_order=["seq2", "seq1"])
    assert tuple(got) == ("GTTTGCA", "ACGT")

    got = smalldemo.iter_seqview_seqs_str()
    assert tuple(got) == ("ACGT", "GTTTGCA")


def test_seqdataview_returns_self():
    assert isinstance(SeqDataView(), SeqDataView)


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
def test_seqdataview_slice_returns_self(index: slice):
    obj = SeqDataView()
    got = obj[index]
    assert isinstance(got, SeqDataView)
