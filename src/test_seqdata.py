import pytest
from cogent3 import get_moltype

from seqdata import SeqData


@pytest.fixture
def smalldemo():
    return SeqData(data=dict(seq1="ACGT", seq2="GTTTGCA"))

def test_default():
    seqdata = SeqData(data=dict(seq1="ACGT"))
    assert seqdata._name_order == ("seq1",)
    assert seqdata._moltype.label == "dna"

def test_get_seq_str(smalldemo: SeqData):
    got = smalldemo.get_seq_str("seq1", start=1, end=4)
    assert got == "CGT"

    got = smalldemo.get_seq_str("seq1")
    assert got == "ACGT"

    with pytest.raises(TypeError):
        smalldemo.get_seq_str()


def test_get_iter_seqs_str(smalldemo: SeqData):
    got = smalldemo.iter_seqs_str(name_order=["seq2"])
    assert tuple(got) == ("GTTTGCA",)

    got = smalldemo.iter_seqs_str(name_order=["seq2", "seq1"])
    assert tuple(got) == ("GTTTGCA", "ACGT")

    got = smalldemo.iter_seqs_str()
    assert tuple(got) == ("ACGT")