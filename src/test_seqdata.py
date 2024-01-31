import pytest
from seqdata import SeqData
from cogent3 import get_moltype


@pytest.fixture
def smalldemo():
    return SeqData(_data=dict(seq1="ACGT", seq2="GTTTGCA"))


def test_get_seq_str(smalldemo: SeqData):
    got = smalldemo.get_seq_str("seq1", start=1, end=4)
    assert got == "CGT"

    got = smalldemo.get_seq_str("seq1")
    assert got == "ACGT"

    with pytest.raises(TypeError):
        smalldemo.get_seq_str()

def test_get_iter_seqs_str(smalldemo: SeqData):
    with pytest.raises(TypeError):
        smalldemo.iter_seqs_str()

    with pytest.raises(TypeError):
        smalldemo.iter_seqs_str(name_order="seq1")

    got = smalldemo.iter_seqs_str(name_order=["seq2"])
    assert got == ("GTTTGCA",)

    got = smalldemo.iter_seqs_str(name_order=["seq2", "seq1"])
    assert got == ("GTTTGCA", "ACGT")
