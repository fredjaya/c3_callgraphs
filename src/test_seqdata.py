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
    
    with pytest.raises(TypeError):
        smalldemo.get_seq_str("seq1")


def test_get_seq_array(smalldemo: SeqData):
    got = smalldemo.get_seq_str("seq1", start=1, end=4)
    assert got == "CGT"
