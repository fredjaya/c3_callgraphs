import pytest

from seqdata import SeqData, SeqDataView


@pytest.fixture
def seq1():
    return "ACTG"


@pytest.fixture
def simple_dict():
    return dict(seq1="ACGT", seq2="GTTTGCA")


@pytest.fixture
def sd_demo(simple_dict: dict[str, str]):
    return SeqData(data=simple_dict)


def test_seqdata_default_attributes(sd_demo: SeqData):
    assert sd_demo._name_order == ("seq1", "seq2")
    assert sd_demo._moltype.label == "dna"


def test_seqdata_name_order(seq1: str):
    with pytest.raises(AttributeError):
        SeqData(seq1)
    # assert SeqData(data)._name_order != ("A", "C", "T", "G")


def test_get_seq_str(sd_demo: SeqData):
    got = sd_demo.get_seq_str(name="seq1", start=1, end=4)
    assert got == "CGT"

    got = sd_demo.get_seq_str(name="seq1")
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


def test_seqdataview_returns_self(seq1: str):
    assert isinstance(SeqDataView(seq1), SeqDataView)


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
    obj = SeqDataView(seq1)
    got = obj[index]
    assert isinstance(got, SeqDataView)
