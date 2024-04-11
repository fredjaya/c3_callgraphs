import numpy as np
import pytest
from cogent3 import get_moltype

from seqdata import SeqData, SeqDataView, AlignedData, process_name_order, seq_index


@pytest.fixture
def seq1():
    return "ACTG"


@pytest.fixture
def simple_dict():
    return dict(seq1="ACGT", seq2="GTTTGCA")


@pytest.fixture
def simple_dict_arr():
    return dict(seq1=np.array([2, 1, 3, 0]), seq2=np.array([3, 0, 0, 0, 3, 1, 2]))


@pytest.fixture
def sd_demo(simple_dict: dict[str, str]):
    return SeqData(data=simple_dict)


@pytest.fixture
def int_arr():
    return np.arange(17, dtype=np.uint8)


@pytest.fixture
def sdv_s2(sd_demo: SeqData) -> SeqDataView:
    return sd_demo.get_seq_view(seqid="seq2")


@pytest.fixture
def aligned_dict():
    return dict(seq1="ACG--T", seq2="-CGAAT")


def test_seqdata_default_attributes(sd_demo: SeqData):
    assert sd_demo._name_order == ("seq1", "seq2")
    assert sd_demo._moltype.label == "dna"


def test_seqdata_seq_if_str(seq1: str):
    with pytest.raises(NotImplementedError):
        SeqData(seq1)
    # assert SeqData(data)._name_order != ("A", "C", "T", "G")


def test_process_name_order_dict_happy(simple_dict):
    got = process_name_order(simple_dict, None)
    assert got == ("seq1", "seq2")
    got = process_name_order(simple_dict, name_order=("seq2", "seq1"))
    assert got == ("seq2", "seq1")


@pytest.mark.parametrize(
    "bad_names", [("bad"), ("bad",), ("bad2", "bad1"), ("seq1",), "seq1"]
)
def test_process_name_order_dict_bad(simple_dict, bad_names):
    with pytest.raises(ValueError):
        process_name_order(simple_dict, bad_names)


@pytest.mark.parametrize(
    "names", (["seq2"], ["seq2", "seq1"], ("seq2",), ("seq2", "seq1"))
)
def test_name_order_tuple_list_happy(names, sd_demo):
    correct_names = sd_demo._name_order
    got = process_name_order(correct_names, names)
    assert got == tuple(names)


@pytest.mark.parametrize("bad_names", [("bad"), ("bad",), ("bad2", "bad1"), "seq1"])
def test_name_order_tuple_list_bad(bad_names, sd_demo):
    correct_names = sd_demo._name_order
    with pytest.raises(ValueError):
        process_name_order(correct_names, bad_names)


@pytest.mark.parametrize(
    "bad_names", [("bad"), ("bad",), ("bad2", "bad1"), ("seq1",), "seq1"]
)
def test_name_order_init(simple_dict, bad_names):
    sd = SeqData(simple_dict)
    assert sd._name_order == ("seq1", "seq2")
    sd = SeqData(simple_dict, name_order=("seq2", "seq1"))
    assert sd._name_order == ("seq2", "seq1")

    with pytest.raises(ValueError):
        SeqData(simple_dict, name_order=bad_names)


def test_seqdata_get_seq_view(sd_demo: SeqData):
    got = sd_demo.get_seq_view("seq1")
    expect = f"SeqDataView(seq={sd_demo}, start=0, stop=4, step=1, offset=0, seqid='seq1', seq_len=4)"
    assert repr(got) == expect

    got = sd_demo.get_seq_view("seq2")
    expect = f"SeqDataView(seq={sd_demo}, start=0, stop=7, step=1, offset=0, seqid='seq2', seq_len=7)"
    assert repr(got) == expect


@pytest.mark.parametrize("seq", ("seq1", "seq2"))
@pytest.mark.parametrize("start", (None, -1, 0, 1, 4))
@pytest.mark.parametrize("stop", (None, -1, 0, 1, 4))
def test_get_seq_str(simple_dict, seq, start, stop):
    # slicing should be tested in test_get_seq_array
    expect = simple_dict[seq][start:stop]
    sd = SeqData(data=simple_dict)
    got = sd.get_seq_str(seqid=seq, start=start, stop=stop)
    assert expect == got


def test_get_seq_str_empty(sd_demo: SeqData):
    with pytest.raises(TypeError):
        sd_demo.get_seq_str()


def test_iter_names(sd_demo: SeqData):
    got = sd_demo.iter_names()
    assert tuple(got) == ("seq1", "seq2")

    got = sd_demo.iter_names(name_order=["seq2", "seq1"])
    assert tuple(got) == ("seq2", "seq1")

    got = sd_demo.iter_names(name_order=["seq2"])
    assert tuple(got) == ("seq2",)

    got = sd_demo.iter_names(name_order=("bad", "bad2"))
    with pytest.raises(ValueError):
        tuple(got)


@pytest.mark.parametrize("idx, seq", ([0, "seq1"], [1, "seq2"]))
def test_iter_seq_view(sd_demo: SeqData, idx, seq):
    got = tuple(sd_demo.iter_seq_view())
    assert len(got) == 2
    assert got[idx].seq == sd_demo
    assert got[idx].seqid == seq


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
    alpha = get_moltype(moltype_name).alphabets.degen_gapped
    got = seq_index(int_arr, alpha)
    assert np.array_equal(got, int_arr)
    assert got.dtype == int_arr.dtype


def test_get_seq_array(simple_dict):
    # TODO: slicing should be tested here, not get_seq_str
    # seq1
    expect = np.array([2, 1, 3, 0], dtype="uint8")
    sd = SeqData(data=simple_dict)
    got = sd.get_seq_array(seqid="seq1")
    assert np.array_equal(got, expect)

    # seq2
    expect = np.array([3, 0, 0, 0, 3, 1, 2], dtype="uint8")
    sd = SeqData(data=simple_dict)
    got = sd.get_seq_array(seqid="seq2")
    assert np.array_equal(got, expect)


def test_get_seq_bytes(sd_demo: SeqData):
    # getting seqid and slicing tested in test_get_seq_str
    got = sd_demo.get_seq_bytes(seqid="seq1")
    assert isinstance(got, bytes)


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


# SeqDataView tests for returning an instance of itself
def test_seqdataview_returns_self(sd_demo: SeqData):
    sdv = sd_demo.get_seq_view("seq1")
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


# SeqDataView tests for value properties
@pytest.mark.parametrize("start", (None, 0, 1, 4, -1, -4))
@pytest.mark.parametrize("stop", (None, 0, 1, 4, -1, -4))
@pytest.mark.parametrize("step", (None, 1, 2, 3, -1, -2, -3))
def test_seqdataview_value(simple_dict: dict, start, stop, step):
    seq = "seq2"
    expect = simple_dict[seq][start:stop:step]
    sd = SeqData(data=simple_dict)
    # Get SeqDataView on seq
    sdv = sd.get_seq_view(seqid=seq)
    sdv2 = sdv[start:stop:step]
    got = sdv2.value
    assert got == expect


@pytest.mark.parametrize("start", (None, 0, 1, 4, -1, -4))
@pytest.mark.parametrize("stop", (None, 0, 1, 4, -1, -4))
@pytest.mark.parametrize("step", (None, 1, 2, 3, -1, -2, -3))
def test_array_value(simple_dict_arr: dict, start, stop, step):
    seq = "seq2"
    expect = simple_dict_arr[seq][start:stop:step]
    sd = SeqData(data=simple_dict_arr)
    # Get SeqDataView on seq
    sdv = sd.get_seq_view(seqid=seq)
    got = sdv.array_value[start:stop:step]
    assert np.array_equal(got, expect)


@pytest.mark.parametrize("start", (None, 0, 1, 4, -1, -4))
@pytest.mark.parametrize("stop", (None, 0, 1, 4, -1, -4))
@pytest.mark.parametrize("step", (None, 1, 2, 3, -1, -2, -3))
def test_bytes_value(simple_dict: dict, start, stop, step):
    seq = "seq2"
    expect = simple_dict[seq][start:stop:step]
    expect = expect.encode("utf8")
    sd = SeqData(data=simple_dict)
    # Get SeqDataView on seq
    sdv = sd.get_seq_view(seqid=seq)
    got = sdv.bytes_value[start:stop:step]
    assert expect == got


# SeqDataView tests for special methods that access "value" properties
def test_array(sdv_s2: SeqDataView):
    expect = sdv_s2.array_value
    got = np.array(sdv_s2)
    assert np.array_equal(expect, got)


def test_bytes(sdv_s2: SeqDataView):
    expect = sdv_s2.bytes_value
    got = bytes(sdv_s2)
    assert expect == got

# AlignedSeqData tests
# TODO: How to test AlignedData.from_strings correctly?

def test_from_string_unequal_seqlens():
    data = dict(seq1="A-A", seq2="AAAAAAA--")
    with pytest.raises(ValueError):
        AlignedData.from_strings(data=data)

def test_aligned_from_string_returns_self(aligned_dict):
    got = AlignedData.from_strings(data=aligned_dict)
    assert isinstance(got, AlignedData)
    # assert gap lengths


# AlignedData get_seq_* tests
def test_aligned_get_seq_array(aligned_dict):
    expect = np.array([2, 1, 3, 0], dtype="uint8")
    ad = AlignedData.from_strings(data=aligned_dict)
    got = ad.get_seq_array(seqid="seq1")
    assert np.array_equal(got, expect)

@pytest.mark.parametrize("seq", ("seq1", "seq2"))
@pytest.mark.parametrize("start", (None, -1, 0, 1, 4))
@pytest.mark.parametrize("stop", (None, -1, 0, 1, 4))
def test_aligned_get_seq_str(aligned_dict, seq, start, stop):
    # slicing should be tested in test_aligned_get_seq_array
    expect = aligned_dict[seq][start:stop]
    sd = SeqData(data=aligned_dict)
    got = sd.get_seq_str(seqid=seq, start=start, stop=stop)
    assert expect == got


def test_aligned_get_seq_bytes(aligned_dict):
    ad = AlignedData.from_strings(aligned_dict)
    got = ad.get_seq_bytes(seqid="seq1")
    assert isinstance(got, bytes)