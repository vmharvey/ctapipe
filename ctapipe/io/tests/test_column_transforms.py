import numpy as np


def test_utf8_max_max_len():
    from ctapipe.io.tableio import encode_utf8_max_len

    assert encode_utf8_max_len("hello", 4) == b"hell"

    # each of these is 2 bytes
    assert encode_utf8_max_len("αβ", 4) == "αβ".encode("utf-8")
    assert encode_utf8_max_len("αβγ", 4) == "αβ".encode("utf-8")
    assert encode_utf8_max_len("αβγ", 3) == "α".encode("utf-8")

    emoji = "foo \N{FACE WITH MEDICAL MASK}"
    assert encode_utf8_max_len(emoji, 8) == emoji.encode("utf-8")
    assert encode_utf8_max_len(emoji, 5) == b"foo "


def test_string_transform():
    from ctapipe.io.tableio import StringTransform

    trafo = StringTransform(10)

    assert trafo("hello") == b"hello"
    assert trafo("12345678910") == b"1234567891"

    strings = np.array(["foo", "bar"])
    assert np.all(trafo(strings) == np.array([b"foo", b"bar"]))

    assert trafo.inverse(b"hello") == "hello"
    assert trafo.inverse("α".encode("utf-8")) == "α"

    nonascii = np.array(["αβßä", "abc"])
    assert np.all(trafo.inverse(trafo(nonascii)) == nonascii)


def test_string_truncated():
    from ctapipe.io.tableio import StringTransform

    trafo = StringTransform(5)
    assert trafo("hello world") == b"hello"

    # 6 bytes, make sure we only write valid utf-8
    assert trafo("ααα") == "αα".encode("utf-8")
