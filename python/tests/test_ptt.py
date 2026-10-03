from types import SimpleNamespace as K

import pytest

from gvision.audio.ptt import Hotkey, key_name, parse_hotkey


def test_parse_hotkey():
    assert parse_hotkey("Alt+3") == (frozenset({"alt"}), "3")
    assert parse_hotkey("f8") == (frozenset(), "f8")
    assert parse_hotkey("ctrl + shift + space") == (frozenset({"ctrl", "shift"}), "space")
    for bad in ["", "alt", "win+3"]:
        with pytest.raises(ValueError):
            parse_hotkey(bad)


def test_key_names_from_pynput_events():
    assert key_name(K(name="alt_l")) == "alt"
    assert key_name(K(name="alt_gr")) == "alt"
    assert key_name(K(name="f8")) == "f8"
    # Alt+3 on a Spanish layout: the char is not '3', the virtual key is.
    assert key_name(K(vk=0x33, char="·")) == "3"
    assert key_name(K(vk=None, char="Q")) == "q"


def test_alt_3_hold_and_release():
    hk = Hotkey("alt+3")
    assert not hk.press("3")  # 3 alone does nothing
    hk.release("3")
    assert not hk.press("alt")
    assert hk.press("3")  # now held
    assert not hk.press("3")  # auto-repeat
    assert not hk.press("alt")
    assert hk.release("alt")  # letting go of either part stops talking
    assert not hk.release("3")
    assert hk.press("alt") is False and hk.press("3") is True


def test_single_key():
    hk = Hotkey("f8")
    assert hk.press("f8") and not hk.press("f8")
    assert not hk.release("x")
    assert hk.release("f8")
