# -*- coding: utf-8 -*-
from __future__ import unicode_literals, absolute_import
from .cantonese import jyutping, jyutping2xsampa
from .mandarin import pinyin, pinyin2xsampa
from .vietnamese import vietnamese, vietnamese2xsampa
from .korean import korean, korean2xsampa

xsampa_consonant = [
    '#',
    '4',
    'G',
    'J',
    'N',
    'b_0',
    'b_<',
    'c',
    'dZ_0',
    'd_0',
    'd_<',
    'f',
    'g_0',
    'h',
    'j',
    'k',
    'k_>',
    'k_h',
    'k_w',
    'k_w_h',
    'k_}',
    'l',
    'm',
    'm:',
    'n',
    'n:',
    'p',
    'p_>',
    'p_h',
    'p_}',
    's',
    's\\',
    's_>',
    's`',
    't',
    'tS_>',
    'tS_h',
    't_h',
    't`s`',
    't`s`_h',
    'ts',
    'ts\\',
    'ts\\_h',
    'ts_',
    'ts_h',
    'v',
    'w',
    'x',
    'z',
    'z`',
]
xsampa_vowel = [
    '1@',
    '1@I',
    '1@U',
    '1I',
    '1U',
    '1_',
    '2',
    '6',
    '6i',
    '6u',
    '7',
    '8',
    '8y',
    '9:',
    '@',
    '@:',
    '@:I',
    '@I',
    '@N',
    '@U',
    '@n',
    '@u',
    'A',
    'E',
    'E:',
    'E:u',
    'EU',
    'HEn',
    'He',
    'M',
    'Mj',
    'N=',
    'O',
    'O:',
    'O:y',
    'OE',
    'OEU',
    'OI',
    'Oa',
    'Oa:',
    'UN',
    'V',
    'a',
    'a:',
    'a:I',
    'a:U',
    'a:i',
    'a:u',
    'a@`',
    'aI',
    'aU',
    'ai',
    'au',
    'e',
    'eU',
    'ei',
    'i',
    'i:',
    'i:u',
    'i@',
    'i@U',
    'iU',
    'jA',
    'jEn',
    'jO',
    'jUN',
    'jV',
    'ja',
    'jau',
    'je',
    'jo',
    'jou',
    'ju',
    'm=',
    'o',
    'oI',
    'oaI',
    'oaI:',
    'ou',
    'u',
    'u:',
    'u:y',
    'u@',
    'u@:',
    'u@I',
    'uI',
    'uI@',
    'ue',
    'ui:',
    'uiU',
    'w@n',
    'wA',
    'wE',
    'wV',
    'wa',
    'wai',
    'we',
    'wei',
    'wi',
    'wo',
    'y',
    'y:',
    'yn',
]


def syl2int(list_list_symbols):
    return [{k: i for i, k in enumerate(_map)} for _map in list_list_symbols]


# def int2syl(list_list_symbols):
#     return [{i: k for i, k in enumerate(_map)} for _map in list_list_symbols]


def map_symbol2index(_symbols, _symbol2xsampa, _xsampa2index):
    return {k: _xsampa2index[_symbol2xsampa[k]] for k in _symbols}


def invert(dictionaries):
    return [{v: k for k, v in d.items()} for d in dictionaries]


xsampa = [xsampa_consonant, xsampa_vowel, xsampa_consonant]

xsampa2index = syl2int(xsampa)

jyutping2index = [
    map_symbol2index(*x) for x in zip(jyutping, jyutping2xsampa, xsampa2index)
]
pinyin2index = [
    map_symbol2index(*x) for x in zip(pinyin, pinyin2xsampa, xsampa2index)
]
korean2index = [
    map_symbol2index(*x) for x in zip(korean, korean2xsampa, xsampa2index)
]
vietnamese2index = [
    map_symbol2index(*x) for x in zip(vietnamese, vietnamese2xsampa, xsampa2index)
]

language_xsampa_maps = {
    1: (pinyin2index, invert(pinyin2index)),  #mandarin
    2: (jyutping2index, invert(jyutping2index)),  #cantonese
    3: (korean2index, invert(korean2index)),  #korean
    4: (vietnamese2index, invert(vietnamese2index))  #vietnamese
}

pinyin_to_index = syl2int(pinyin)
jyutping_to_index = syl2int(jyutping)
korean_to_index = syl2int(korean)
vietnamese_to_index = syl2int(vietnamese)

language_direct_maps = {
    1: (pinyin_to_index, invert(pinyin_to_index)),  #mandarin
    2: (jyutping_to_index, invert(jyutping_to_index)),  #cantonese
    3: (korean_to_index, invert(korean_to_index)),  #korean
    4: (vietnamese_to_index, invert(vietnamese_to_index))  #vietnamese
}
