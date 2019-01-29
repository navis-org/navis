#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along

import itertools

from . import core, config

# Set up logging
logger = config.logger

def __guess_sentiment(x):
    """ Tries to classify a list of words into either <type>, <nickname>,
    <tracer> or <generic> annotations.
    """

    sent = []
    for i, w in enumerate(x):
        # If word is a number, it's most likely something generic
        if w.isdigit():
            sent.append('generic')
        elif w == 'neuron':
            # If there is a lonely "neuron" followed by a number, it's generic
            if i != len(x) and x[i + 1].isdigit():
                sent.append('generic')
            # If not, it's probably type
            else:
                sent.append('type')
        # If there is a short, all upper case word after the generic information
        elif w.isupper() and len(w) > 1 and w.isalpha() and 'generic' in sent:
            # If there is no number in that word, it's probably tracer initials
            sent.append('tracer')
        else:
            # If the word is AFTER the generic number, it's probably a nickname
            if 'generic' in sent:
                sent.append('nickname')
            # If not, it's likely type information
            else:
                sent.append('type')

    return sent


def parse_neuronname(x):
    """ Tries parsing neuron names into type, nickname, tracer and generic
    information.

    This works best if neuron name follows this convention::

    '{type} {generic} {nickname} {tracer initials}'

    Parameters
    ----------
    x :     str | TreeNeuron
            Neuron name.

    Returns
    -------
    type :          str
    nickname :      str
    tracer :        str
    generic :       str

    Examples
    --------
    >>> pymaid.utils.parse_neuronname('AD1b2#7 3080184 Dust World JJ PS')
    ('AD1b2#7', 'Dust World', 'JJ PS', '3080184')
    """

    if isinstance(x, core.TreeNeuron):
        x = x.neuron_name

    if not isinstance(x, str):
        raise TypeError('Unable to parse name: must be str, not {}'.format(type(x)))

    # Split name into single words
    words = x.split(' ')
    sentiments = __guess_sentiment(words)

    type_str = [w for w, s in zip(words, sentiments) if s == 'type']
    nick_str = [w for w, s in zip(words, sentiments) if s == 'nickname']
    tracer_str = [w for w, s in zip(words, sentiments) if s == 'tracer']
    gen_str = [w for w, s in zip(words, sentiments) if s == 'generic']

    return ' '.join(type_str), ' '.join(nick_str), ' '.join(tracer_str), ' '.join(gen_str)


def shorten_name(x, max_len=30):
    """ Shorten a neuron name by iteratively removing non-essential
    information.

    Prioritises generic -> tracer -> nickname -> type information when removing
    until target length is reached. This works best if neuron name follows
    this convention::

    '{type} {generic} {nickname} {tracers}'

    Parameters
    ----------
    x :         str | TreeNeuron
                Neuron name.
    max_len :   int, optional
                Max length of shortened name.

    Returns
    -------
    shortened name :    str

    Examples
    --------
    >>> pymaid.shorten_name('AD1b2#7 3080184 Dust World JJ PS', 30)
    'AD1b2#7 Dust World [..]'
    """

    if isinstance(x, core.TreeNeuron):
        x = x.neuron_name

    # Split into individual words and guess their type
    words = x.split(' ')
    sentiments = __guess_sentiment(words)

    # Make sure we're working on a copy of the original neuron name
    short = str(x)

    ty = ['generic', 'tracer', 'nickname', 'type']
    # Iteratively remove generic -> tracer -> nickname -> type information
    for t, (w, sent) in itertools.product(ty, zip(words[::-1], sentiments[::-1])):
        # Stop if we are below max length
        if len(short) <= max_len:
            break
        # Stop if there is only a single word left
        elif len(short.replace('[..]', '').strip().split(' ')) == 1:
            break
        # Skip if this word is not of the right sentiment
        elif t != sent:
            continue
        # Remove this word
        short = short.replace(w, '[..]').strip()
        # Make sure to merge consecutive '[..]''
        while '[..] [..]' in short:
            short = short.replace('[..] [..]', '[..]')

    return short