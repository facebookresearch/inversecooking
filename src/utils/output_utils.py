# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

replace_dict = {' .': '.',
                ' ,': ',',
                ' ;': ';',
                ' :': ':',
                '( ': '(',
                ' )': ')',
               " '": "'"}


def get_recipe(ids, vocab):
    toks = []
    for id_ in ids:
        toks.append(vocab[id_])
    return toks


def get_ingrs(ids, ingr_vocab_list):
    gen_ingrs = []
    for ingr_idx in ids:
        ingr_name = ingr_vocab_list[ingr_idx]
        if ingr_name == '<pad>':
            break
        gen_ingrs.append(ingr_name)
    return gen_ingrs


def prettify(toks, replace_dict):
    toks = ' '.join(toks)
    toks = toks.split('<end>')[0]
    sentences = toks.split('<eoi>')

    pretty_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = sentence.capitalize()
        for k, v in replace_dict.items():
            sentence = sentence.replace(k, v)
        if sentence != '':
            pretty_sentences.append(sentence)
    return pretty_sentences


def colorized_list(ingrs, ingrs_gt, colorize=False):
    if colorize:
        colorized_list = []
        for word in ingrs:
            if word in ingrs_gt:
                word = '\033[1;30;42m ' + word + ' \x1b[0m'
            else:
                word = '\033[1;30;41m ' + word + ' \x1b[0m'
            colorized_list.append(word)
        return colorized_list
    else:
        return ingrs


def prepare_output(ids, gen_ingrs, ingr_vocab_list, vocab):

    toks = get_recipe(ids, vocab)
    is_valid = True
    reason = 'All ok.'
    try:
        cut = toks.index('<end>')
        toks_trunc = toks[0:cut]
    except:
        toks_trunc = toks
        is_valid = False
        reason = 'no eos found'

    # repetition score
    score = float(len(set(toks_trunc))) / float(len(toks_trunc))

    prev_word = ''
    found_repeat = False
    for word in toks_trunc:
        if prev_word == word and prev_word != '<eoi>':
            found_repeat = True
            break
        prev_word = word

    toks = prettify(toks, replace_dict)
    title = toks[0]
    toks = toks[1:]

    if gen_ingrs is not None:
        gen_ingrs = get_ingrs(gen_ingrs, ingr_vocab_list)

    if score <= 0.3:
        reason = 'Diversity score.'
        is_valid = False
    elif len(toks) != len(set(toks)):
        reason = 'Repeated instructions.'
        is_valid = False
    elif found_repeat:
        reason = 'Found word repeat.'
        is_valid = False

    valid = {'is_valid': is_valid, 'reason': reason, 'score': score}
    outs = {'title': title, 'recipe': toks, 'ingrs': gen_ingrs}

    return outs, valid
