#!/usr/bin/env python3

"""This module has a function that
calculates the n-gram BLEU score"""
import numpy as np
# from collections import Counter


# def ngram_bleu(references, sentence, n):
#     """calculate the n-gram BLEU score
#     references - list of reference translations
#     reference translation - list of words in translation
#     sentence - list of model proposed sentence
#     n - size of n-gram for evaluation
#     """

#     # Generate n-grams for the sentence
#     sentence_ngrams = Counter(tuple(sentence[i:i+n]) for i in
#                               range(len(sentence) - n + 1))

#     # Generate n-grams for the references
#     reference_ngrams = [Counter(tuple(ref[i:i+n]) for i in
#                           range(len(ref) - n + 1)) for ref in references]

#     # Count the total number of n-grams in the sentence
#     sentence_count = sum(sentence_ngrams.values())

#     # Find the reference length that is closest to the sentence length
#     ref_lengths = [len(ref) for ref in references]
# closest_ref_count = min(ref_lengths, key=lambda ref_len: \
# (abs(ref_len - len(sentence)), ref_len))

#     # Count the clipped n-grams
#     clipped_count = 0
#     for ngram, count in sentence_ngrams.items():
#         max_ref_count = max(ref_ngram.get(ngram, 0) for ref_ngram in
#                             reference_ngrams)
#         clipped_count += min(count, max_ref_count)

#     # Calculate precision
#     precision = clipped_count / sentence_count if sentence_count > 0 else 0

#     # Calculate brevity penalty
#     if len(sentence) > closest_ref_count:
#          brevity_penalty = 1.0
#     else:
#         brevity_penalty = np.exp(1 - closest_ref_count / len(sentence))

#     # Calculate BLEU score
#     bleu_score = brevity_penalty * precision

#     return bleu_score

def transform_grams(references, sentence, n):
    """
    Transforms references and sentence based on grams
    """
    if n == 1:
        return references, sentence

    ngram_sentence = []
    sentence_length = len(sentence)

    for i, word in enumerate(sentence):
        count = 0
        w = word
        for j in range(1, n):
            if sentence_length > i + j:
                w += " " + sentence[i + j]
                count += 1
        if count == j:
            ngram_sentence.append(w)

    ngram_references = []

    for ref in references:
        ngram_ref = []
        ref_length = len(ref)

        for i, word in enumerate(ref):
            count = 0
            w = word
            for j in range(1, n):
                if ref_length > i + j:
                    w += " " + ref[i + j]
                    count += 1
            if count == j:
                ngram_ref.append(w)
        ngram_references.append(ngram_ref)

    return ngram_references, ngram_sentence


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    parameters:
        references [list]:
            contains reference translations
        sentence [list]:
            contains the model proposed sentence
        n [int]:
            the size of the n-gram to use for evaluation

    returns:
        the n-gram BLEU score
    """
    ngram_references, ngram_sentence = transform_grams(references, sentence, n)
    ngram_sentence_length = len(ngram_sentence)
    sentence_length = len(sentence)

    sentence_dictionary = {word: ngram_sentence.count(word) for
                           word in ngram_sentence}
    references_dictionary = {}

    for ref in ngram_references:
        for gram in ref:
            if references_dictionary.get(gram) is None or \
               references_dictionary[gram] < ref.count(gram):
                references_dictionary[gram] = ref.count(gram)

    matchings = {word: 0 for word in ngram_sentence}

    for ref in ngram_references:
        for gram in matchings.keys():
            if gram in ref:
                matchings[gram] = sentence_dictionary[gram]

    for gram in matchings.keys():
        if references_dictionary.get(gram) is not None:
            matchings[gram] = min(references_dictionary[gram], matchings[gram])

    precision = sum(matchings.values()) / ngram_sentence_length

    index = np.argmin([abs(len(word) - sentence_length) for
                       word in references])
    references_length = len(references[index])

    if sentence_length > references_length:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(references_length) / sentence_length)

    BLEU_score = BLEU * precision

    return BLEU_score
