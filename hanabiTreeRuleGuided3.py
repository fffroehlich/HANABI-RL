from multiprocessing import Pool, cpu_count
from copy import copy, deepcopy
from random import shuffle, randint, random
from numpy.random import choice
import numpy as np
from itertools import product
import sys

#################################################################################################################################

from keras.layers import Dense, Input, Flatten, BatchNormalization, Add, Concatenate, Lambda, Reshape, Activation, Softmax, Multiply, Dot
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import binary_crossentropy

#################################################################################################################################

inp_1 = Input(shape=(4, 25,))
inp_2 = Input(shape=(1196,))

########################################################################

# playable / crucial / irrelevant

plb = Lambda(lambda x: x[:, 1121:1146])(inp_2)
irv = Lambda(lambda x: x[:, 1146:1171])(inp_2)
crc = Lambda(lambda x: x[:, 1171:1196])(inp_2)

########################################################################

x_d_plb = Dot(axes=-1)([inp_1, plb])
x_d_irv = Dot(axes=-1)([inp_1, irv])
x_d_crc = Dot(axes=-1)([inp_1, crc])

deck_p = Flatten()(inp_1)

x = Dense(256, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp_2)

x = Concatenate()([x, deck_p, x_d_plb, x_d_irv, x_d_crc])

x = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

out = Dense(48, kernel_initializer='zeros', bias_initializer='zeros', activation='linear')(x)

q_net = Model(inputs=[inp_1, inp_2], outputs=out)
q_net.load_weights("/oldqd_nomp_final.h5")

W = K.batch_get_value(getattr(q_net, 'weights'))

#################################################################################################################################

def pred (inp1, inp2, weights):

    inp_d = inp2[:, :-75]

    plb = inp2[:, -75:-50]
    irv = inp2[:, -50:-25]
    crc = inp2[:, -25:]

    x_d_plb = np.dot(inp1, plb.T)
    x_d_irv = np.dot(inp1, irv.T)
    x_d_crc = np.dot(inp1, crc.T)

    x = np.dot(inp2, weights[0]) + weights[1]
    x[x<0] = 0

    xd = inp1.reshape(-1)

    x = np.concatenate([np.squeeze(z) for z in (x, xd, x_d_plb, x_d_irv, x_d_crc)], axis=-1)

    x = np.dot(x, weights[2]) + weights[3]
    x[x<0] = 0

    q = np.dot(x, weights[4]) + weights[5]

    return q

def pred2 (inp1, inp2):

    return pred (inp1, inp2, W)

#################################################################################################################################

def get_rest (player, stack, discarded, decks):

    _rest = list(product(range(N_SUITS), (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)))

    for i, deck in enumerate(decks):
        if i != player:
            for card in deck:
                if card != (-1, -1):
                    _rest.remove(card)

    for card in stack + discarded:
        if card != (-1, -1):
            _rest.remove(card)

    return _rest

#################################################################################################################################

def decks2vec2 (decks):

    # alternative encoding

    d = np.zeros((20, N_SUITS*5))

    for i, deck in enumerate(decks):
        for j, (s, v) in enumerate(deck):
            if s != -1:
                d[4*i+j, 5 * s + v - 1] = 1

    return d

def decks2vec (decks):

    d = np.zeros((20, 10))
    i = 0

    for deck in decks:
        for s, v in deck:
            if s != -1:
                d[i, s] = 1
                d[i, 4+v] = 1
            i += 1

    return d

def one_hot(k, n):

    x = np.zeros(n,)
    x[k] = 1

    return x

def stack2vec (stack):

    d = np.zeros((N_SUITS*5, N_SUITS+5))

    for i, (s, v) in enumerate(stack):
        d[i, s] = 1
        d[i, N_SUITS-1+v] = 1

    return d

def discarded2vec (discarded):

    d = np.zeros((N_SUITS*10, N_SUITS+5))

    for i, (s, v) in enumerate(discarded):
        d[i, s] = 1
        d[i, N_SUITS-1+v] = 1

    return d

#################################################################################################################################

def encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens):

    all_cards_nd = list(product(range(5), range(1, 6)))

    plb = np.array([valid_card(card, stack) for card in all_cards_nd], dtype=int)
    irv = np.array([irrelevant_card(card, stack, discarded) for card in all_cards_nd], dtype=int)
    crc = np.array([crucial_card(card, stack, discarded) for card in all_cards_nd], dtype=int)

    true_decks = decks2vec2(decks)

    decks = decks2vec(decks)

    enc_disclosed = np.copy(disclosed)
    enc_disclosed = np.concatenate([enc_disclosed[player*4:player*4+4], enc_disclosed[:player*4], enc_disclosed[player*4+4:]])

    # awkward reordering
    decks = np.concatenate([decks[:player*4], decks[player*4+4:]])

    stack = stack2vec(stack)
    discarded = discarded2vec(discarded)
    ninf = one_hot(n_inf_tokens-1, 8)
    nfuse = one_hot(n_fuse_tokens-1, 3)

    state = np.concatenate([a.reshape(-1) for a in [decks, stack, discarded, ninf, nfuse, enc_disclosed, plb, irv, crc]], axis=0)

    # state = np.copy(decks[player*4:player*4+4]).reshape(-1)

    return state#, true_decks

#################################################################################################################################

def valid_card (card, stack):

    s, v = card

    if v == 1:
        if not stack:
            return True
        if not any(suit == s for suit, _ in stack):
            return True

    match = [value for suit, value in stack if suit == s]

    if not match:
        return False

    if max(match) + 1 == v:
        return True

    return False

def irrelevant_card (card, stack, discarded):

    if card in stack:
        return True

    s, v = card
    d = [vv for ss, vv in discarded if ss==s and vv<v]
    c = np.zeros(5,)
    for dd in d:
        c[dd-1] += 1
    if c[0] > 2 or any(c[1:] > 1):
        return True

    return False

def crucial_card (card, stack, discarded):

    s, v = card

    if v == 5:
        return True

    if irrelevant_card(card, stack, discarded):
        return False

    L = len([c for c in discarded if c == card])

    if v == 1:
        if L > 1:
            return True
        return False

    if L > 0:
        return True

    return False

#################################################################################################################################

def deck_probs (n_available, disclosed):

    # disclosed for player

    nava = np.repeat(n_available[np.newaxis], 4, axis=0)

    for i in range(4):

        for j in range(10):

            if disclosed[i, j] == -1:

                if j < N_SUITS:

                    nava[i, j*5:(j+1)*5] = 0

                else:

                    nava[i, j::5] = 0


            if disclosed[i, j] == 1:

                if j < N_SUITS:

                    nava[i, :j*5] = 0
                    nava[i, (j+1)*5:] = 0

                else:

                    nava[i] = 0
                    nava[j::5] = 1

    s = np.sum(nava, axis=0)
    s[s==0] = 1

    nava *= nava / s[np.newaxis]

    s = np.sum(nava, axis=-1)
    s[s==0] = 1
    nava /= s[:, np.newaxis]

    return nava

#################################################################################################################################

def max_still_achievable_score (discarded):

    c = [0] * 25

    for (s, v) in discarded:
        c[s*5+v-1] += 1

    msas = 25

    for s in range(5):
        for v in range(5):
            cc = c[5*s+v]
            if (v == 0 and cc > 2) or (0 < v < 4 and cc > 1) or (v > 3 and cc > 0):
                msas -= (5 - v)
                break

    return msas

#################################################################################################################################

def n_available (player, stack, discarded, decks):

    n_available = np.zeros((N_SUITS*5,))

    for s in range(N_SUITS):
        n_available[s*5] = 3
        n_available[s*5+1:s*5+4] = 2
        n_available[s*5+4] = 1

    for (s, v) in stack + discarded:
        if s != -1:
            n_available[5*s+v-1] -= 1

    for i, deck in enumerate(decks):
        if i != player:
            for (s, v) in deck:
                if s != -1:
                    n_available[5*s+v-1] -= 1

    return n_available

#################################################################################################################################

def sample_deck (probs, player, stack, discarded, decks):

    _probs = np.copy(probs)

    nava = n_available (player, stack, discarded, decks)

    random_indices = np.random.choice(4, 4, replace=False)

    deck = []
    deck_enc = np.zeros((4, N_SUITS * 5))

    for i in random_indices:

        _probs[:, nava==0] = 0
        s = np.sum(_probs, axis=-1)[:, np.newaxis]

        if s[i] == 0:
            deck += [(-1, -1)]
            continue

        s[s==0] = 1
        _probs /= s

        k = choice(np.arange(N_SUITS*5), p=_probs[i])

        s = k // 5
        v = (k % 5) + 1

        deck += [(s, v)]

        deck_enc[i, k] = 1

        nava[k] -= 1

    return deck, deck_enc

#################################################################################################################################

N_SUITS = 5
N_ACTIONS = 8 + 4 * (N_SUITS + 5)

#################################################################################################################################

all_cards = list(product(range(N_SUITS), (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)))

#################################################################################################################################

# # random
# def rollout (action, player, decks, stack, discarded, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps):
#
#     # q guided
#     # player : current player
#
#     _decks = deepcopy(decks)
#     _stack = copy(stack)
#     _discarded = copy(discarded)
#
#     _rest = get_rest (player, stack, discarded, decks)
#
#     shuffle(_rest)
#
#     ########################################################
#
#     _player = player
#
#     # PLAY
#     R = 0
#
#     step = 0
#
#     while 1:
#
#         r = 0
#
#         if moves_left < 1:
#             break
#
#         if final_round:
#             moves_left -= 1
#
#         valid_actions = np.ones((N_ACTIONS,), dtype=bool)
#
#         for i, (s, v) in enumerate(_decks[_player]):
#
#             if s < 0:
#
#                 valid_actions[i] = 0
#                 valid_actions[4+i] = 0
#
#         others = list(range(5))
#         others.remove(_player)
#
#         if n_inf_tokens < 1:
#
#             valid_actions[8:] = 0
#
#         else:
#
#             sv = 5 + N_SUITS
#
#             for i in range(N_ACTIONS-8):
#
#                 if i%sv < N_SUITS:
#
#                     valid_actions[8+i] = int(any(i%sv == ss for ss, _ in _decks[others[i // (5+N_SUITS)]]))
#
#                 else:
#
#                     valid_actions[8+i] = int(any((i%sv)-N_SUITS+1 == vv for _, vv in _decks[others[i // (5+N_SUITS)]]))
#
#         ########################################################################
#
#         actions = np.arange(N_ACTIONS)[valid_actions]
#
#         if step > 0:
#
#
#             action = choice(actions)
#
#         ########################################################################
#
#         if action < 4:
#
#             # play card w index action
#
#             if valid_card (_decks[player][action], _stack):
#
#                 r = 1
#
#                 _stack += [_decks[player][action]]
#
#                 if _rest:
#                     _decks[player][action] = _rest.pop()
#                 else:
#                     final_round = True
#                     _decks[player][action] = (-1, -1)
#
#                 if _stack[-1][1] == N_SUITS:
#                     # completing a suit retrieves one information token
#                     n_inf_tokens = min(n_inf_tokens + 1, 8)
#
#                 if not _rest:
#                     final_round = True
#
#             else:
#
#                 _discarded += [_decks[player][action]]
#
#                 if _rest:
#                     _decks[player][action] = _rest.pop()
#                 else:
#                     final_round = True
#                     _decks[player][action] = (-1, -1)
#
#                 # cause of death: old age
#                 n_fuse_tokens -= 1
#
#                 if n_fuse_tokens <= 0:
#                     r = -1
#                     break
#
#                 if not _rest:
#                     final_round = True
#
#         elif action < 8:
#
#             # discard card w index action-4
#
#             _discarded += [_decks[player][action%4]]
#
#             if _rest:
#                 _decks[player][action%4] = _rest.pop()
#             else:
#                 final_round = True
#                 _decks[player][action%4] = (-1, -1)
#
#             n_inf_tokens = min(n_inf_tokens + 1, 8)
#
#             if not _rest:
#                 final_round = True
#
#         else:
#
#             # give hint
#
#             if n_inf_tokens < 1:
#                 print ("INVALID ACTION")
#
#
#             n_inf_tokens -= 1
#
#         step += 1
#         R += r
#         _player = (_player + 1) % 5
#
#         if step > max_steps:
#
#             break
#
#     return max_still_achievable_score(_discarded) + len(_stack) + .1 * n_inf_tokens + 5 * n_fuse_tokens

# q_guided
# def rollout (action, player, decks, stack, discarded, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps):
#
#     # q guided
#     # player : current player
#
#     _decks = deepcopy(decks)
#     _stack = copy(stack)
#     _discarded = copy(discarded)
#     _disclosed = np.zeros((20, 10))
#
#     _rest = get_rest (player, stack, discarded, decks)
#
#     shuffle(_rest)
#
#     ########################################################
#
#     _player = player
#
#     # PLAY
#     R = 0
#
#     step = 0
#
#     while 1:
#
#         r = 0
#
#         if moves_left < 1:
#             break
#
#         if final_round:
#             moves_left -= 1
#
#         valid_actions = np.ones((N_ACTIONS,), dtype=bool)
#
#         for i, (s, v) in enumerate(_decks[_player]):
#
#             if s < 0:
#
#                 valid_actions[i] = 0
#                 valid_actions[4+i] = 0
#
#         others = list(range(5))
#         others.remove(_player)
#
#         if n_inf_tokens < 1:
#
#             valid_actions[8:] = 0
#
#         else:
#
#             sv = 5 + N_SUITS
#
#             for i in range(N_ACTIONS-8):
#
#                 if i%sv < N_SUITS:
#
#                     valid_actions[8+i] = int(any(i%sv == ss for ss, _ in _decks[others[i // (5+N_SUITS)]]))
#
#                 else:
#
#                     valid_actions[8+i] = int(any((i%sv)-N_SUITS+1 == vv for _, vv in _decks[others[i // (5+N_SUITS)]]))
#
#         ########################################################################
#
#         actions = np.arange(N_ACTIONS)[valid_actions]
#
#         if step > 0:
#
#             _nava = n_available (_player, _stack, _discarded, _decks)
#             _dp = deck_probs (_nava, _disclosed)
#
#             _state = encode_state (_player, _decks, _stack, _discarded, _disclosed, n_inf_tokens, n_fuse_tokens)
#
#             q = pred2(_dp[np.newaxis], _state[np.newaxis])
#             q = q[valid_actions]
#
#             action = actions[np.argmax(q)]
#
#             # all_cards_nd = list(product(range(5), range(1, 6)))
#             #
#             # plb = [valid_card(card, _stack) for card in all_cards_nd]
#             #
#             # plb = np.dot(_dp, plb)
#             #
#             # plb *= valid_actions[:4]
#             #
#             # if np.max(plb) > .7:
#             #
#             #     action = np.argmax(plb)
#             #
#             # elif n_inf_tokens > 0:
#             #
#             #     action = choice(np.arange(8, 48)[valid_actions[8:]])
#             #
#             # else:
#             #
#             #     irv = [irrelevant_card(card, _stack, _discarded) for card in all_cards_nd]
#             #     irv = np.dot(_dp, irv)
#             #     irv *= valid_actions[4:8]
#             #
#             #     action = np.argmax(irv) + 4
#
#             #action = choice(actions)
#
#         ########################################################################
#
#         if action < 4:
#
#             # play card w index action
#
#             if valid_card (_decks[player][action], _stack):
#
#                 r = 1
#
#                 _stack += [_decks[player][action]]
#
#                 if _rest:
#                     _decks[player][action] = _rest.pop()
#                 else:
#                     final_round = True
#                     _decks[player][action] = (-1, -1)
#
#                 if _stack[-1][1] == N_SUITS:
#                     # completing a suit retrieves one information token
#                     n_inf_tokens = min(n_inf_tokens + 1, 8)
#
#                 if not _rest:
#                     final_round = True
#
#             else:
#
#                 _discarded += [_decks[player][action]]
#
#                 if _rest:
#                     _decks[player][action] = _rest.pop()
#                 else:
#                     final_round = True
#                     _decks[player][action] = (-1, -1)
#
#                 # cause of death: old age
#                 n_fuse_tokens -= 1
#
#                 if n_fuse_tokens <= 0:
#                     r = -1
#                     break
#
#                 if not _rest:
#                     final_round = True
#
#         elif action < 8:
#
#             # discard card w index action-4
#
#             _discarded += [_decks[player][action%4]]
#
#             if _rest:
#                 _decks[player][action%4] = _rest.pop()
#             else:
#                 final_round = True
#                 _decks[player][action%4] = (-1, -1)
#
#             n_inf_tokens = min(n_inf_tokens + 1, 8)
#
#             if not _rest:
#                 final_round = True
#
#         else:
#
#             # give hint
#
#             if n_inf_tokens < 1:
#                 print ("INVALID ACTION")
#
#             p = (action - 8) // 10
#             p = others[p]
#
#             k = (action - 8) % 10
#
#             if k < 5:
#                 inds = [i for i, (ss, _) in enumerate(_decks[p]) if ss == k]
#             else:
#                 inds = [i for i, (_, vv) in enumerate(_decks[p]) if vv == k-4]
#
#
#             _disclosed[4*p:4*p+4, k] = -1
#
#             for i in inds:
#                 _disclosed[4*p+i, k] = 1
#
#
#             n_inf_tokens -= 1
#
#         if action < 8:
#
#             _disclosed[player*4+(action%4), :] = 0
#
#
#         step += 1
#         R += r
#         _player = (_player + 1) % 5
#
#         if step > max_steps:
#
#             break
#
#     score = len(stack)
#
#     # return max_still_achievable_score(_discarded) + len(_stack) + n_inf_tokens + n_fuse_tokens
#
#     _nava = n_available (_player, _stack, _discarded, _decks)
#     _dp = deck_probs (_nava, _disclosed)
#     _state = encode_state (_player, _decks, _stack, _discarded, _disclosed, n_inf_tokens, n_fuse_tokens)
#     q = pred2(_dp[np.newaxis], _state[np.newaxis])
#     v = np.mean(q) # (or max?)
#
#     return np.max(q)

# this one is rule guided
def rollout (action, player, decks, stack, discarded, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps):

    # player : current player

    _decks = deepcopy(decks)
    _stack = copy(stack)
    _discarded = copy(discarded)
    _disclosed = np.zeros((20, 10))

    _rest = get_rest (player, stack, discarded, decks)

    shuffle(_rest)

    ########################################################

    _player = player

    # PLAY
    R = 0

    step = 0

    while 1:

        r = 0

        if moves_left < 1:
            break

        if final_round:
            moves_left -= 1

        valid_actions = np.ones((N_ACTIONS,), dtype=bool)

        for i, (s, v) in enumerate(_decks[_player]):

            if s < 0:

                valid_actions[i] = 0
                valid_actions[4+i] = 0

        others = list(range(5))
        others.remove(_player)

        if n_inf_tokens < 1:

            valid_actions[8:] = 0

        else:

            sv = 5 + N_SUITS

            for i in range(N_ACTIONS-8):

                if i%sv < N_SUITS:

                    valid_actions[8+i] = int(any(i%sv == ss for ss, _ in _decks[others[i // (5+N_SUITS)]]))

                else:

                    valid_actions[8+i] = int(any((i%sv)-N_SUITS+1 == vv for _, vv in _decks[others[i // (5+N_SUITS)]]))

        ########################################################################

        actions = np.arange(N_ACTIONS)[valid_actions]

        if step > 0:

            _nava = n_available (_player, _stack, _discarded, _decks)
            _dp = deck_probs (_nava, _disclosed)

            all_cards_nd = list(product(range(5), range(1, 6)))

            plb = [valid_card(card, _stack) for card in all_cards_nd]

            plb = np.dot(_dp, plb)

            plb *= valid_actions[:4]

            if np.max(plb) > .7:

                action = np.argmax(plb)

            elif n_inf_tokens > 0:

                action = choice(np.arange(8, 48)[valid_actions[8:]])

            else:

                irv = [irrelevant_card(card, _stack, _discarded) for card in all_cards_nd]
                irv = np.dot(_dp, irv)
                irv *= valid_actions[4:8]

                action = np.argmax(irv) + 4

            #action = choice(actions)

        ########################################################################

        if action < 4:

            # play card w index action

            if valid_card (_decks[player][action], _stack):

                r = 1

                _stack += [_decks[player][action]]

                if _rest:
                    _decks[player][action] = _rest.pop()
                else:
                    final_round = True
                    _decks[player][action] = (-1, -1)

                if _stack[-1][1] == N_SUITS:
                    # completing a suit retrieves one information token
                    n_inf_tokens = min(n_inf_tokens + 1, 8)

                if not _rest:
                    final_round = True

            else:

                _discarded += [_decks[player][action]]

                if _rest:
                    _decks[player][action] = _rest.pop()
                else:
                    final_round = True
                    _decks[player][action] = (-1, -1)

                # cause of death: old age
                n_fuse_tokens -= 1

                if n_fuse_tokens <= 0:
                    r = -1
                    break

                if not _rest:
                    final_round = True

        elif action < 8:

            # discard card w index action-4

            _discarded += [_decks[player][action%4]]

            if _rest:
                _decks[player][action%4] = _rest.pop()
            else:
                final_round = True
                _decks[player][action%4] = (-1, -1)

            n_inf_tokens = min(n_inf_tokens + 1, 8)

            if not _rest:
                final_round = True

        else:

            # give hint

            if n_inf_tokens < 1:
                print ("INVALID ACTION")

            p = (action - 8) // 10
            p = others[p]

            k = (action - 8) % 10

            if k < 5:
                inds = [i for i, (ss, _) in enumerate(_decks[p]) if ss == k]
            else:
                inds = [i for i, (_, vv) in enumerate(_decks[p]) if vv == k-4]


            _disclosed[4*p:4*p+4, k] = -1

            for i in inds:
                _disclosed[4*p+i, k] = 1


            n_inf_tokens -= 1

        if action < 8:

            _disclosed[player*4+(action%4), :] = 0


        step += 1
        R += r
        _player = (_player + 1) % 5

        if step > max_steps:

            break

    score = len(stack)

    return R
#
# def rollout (action, player, decks, stack, discarded, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps):
#
#     # player : current player
#
#     _decks = deepcopy(decks)
#     _stack = copy(stack)
#     _discarded = copy(discarded)
#
#     _rest = get_rest (player, stack, discarded, decks)
#
#     shuffle(_rest)
#
#     ########################################################
#
#     _player = player
#
#     # PLAY
#     R = 0
#
#     step = 0
#
#     while 1:
#
#         r = 0
#
#         if moves_left < 1:
#             break
#
#         if final_round:
#             moves_left -= 1
#
#         valid_actions = np.ones((N_ACTIONS,), dtype=bool)
#
#         for i, (s, v) in enumerate(_decks[_player]):
#
#             if s < 0:
#
#                 valid_actions[i] = 0
#                 valid_actions[4+i] = 0
#
#         others = list(range(5))
#         others.remove(_player)
#
#         if n_inf_tokens < 1:
#
#             valid_actions[8:] = 0
#
#         else:
#
#             sv = 5 + N_SUITS
#
#             for i in range(N_ACTIONS-8):
#
#                 if i%sv < N_SUITS:
#
#                     valid_actions[8+i] = int(any(i%sv == ss for ss, _ in _decks[others[i // (5+N_SUITS)]]))
#
#                 else:
#
#                     valid_actions[8+i] = int(any((i%sv)-N_SUITS+1 == vv for _, vv in _decks[others[i // (5+N_SUITS)]]))
#
#         ########################################################################
#
#         actions = np.arange(N_ACTIONS)[valid_actions]
#
#         if step > 0:
#
#             action = choice(actions)
#
#         ########################################################################
#
#         if action < 4:
#
#             # play card w index action
#
#             if valid_card (_decks[player][action], _stack):
#
#                 r = 1
#
#                 _stack += [_decks[player][action]]
#
#                 if _rest:
#                     _decks[player][action] = _rest.pop()
#                 else:
#                     final_round = True
#                     _decks[player][action] = (-1, -1)
#
#                 if _stack[-1][1] == N_SUITS:
#                     # completing a suit retrieves one information token
#                     n_inf_tokens = min(n_inf_tokens + 1, 8)
#
#                 if not _rest:
#                     final_round = True
#
#             else:
#
#                 _discarded += [_decks[player][action]]
#
#                 if _rest:
#                     _decks[player][action] = _rest.pop()
#                 else:
#                     final_round = True
#                     _decks[player][action] = (-1, -1)
#
#                 # cause of death: old age
#                 n_fuse_tokens -= 1
#
#                 if n_fuse_tokens <= 0:
#                     r = -1
#                     break
#
#                 if not _rest:
#                     final_round = True
#
#         elif action < 8:
#
#             # discard card w index action-4
#
#             _discarded += [_decks[player][action%4]]
#
#             if _rest:
#                 _decks[player][action%4] = _rest.pop()
#             else:
#                 final_round = True
#                 _decks[player][action%4] = (-1, -1)
#
#             n_inf_tokens = min(n_inf_tokens + 1, 8)
#
#             if not _rest:
#                 final_round = True
#
#         else:
#
#             # give hint
#
#             if n_inf_tokens < 1:
#                 print ("INVALID ACTION")
#
#             n_inf_tokens -= 1
#
#         step += 1
#         R += r
#         _player = (_player + 1) % 5
#
#         if step > max_steps:
#
#             break
#
#     score = len(stack)
#
#     return R

#################################################################################################################################

n_workers = min(10, cpu_count())
pool = Pool(processes=n_workers)

def sample_rollout (P, action, player, stack, discarded, decks, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps):

    smp_deck, deck_enc = sample_deck (P, player, stack, discarded, decks)
    _decks = copy(decks)
    _decks[player] = smp_deck

    p = np.prod(P[deck_enc.astype("bool")])

    return p * rollout(action, player, _decks, stack, discarded, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps)

def _sample_rollout (args):

    return sample_rollout(*args)

def get_vals (n, P, action, player, stack, discarded, decks, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps):

    X = list(pool.imap_unordered(_sample_rollout, [(P, action, player, stack, discarded, decks, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps)]*n, min(100, n // n_workers)))

    return X

#################################################################################################################################

def play_one_game ():

    S = []
    A = []
    R = []
    F = []
    D = []

    all_cards = list(product(range(N_SUITS), (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)))
    shuffle(all_cards)

    i = 0
    decks = []
    for _ in range(5):
        decks += [copy(all_cards[i:i+4])]
        i += 4
    rest = copy(all_cards[i:])

    n_fuse_tokens = 3
    n_inf_tokens = 8

    disclosed = np.zeros((20, N_SUITS+5))

    player = 0

    stack = []
    discarded = []

    final_round = False
    moves_left = 5

    # PLAY

    while 1:

        r = 0

        if moves_left < 1:
            break

        if final_round:
            moves_left -= 1

        valid_actions = np.ones((N_ACTIONS,), dtype=bool)


        for i, (s, v) in enumerate(decks[player]):

            if s < 0:

                valid_actions[i] = 0
                valid_actions[4+i] = 0

        others = list(range(5))
        others.remove(player)

        if n_inf_tokens < 1:

            valid_actions[8:] = 0

        else:

            sv = 5 + N_SUITS

            for i in range(N_ACTIONS-8):

                if i%sv < N_SUITS:

                    valid_actions[8+i] = int(any(i%sv == ss for ss, _ in decks[others[i // (5+N_SUITS)]]))

                else:

                    valid_actions[8+i] = int(any((i%sv)-N_SUITS+1 == vv for _, vv in decks[others[i // (5+N_SUITS)]]))

        ########################################################################


        actions = np.arange(N_ACTIONS)[valid_actions]

        nava = n_available (player, stack, discarded, decks)
        deck_p = deck_probs (nava, disclosed)

        state = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)

        q = q_net.predict([deck_p[np.newaxis], state[np.newaxis]])
        q = np.squeeze(q)[valid_actions]

        # action = actions[np.argmax(q)]

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        n_samples = 100
        max_steps = 3
        n_cand = 5
        print ("step")
        qsorted = sorted(q)
        qcand = q[q>=qsorted[-n_cand]]
        acand = actions[q>=qsorted[-n_cand]]

        for i in range(n_cand):
            print ("i", i)

            # qcand[i] += sum(get_vals (n_samples, deck_p, acand[i], player, stack, discarded, decks, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps))
            qcand[i] += sum(get_vals (n_samples, deck_p, acand[i], player, stack, discarded, decks, n_inf_tokens, n_fuse_tokens, final_round, moves_left, max_steps))

        action = acand[np.argmax(qcand)]

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        ########################################################################

        if action < 4:

            # play card w index action

            if valid_card (decks[player][action], stack):

                r = 1

                stack += [decks[player][action]]

                if rest:
                    decks[player][action] = rest.pop()
                else:
                    final_round = True
                    decks[player][action] = (-1, -1)

                if stack[-1][1] == N_SUITS:
                    # completing a suit retrieves one information token
                    n_inf_tokens = min(n_inf_tokens + 1, 8)

                if not rest:
                    final_round = True

            else:

                discarded += [decks[player][action]]

                if rest:
                    decks[player][action] = rest.pop()
                else:
                    final_round = True
                    decks[player][action] = (-1, -1)

                # cause of death: old age
                n_fuse_tokens -= 1

                if n_fuse_tokens <= 0:
                    r = -1
                    break

                if not rest:
                    final_round = True

        elif action < 8:

            # discard card w index action-4

            discarded += [decks[player][action%4]]

            if rest:
                decks[player][action%4] = rest.pop()
            else:
                final_round = True
                decks[player][action%4] = (-1, -1)

            n_inf_tokens = min(n_inf_tokens + 1, 8)

            if not rest:
                final_round = True

        else:

            # give hint

            if n_inf_tokens < 1:
                print ("INVALID ACTION")

            p = (action - 8) // 10
            p = others[p]

            k = (action - 8) % 10

            if k < 5:
                inds = [i for i, (ss, _) in enumerate(decks[p]) if ss == k]
            else:
                inds = [i for i, (_, vv) in enumerate(decks[p]) if vv == k-4]


            disclosed[4*p:4*p+4, k] = -1

            for i in inds:
                disclosed[4*p+i, k] = 1


            n_inf_tokens -= 1

        if action < 8:

            disclosed[player*4+(action%4), :] = 0

        player = (player + 1) % 5

        # S += [(copy(stack), copy(discarded), n_inf_tokens, n_fuse_tokens, P[player])]

        S += [(player, copy(stack), copy(discarded), n_inf_tokens, n_fuse_tokens, deepcopy(decks))]
        A += [action]
        R += [r]
        F += [0]
        D += [deck_p]


    # because 'break'
    A += [action]
    R += [r]
    F += [1]
    D += [deck_p]

    # print ("P", P)

    score = len(stack)

    # return score, sum(R), stack
    return S, A, R, F, D

#################################################################################################################################

# if __name__ == "__main__":

S, _, _, _, _ = play_one_game()
scores = []
for iii in range(1):
    scores += [len(S[-1][1])]

np.mean(scores)
