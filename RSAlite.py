import pickle
from matplotlib import pyplot as plt
from copy import copy, deepcopy
import sys
from random import shuffle, randint, random
from numpy.random import choice
import numpy as np
from itertools import product, permutations
from time import time, sleep
from multiprocessing import Pool, cpu_count
from itertools import tee
from keras.layers import Dense, Input, Flatten, BatchNormalization, Add, Concatenate, Lambda, Reshape, Activation, Softmax, Multiply, Dot
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import binary_crossentropy
# import gym
from time import time

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

#################################################################################################################################

N_SUITS = 5

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

def n_available (player, stack, discarded, decks):

    n_available = np.zeros((N_SUITS*5,))

    for s in range(N_SUITS):
        n_available[s*5] = 3
        n_available[s*5+1:s*5+4] = 2
        n_available[s*5+4] = 1

    for (s, v) in stack + discarded:
        n_available[5*s+v-1] -= 1

    for i, deck in enumerate(decks):
        if i != player:
            for (s, v) in deck:
                if s >= 0:
                    n_available[5*s+v-1] -= 1

    return n_available

def nava_disc (nava, disclosed):

    # incorporates disc inf in nava

    NAVA = np.repeat(nava[np.newaxis], 4, axis=0)

    for i in range(4):

        for j in range(10):

            if disclosed[i, j] == -1:

                if j < N_SUITS:

                    NAVA[i, j*5:j*5+5] = 0

                else:
                    k = j - 5
                    NAVA[i, k::5] = 0

            if disclosed[i, j] == 1:

                if j < N_SUITS:

                    NAVA[i, :j*5] = 0
                    NAVA[i, j*5+5:] = 0

                else:
                    k = j - 5
                    _tmp = np.copy(NAVA[i, k::5])
                    NAVA[i] = 0
                    NAVA[i, k::5] = _tmp

    return NAVA

def sample_deck (NAVA, probs, tol=.2):

    # samples dumb

    # x = np.random.random(size=(4, 25))
    # x[x<probs] = 1
    # x[x<1] = 0
    #
    # return x

    logtol = - np.log(tol)
    # logtol = - np.log(1/20000)
    # logtol = - 11212
    #
    # p = np.copy(probs)
    # p[p>0] = np.log(p[p>0])
    # p = np.sum(np.mean(p, axis=-1))
    # logtol = - p

    # tol doesn't make sense like this
    # need to adapt to given probabilities
    # eg. if i don't know anything, all cards available
    # the probabilities will be smaller

    # NAVA is nava w disc inf

    DONE = False

    nn = np.sum(NAVA, axis=-1)

    ix = sorted(list(range(4)), key = lambda i : -nn[i])

    z = 0

    while not DONE:

        logp = 0

        _probs = np.copy(probs)
        _nava = np.copy(NAVA)

        z += 1
        # sys.stdout.write(str(z)+"       \r")
        # sys.stdout.flush()
        if z > 10:
            z = 0
            logtol *= 1.5
            print ("shite")
            # return False

        deck = []
        deckArray = np.zeros((4, N_SUITS * 5))

        for i in ix:

            if NAVA[i, 0] < 0:
                deck += [(-1, -1)]
                continue

            _probs[_nava==0] = 0
            s = np.sum(_probs, axis=-1)[:, np.newaxis]

            if s[i] == 0:
                break
                # repeat

            s[s==0] = 1
            _probs /= s

            k = choice(np.arange(25), p=_probs[i])

            logp -= np.log(_probs[i, k])

            s = k // 5
            v = (k % 5) + 1

            deck += [(s, v)]

            deckArray[i, k] = 1

            _nava[:, k] -= 1
            _nava[_nava<0] = 0


        DONE = len(deck) == 4# and logp < logtol

        # if logp>=logtol:
        #     print ("REJECT!", logp, logtol)

    return deckArray, deck

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

def deck_probs (n_available, disclosed):

    # disclosed for player

    nava = np.repeat(n_available[np.newaxis], 4, axis=0)

    for i in range(4):

        for j in range(10):

            if disclosed[i, j] == -1:

                if j < N_SUITS:

                    nava[i, j*5:j*5+5] = 0

                else:
                    k = j - 5
                    nava[i, k::5] = 0

            if disclosed[i, j] == 1:

                if j < N_SUITS:

                    nava[i, :j*5] = 0
                    nava[i, j*5+5:] = 0

                else:
                    k = j - 5
                    _tmp = np.copy(nava[i, k::5])
                    nava[i] = 0
                    nava[i, k::5] = _tmp

    nava1 = np.copy(nava)

    s = np.sum(nava, axis=0)
    s[s==0] = 1

    # nava *= nava1 / s[np.newaxis]
    nava /= s[np.newaxis]
    nava *= nava1

    s = np.sum(nava, axis=-1)
    s[s==0] = 1
    nava /= s[:, np.newaxis]


    return nava

#################################################################################################################################

def _sample_deck (args):
    dkArray, dk = sample_deck(*args)
    return dkArray, dk

n_workers = min(10, cpu_count())
pool = Pool(processes=n_workers)

def get_deck_samples (n, n_available, probs):

    probs = np.copy(probs)
    n_available = np.copy(n_available)

    X = list(pool.imap_unordered(_sample_deck, [(n_available, probs)]*n, min(100, n // n_workers)))

    xArray = [x for x, _ in X]
    x = [x for _, x in X]

    return xArray, x

# pool.close()
# pool.join()

N_SUITS = 5
N_ACTIONS = 8 + 4 * (N_SUITS + 5)

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

######################################################################

def fast_disc (disc):

    fd = np.ones((4, 25))
    for i in range(4):
        for s in range(5):
            for v in range(5):
                if disc[i, s] == -1:
                    fd[i, 5*s:5*s+5] = 0
                if disc[i, 4+v] == -1:
                    fd[i, v::5] = 0
                if disc[i, s] == 1:
                    fd[i, :5*s] = 0
                    fd[i, s*5+5:] = 0
                if disc[i, 4+v] == 1:
                    _tmp = fd[i, v::5]
                    fd[i] = 0
                    fd[i, v::5] = _tmp

    return fd

######################################################################

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

######################################################################

# does this even make sense, conceptually?

# because u barely differs
# decksamples sometimes are identical?

# is some information not integrated? if a card is not new but new information
# is available (new cards played, discarded), this should be taken into account
# what to choose as prior?

# might also be a problem that deck probs are not accurate

all_cards = list(product(range(N_SUITS), (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)))

psum = np.zeros(5,)

# data

# np.random.random(size=(5, 7), mean=(.01, .1))

def play_one_game (q_net=None, verbose=False, EPSILON=1, training=True):

    DATA = {}

    DATA['naive'] = []
    DATA['rsalite'] = []
    DATA['sampling'] = []
    DATA['mean'] = []

    DATA['U'] = []
    DATA['DECKP_0'] = []
    DATA['DECKP_0_NAIVE'] = []

    S = []
    A = []
    R = []
    F = []

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

    player = 0

    stack = []
    discarded = []

    disclosed = np.zeros((20, N_SUITS+5)) # 'negative' information

    final_round = False
    moves_left = 5

    S += [(player, copy(stack), copy(discarded), n_inf_tokens, n_fuse_tokens, deepcopy(decks), np.copy(disclosed))]

    DECKP = - np.ones((5, 4, N_SUITS*5))

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

        state = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)

        # nava = n_available(player, stack, discarded, decks)
        # deck_p = deck_probs(nava, disclosed[player*4:player*4+4])
        # need to exlude last player ...

        ########################################################################

        n_samples = 200

        if len(A) > 0:

            last_player, last_stack, last_discarded, last_n_inf_tokens, last_n_fuse_tokens, last_decks, last_disclosed = S[-1]

            for PLAYER in range(5):

                if PLAYER != last_player:

                    # init new cards

                    nava = n_available(PLAYER, stack, discarded, decks)
                    deck_p = deck_probs(nava, disclosed[PLAYER*4:PLAYER*4+4])

                    if PLAYER==0:
                        DATA['DECKP_0_NAIVE'] += [np.copy(deck_p)]

                    DECKP[PLAYER][DECKP[PLAYER]<0] = deck_p[DECKP[PLAYER]<0]

                    # not sure which nava ...

                    nava = n_available(PLAYER, stack, discarded, decks)
                    NAVA = nava_disc (nava, disclosed[PLAYER*4:PLAYER*4+4])

                    for i, (s, v) in enumerate(decks[PLAYER]):
                        if s < 0:
                            NAVA[i, :] = -7

                    deck_p = deck_probs(nava, disclosed[PLAYER*4:PLAYER*4+4])

                    # questionable

                    DECKP[PLAYER] *= deck_p
                    DECKP[PLAYER][NAVA==0] = 0
                    s = np.sum(DECKP[PLAYER], axis=-1)[:, np.newaxis]
                    s[s==0] = 1
                    DECKP[PLAYER] /= s

                    # print ("++++"*30)
                    # print (decks[PLAYER])
                    # print (NAVA)
                    # print (DECKP[PLAYER])
                    # print ("++++"*30)

                    # print ("S:", np.sum(deck_p[NAVA==0]))
                    # samplesA, samples = get_deck_samples(n_samples, NAVA, DECKP[PLAYER])
                    samplesA, samples = get_deck_samples(n_samples, NAVA, deck_p)
                    samplesA = np.stack(samplesA)

                    mean = np.mean(samplesA, axis=0)
                    # mean *= deck_p
                    s = np.sum(mean, axis=-1)[:, np.newaxis]
                    s[s==0] = 1
                    mean /= s

                    # print ("+"*100)
                    #
                    # print ("deckp:", deck_p)
                    # print ("mean:", mean)
                    # # print ("samples:")
                    # # for k in range(10):
                    # #     print (samplesA[k])
                    # print ("+"*100)
                    #


                    # print (np.sum(np.abs(mean-deck_p)))
                    #
                    # t1b = time()

                    last_state = encode_state(last_player, last_decks, last_stack, last_discarded, last_disclosed, last_n_inf_tokens, last_n_fuse_tokens)

                    states = np.repeat(last_state[np.newaxis, :], n_samples, axis=0)



                    last_decks2 = deepcopy(last_decks)
                    last_decks2[PLAYER] = []
                    last_nava = n_available (last_player, last_stack, last_discarded, last_decks2)
                    last_nava = np.repeat(last_nava[np.newaxis], 4, axis=0)
                    last_nava = np.repeat(last_nava[np.newaxis], n_samples, axis=0)

                    disc = disclosed[last_player*4:last_player*4+4]

                    fd = fast_disc (disc)

                    for i, dk in enumerate(samples):
                        for s, v in dk:
                            if s >= 0:
                                last_nava[i, :, 5*s+v-1] -= 1

                    last_nava *= fd[np.newaxis]
                    s = np.sum(last_nava, axis=1)
                    s[s==0] = 1
                    last_nava *= last_nava / s[:, np.newaxis, :]
                    s = np.sum(last_nava, axis=-1)
                    s[s==0] = 1
                    last_nava /= s[:, :, np.newaxis]
                    dp = np.copy(last_nava)

                    samplesEnc = np.zeros((n_samples, 4, 10))

                    for i in range(n_samples):

                        for j in range(4):

                            s, v = samples[i][j]

                            samplesEnc[i, j, s] = 1
                            samplesEnc[i, j, 4+v] = 1

                    samplesEnc = samplesEnc.reshape((n_samples, -1))

                    # encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens):

                    # ix = 25 * 10 + 50 * 10 + 3 + 8 + PLAYER * (4 * 10)
                    ix = PLAYER * (4 * 10)
                    states[:, ix:ix+40] = samplesEnc




                    u = q_net.predict([dp, states])

                    alpha = 1.
                    u = np.exp(alpha * u)
                    u *= last_valid_actions[np.newaxis, :]
                    u /= np.sum(u, axis=-1)[:, np.newaxis]
                    u = u[:, A[-1]]
                    u /= np.sum(u)

                    DATA['U'] += [u]

                    # dp = np.dot(u, samplesA.reshape((4, n_samples, N_SUITS * 5)))

                    # not really a dot product?
                    dp = np.sum(u[:, np.newaxis, np.newaxis] * samplesA, axis=0)


                    # print (np.sum(np.square(dp-dpX)))
                    # print (np.sum(np.square(dp[0]-dp[1])))
                    # print (np.sum(np.square(dp[0]-dp[2])))
                    # print (np.sum(np.square(dp[0]-dp[3])))
                    # print (np.sum(np.square(samplesA[:, 0]-samplesA[:, 1])))
                    # print (np.sum(np.square(samplesA[:, 0]-samplesA[:, 2])))
                    # print (np.sum(np.square(samplesA[:, 0]-samplesA[:, 3])))
                    # print ()

                    test = np.copy(dp)
                    test *= deck_p

                    s = np.sum(test, axis=-1)[:, np.newaxis]
                    s[s==0] = 0
                    test /= s

                    # sampling alone
                    sampling = np.copy(dp)
                    s = np.sum(sampling, axis=-1)[:, np.newaxis]
                    s[s==0] = 0
                    sampling /= s



                    # this doesn't make any sense because this way,
                    # new information is lost

                    # dp = dp1 * deck_p # DECKP[PLAYER]

                    # so there are no sampling accidents
                    dp = dp * DECKP[PLAYER] + 1e-5
                    s = np.sum(dp, axis=-1)[:, np.newaxis]
                    s[s==0] = 1
                    dp /= s
                    DECKP[PLAYER] = dp

                    # print (np.sum(np.abs(tmp-dp)))


                    if 1: #PLAYER == 0:

                        # in decknet ... was cce applied
                        # accross correct axis?

                        td = decks2vec2(decks)[PLAYER*4:PLAYER*4+4].astype('bool')

                        # print ("step")
                        # print ()
                        # print ("r", dp[td])
                        # print ("n", deck_p[td])
                        # print ("s", dp2[td])
                        # print ()

                        # rsalite = np.prod(dp[td]+1e-9)
                        # naive = np.prod(deck_p[td]+1e-9)
                        # sampling = np.prod(dp2[td]+1e-9) # sampling alone

                        # rsalite = np.sum(dp[td]+1e-9)
                        # naive = np.sum(deck_p[td]+1e-9)
                        # sampling = np.sum(dp2[td]+1e-9) # sampling alone


                        rsalite = - np.sum(np.log(dp[td]+1e-9))
                        naive = - np.sum(np.log(deck_p[td]+1e-9))
                        sampling = - np.sum(np.log(sampling[td]+1e-9)) # sampling alone
                        mean = - np.sum(np.log(mean[td]+1e-9)) # mean

                        DATA["rsalite"] += [rsalite]
                        DATA["naive"] += [naive]
                        DATA["sampling"] += [sampling]
                        DATA["mean"] += [mean]

                        # test = - np.sum(np.log(test[td]+1e-9)) # sampling alone
                        # test2 = - np.sum(np.log(test2[td]+1e-9)) # sampling alone
                        #
                        # RSALITE += [rsalite]
                        # NAIVE += [naive]
                        # # SAMPLING += [sampling]
                        # SAMPLING += [test2]

                    # psum[0] += np.sum(tmp[td])
                    # psum[1] += np.sum(deck_p[td])
                    # psum[2] += np.sum(dp1[td])
                    # psum[3] += np.sum(dp[td])
                    # psum[4] += np.sum(dp2[td])

                    # print ("-"*50)
                    # print (np.sum(tmp[td]))
                    # print (np.sum(deck_p[td]))
                    # print (np.sum(dp1[td]))
                    # print (np.sum(dp[td]))
                    # print (np.sum(dp2[td]))
                    # print ("-"*50)

            t1 = time()

            # print (t1-t0, t1b-t0b, t1c-t0c, t1d-t0d)
            # print (t1-t0, t1b-t0b, t1c-t0c)

        DATA['DECKP_0'] += [np.copy(DECKP[0])]

        ########################################################################

        # differs for player ... but that's okay, it's the correct player
        last_valid_actions = np.copy(valid_actions)

        # q = q_net.predict([DECKP[player][np.newaxis], state[np.newaxis]])
        # q = q_net.predict([DECKP[player][np.newaxis], state[np.newaxis]])
        nava = n_available (player, stack, discarded, decks)
        deck_p = deck_probs (nava, disclosed[4*player:4*player+4])
        state = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)
        # q = q_net.predict([DECKP[player][np.newaxis], state[np.newaxis]])
        q = q_net.predict([deck_p[np.newaxis], state[np.newaxis]])
        q = np.squeeze(q)

        q = q[valid_actions]
        p = np.exp(q)
        p /= np.sum(p)
        # action = actions[np.argmax(q)]
        action = choice(actions, p=p)

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

            DECKP[:, action%4, :] = -1

        player = (player + 1) % 5

        S += [(player, copy(stack), copy(discarded), n_inf_tokens, n_fuse_tokens, deepcopy(decks), np.copy(disclosed))]
        A += [action]
        R += [r]
        F += [0]

    # because 'break'
    A += [action]
    R += [r]
    F += [1]

    # print ("P", P)

    score = len(stack)


    return S, A, R, F, DATA

# _, _, _, _, RSALITE, SAMPLING, NAIVE, _, _ = play_one_game(q_net=q_net)

S, A, R, F, DATA = play_one_game(q_net=q_net)


rsalite_mean = np.mean(DATA['rsalite'])
sampling_mean = np.mean(DATA['sampling'])
mean_mean = np.mean(DATA['mean'])
naive_mean = np.mean(DATA['naive'])

plt.figure(figsize=(10, 10))
plt.title("#samples = 200; mean but samples from naive, sampling, rsalite, naive")
plt.ylim([0, 50])
plt.bar(range(4), [mean_mean, sampling_mean, rsalite_mean, naive_mean])
# plt.savefig("/Users/rianfroehlich/Desktop/pred200samplefromnaive.png")
plt.show()

rsalite_mean, sampling_mean, mean_mean, naive_mean



S[-1]
# multiprocessing! (though not a priority)
# S, _, _, _, _, _, _, DECKP_0, DECKP_0_NAIVE = play_one_game(q_net=q_net)
#
#
# for dp0, dp0n in zip(DECKP_0, DECKP_0_NAIVE):
#     # minus naivke
#     plt.imshow(dp0-dp0n, cmap="inferno", vmin=-1, vmax=1)
#     plt.show()
sum(RSALITE)
sum(SAMPLING)
sum(NAIVE)

R = []
S = []
N = []

for _ in range(3):

    print ("iter", _)

    try:

        _, _, _, _, RSALITE, SAMPLING, NAIVE, _, _ = play_one_game(q_net=q_net)

        R += RSALITE
        S += SAMPLING
        N += NAIVE

    except:

        continue

uu = DATA["U"][44]

# plt.figure(figsize=(10, 10))
# plt.bar(range(len(uu)), uu)
# plt.savefig("/Users/florianfroehlich/Desktop/uAcross.png")
# plt.show()


plt.figure(figsize=(10, 10))
plt.plot(RSALITE[::5], c="blue")
plt.plot(SAMPLING[::5], c="yellow")
plt.plot(NAIVE[::5], c="red")
plt.show()

mr = np.mean(R)
ms = np.mean(S)
mn = np.mean(N)

mr, ms, mn

plt.figure(figsize=(10, 10))
plt.bar([0, 1, 2], [ms, mr, mn])
# plt.savefig("/Users/florianfroehlich/Desktop/doesntwork200nodp_sfromdp.png")
plt.show()

# try: sample from deckp

S[-1]
psum

######################################################################

def make_game_readable (S, A, R, F):

    player = 0

    for s, a, r, f in zip(S, A, R, F):

        player, stack, discarded, n_inf_tokens, n_fuse_tokens, decks = s

        others = list(range(5))
        others.remove(player)

        state, td = encode_state(player, stack, discarded, n_inf_tokens, n_fuse_tokens, decks)

        q = q_net.predict(state.reshape(1, -1))

        w = N_SUITS * 5

        q = (100 * (q / np.sum(q))).astype(int).tolist()

        dk = deck_net.predict(state.reshape(1, -1))

        dk = (100 * dk).astype('int')

        print ("\n"+"*"*100)
        print ("PLAYER " + str(player+1) + "'s TURN\n")
        print ("DECKS:          ", decks)
        print ("DECKP:          ")
        print (dk)
        print ("STACK:          ", stack)
        print ("DISCARDED:      ", discarded)
        print ("FINAL:          ", f)
        # print ("DISCLOSED:      ", disclosed)
        print ("n_inf_tokens:   ", n_inf_tokens)
        print ("n_fuse_tokens:  ", n_fuse_tokens)
        # print ("len rest:       ", (N_SUITS * 10) - len(discarded) - len(stack) - (N_SUITS * 5)) # ??
        print ("len rest:       ", (N_SUITS * 10) - len(discarded) - len(stack) - 20) # ??
        # print ("playable:       ", plb)
        print ("play me true:  ", [int(valid_card(card, stack)) for card in decks[player]])
        print ("-"*100)
        print ("q:              ", q)
        print ("-"*100)
        print ("action:         ", a)#, "(q = " + str(q[a]) + ")")
        print ("reward:         ", r)

        # q = np.exp(2*q)
        # q /= np.sum(q)
        # print (q)


        if a < 4:
            print ("Play card", a, ":", decks[player][a])
        elif a < 8:
            print ("Discard card", a-4, ":", decks[player][a-4])
        else:

            p = (a - 8) // (N_SUITS + 5)
            p = others[p]

            k = (a - 8) % (N_SUITS + 5)

            if k < N_SUITS:
                inds = [i for i, (ss, _) in enumerate(decks[p]) if ss == k]
                print ("Hint: Player", (p+1), "suit", k, " --- inds:", inds)
            else:
                inds = [i for i, (_, vv) in enumerate(decks[p]) if vv == k-N_SUITS-1]
                print ("Hint: Player", (p+1), "value", k-N_SUITS-1, " --- inds:", inds)


        print ("\n" + "*"*100 + "\n")
