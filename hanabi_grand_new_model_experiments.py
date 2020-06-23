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

# HANABI GRAND SCHEME
# only qvals
# only stack, discard, n_inf, n_fuse
# RSA update
# deck net? remove for now

# quite slow ...

# update needs to happen for everyone
# really doesn't make much sense if no private information in action phase

#################################################################################################################################
########################################################################

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

#################################################################################################################################

N_SUITS = 5
N_ACTIONS = 8 + 4 * (N_SUITS + 5)

#################################################################################################################################

all_cards = list(product(range(N_SUITS), (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)))

# public states of intentions
P = np.zeros((5, 20, 3))
P[:, :, :] = 1/3

# x = np.random.randint(0, 9, size=(20, 3))
# y = x.reshape((60, 1))
# y = y.reshape((20, 3))
# x-y
# x = np.random.randint(0, 9, size=(5, 3))
# x
# x = x.reshape((1, -1))
# x

alpha = 1.

def play_one_game (q_net_small=None, q_net_big=None, verbose=False, EPSILON=1):

    S = []
    A = []
    R = []
    F = []

    VA = [] # store valid actions for RSA update

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


    final_round = False
    moves_left = 5

    # S += [(copy(stack), copy(discarded), n_inf_tokens, n_fuse_tokens, P[player])]
    S += [(player, copy(stack), copy(discarded), n_inf_tokens, n_fuse_tokens, deepcopy(decks), np.copy(P[player]))]

    # PLAY

    while 1:

        r = 0

        if moves_left < 1:
            break

        if final_round:
            moves_left -= 1

        valid_actions = np.ones((N_ACTIONS,), dtype=bool)

        # !!
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

        VA += [valid_actions]

        ########################################################################

        # ******************************************************************** #
        # RSA update, I think (ignores context)

        if A:

            last_action = A[-1]
            last_valid_actions = VA[-2]

            last_state = S[-1]
            last_state = [last_state[1], last_state[2], last_state[3], last_state[4]]

            P1 = P[player]

            sx = np.zeros((20, 3, inp_size_small))
            sx[:, :, :] = encode_state_small ((player-1)%5, *last_state, P1) # not sure

            for c in range(20):

                for i in range(3):

                    P2 = np.copy(P1)
                    P2[c, :] = 0
                    P2[c, i] = 1

                    sx[c, i, -60:] = P2.reshape(-1)

            sx = sx.reshape((60, inp_size_small))

            U = q_net_small.predict(sx)
            U = np.exp(alpha * U)
            U *= last_valid_actions[np.newaxis, :]
            U /= np.sum(U, axis=-1)[:, np.newaxis]
            U = U[:, last_action]

            # print ("U", U)

            U = U.reshape((20, 3))
            U *= P1 # prior; should be different for every player but atm doesn't make a difference

            U /= np.sum(U, axis=-1)[:, np.newaxis]

            P[:, :, :] = U

            # print ("asd", (P[0]*100).astype('int'))

            # + + + + + + + + + + + + + + + + + + + + + + + + + + + #

            # get utilities of intentions
            # (though i still don't know what to input for P and how to get U("keep"))
            # what if an action is invalid? --- ignore for now but keep in mind
            # confused as to what utilities are for players if it's not their turn

            # s = np.zeros((5, inp_size_small))
            #
            # for p in range(5):
            #
            #     # s[p, :] = encode_state_small (p, *last_state, P[p])
            #     s[p, :] = encode_state_small (p, *last_state, P1)
            #
            # q = q_net_small.predict(s)
            # q = np.exp(alpha * q)
            # q /= np.sum(q, axis=-1)[:, np.newaxis]
            #
            # q = q[:, :8]
            #
            # q_play = q[:, :4].reshape((1, -1))
            # q_disc = q[:, 4:].reshape((1, -1))
            # q_keep = np.ones_like(q_play) - q_play - q_disc # dubious
            #
            # q = np.concatenate([q_play, q_disc, q_keep], axis=0)
            #
            # print ("q", q)

            # + + + + + + + + + + + + + + + + + + + + + + + + + + + #


        # ******************************************************************** #

        # full board state sample test

        # last_state = S[-1]
        # last_state = [last_state[1], last_state[2], last_state[3], last_state[4]]
        #
        # s = encode_state_small(player, *last_state, P[player])
        #
        # probs = o_net.predict(s.reshape((1, -1)))
        #
        # board_state = sample_board_state(np.squeeze(probs), stack, discarded)
        #
        # print ("rO:", board_state)

        # ******************************************************************** #

        actions = np.arange(N_ACTIONS)[valid_actions]

        state, _ = encode_state_big (player, stack, discarded, n_inf_tokens, n_fuse_tokens, decks, P[player])

        q = np.squeeze(q_net_big.predict(state.reshape(1, -1)))
        q = q[valid_actions]

        q = np.exp(alpha * q)
        q /= np.sum(q)

        # action = choice(actions[q == np.max(q)])
        action = choice(actions, p=q)

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
                    # r = -1
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

            n_inf_tokens -= 1

        if action < 8:

            P[:, player*4+(action%4), :] = 1/3

        player = (player + 1) % 5

        # S += [(copy(stack), copy(discarded), n_inf_tokens, n_fuse_tokens, P[player])]

        S += [(player, copy(stack), copy(discarded), n_inf_tokens, n_fuse_tokens, deepcopy(decks), np.copy(P[player]))]
        A += [action]
        R += [r]
        F += [0]


    # because 'break'
    A += [action]
    R += [r]
    F += [1]

    # print ("P", P)

    score = len(stack)

    if verbose: print (score)

    return S, A, R, F

S, A, R, F = play_one_game(q_net_small=q_net_small, q_net_big=q_net_big)

#################################################################################################################################

# get intention utility
# not sure what to input for P (for now: random or whatever)

# in hindsight non-player-specific utilities (qvals) just don't make any sense
# need some representation of whos turn it is
# for now: simply re-order P such that current player comes first (but leave P "outside" as is)
# alternatively there could be 5*48 = 240 actions ... but learning would be really slow

#################################################################################################################################

def sample_board_state (probs, stack, discarded):

    # given deck probs, gives random sample
    # need stack + discard pile to know which cards are available
    
    # this method might result in overall unlikely board states
    # sampling cards independently is not the same as sampling board states
    # but might do well enough for now

    n_available = np.zeros((N_SUITS*5,))

    for s in range(N_SUITS):
        n_available[s*5] = 3
        n_available[s*5+1:s*5+4] = 2
        n_available[s*5+4] = 1

    for (s, v) in stack + discarded:
        n_available[5*s+v-1] -= 1

    random_indices = np.random.choice(20, 20, replace=False)

    board_state = np.zeros((20, N_SUITS*5))

    for i in random_indices:

        probs[:, n_available==0] = 0
        probs /= np.sum(probs, axis=-1)[:, np.newaxis]

        k = choice(np.arange(N_SUITS*5), p=probs[i])
        board_state[i, k] = 1
        n_available[k] -= 1

    return board_state

# sample_O(None, None, None, None)

# probs = np.random.random((20, N_SUITS*5))
# probs /= np.sum(probs, axis=-1)[:, np.newaxis]
# stack = [(3, 1), (2, 1), (2, 2)]
# discarded = [(3, 3), (2, 2), (3, 2), (4, 1)]
#
# sample_O(20, probs, stack, discarded)

#################################################################################################################################

def decks2vec (decks):

    # alternative encoding

    d = np.zeros((20, N_SUITS*5))

    for i, deck in enumerate(decks):
        for j, (s, v) in enumerate(deck):
            if s != -1:
                d[4*i+j, 5 * s + v - 1] = 1

    return d

#################################################################################################################################

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

def encode_state_small (player, stack, discarded, n_inf_tokens, n_fuse_tokens, P):

    # re-order P such that own cards come first
    P = np.concatenate([P[player*4:(player+1)*4], P[:player*4], P[(player+1)*4:]])

    stack = stack2vec(stack)
    discarded = discarded2vec(discarded)
    ninf = one_hot(n_inf_tokens-1, 8)
    nfuse = one_hot(n_fuse_tokens-1, 3)

    state = np.concatenate([a.reshape(-1) for a in [stack, discarded, ninf, nfuse, P]], axis=0)

    return state

def encode_states_small (S):

    return [encode_state_small(player%5, stack, discarded, n_inf_tokens, n_fuse_tokens, P)[0] for player, (stack, discarded, n_inf_tokens, n_fuse_tokens, P) in enumerate(S)]

######################################################################

def encode_state_big (player, stack, discarded, n_inf_tokens, n_fuse_tokens, decks, P):

    # re-order P such that own cards come first
    P = np.concatenate([P[player*4:(player+1)*4], P[:player*4], P[(player+1)*4:]])

    decks = decks2vec(decks)
    dk = decks[4*player:4*(player+1)] # true deck
    decks = np.concatenate([decks[:4*player], decks[4*(player+1):]])

    stack = stack2vec(stack)
    discarded = discarded2vec(discarded)

    ninf = one_hot(n_inf_tokens-1, 8)
    nfuse = one_hot(n_fuse_tokens-1, 3)

    state = np.concatenate([a.reshape(-1) for a in [stack, discarded, ninf, nfuse, decks, P]], axis=0)

    return state, dk

def encode_states_big (S):

    return [encode_state_big(player%5, stack, discarded, n_inf_tokens, n_fuse_tokens, P)[0] for player, (stack, discarded, n_inf_tokens, n_fuse_tokens, P) in enumerate(S)]

######################################################################

inp_size_small = N_SUITS * 10 * (N_SUITS + 5) + (N_SUITS * 5 * (N_SUITS + 5)) + 8 + 3 + 60
inp_size_big = N_SUITS * 10 * (N_SUITS + 5) + (N_SUITS * 5 * (N_SUITS + 5)) + 8 + 3 + 60 + 16 * 5 * N_SUITS

######################################################################

def get_models ():

    inp_1 = Input(shape=(inp_size_small,))

    x = Dense(256, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp_1)

    x = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

    out_q_small = Dense(N_ACTIONS, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='linear', name='qval')(x)

    q_net_small = Model(inputs=inp_1, outputs=out_q_small)

    ######################################################

    inp_dk = Input(shape=(inp_size_big,))

    x = Dense(512, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp_dk)

    x = Dense(N_SUITS*5*4, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='linear')(x)

    x = Reshape((4, N_SUITS*5))(x)

    out_deck = Softmax(axis=-1)(x)

    deck_net = Model(inputs=inp_dk, outputs=out_deck)

    ######################################################

    inp_2 = Input(shape=(inp_size_big,))

    x = Dense(512, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp_2)

    x = Dense(256, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

    x_deck = deck_net(inp_2)
    x_deck = Flatten()(x_deck)

    x = Concatenate()([x_deck, x])

    x = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

    out_q_big = Dense(N_ACTIONS, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='linear', name='qval')(x)

    q_net_big = Model(inputs=inp_2, outputs=out_q_big)

    ######################################################

    inp_3 = Input(shape=(inp_size_small,))

    x = Dense(512, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp_3)

    x = Dense(256, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

    x = Dense(20*5*N_SUITS, kernel_initializer='he_normal', bias_initializer='zeros', activation='linear')(x)

    x = Reshape((20, N_SUITS*5))(x)

    out_o = Softmax(axis=-1)(x)

    o_net = Model(inputs=inp_3, outputs=out_o)

    ######################################################

    return q_net_small, q_net_big, deck_net, o_net

########################################################################

def _encode_state_small (S):

    s = encode_state_small(*S)

    return s.reshape(1, -1)

########################################################################

def _encode_state_big (S):

    s, dk = encode_state_big(*S)

    return s.reshape(1, -1), dk.reshape(1, 4, 5*N_SUITS)

########################################################################

def play_n_games (n, q_net_small=None, q_net_big=None, GAMMA=.999, EPS=1.):

    ALL = [play_one_game(q_net_small=q_net_small, q_net_big=q_net_big, EPSILON=EPS) for _ in range(n)]

    EPS *= .99

    _S = [S for S, _, _, _ in ALL]
    S = []
    for s in _S:
        S += s

    X_small = [_encode_state_small([s[0], s[1], s[2], s[3], s[4], s[6]]) for s in S]
    X_big = [_encode_state_big(s) for s in S]

    # get decks to predict full open board state

    X4O = np.concatenate([_encode_state_small([0, s[1], s[2], s[3], s[4], s[6]]) for s in S], axis=0)
    D4O = np.concatenate([decks2vec(s[-2]).reshape((1, 20, N_SUITS * 5)) for s in S], axis=0)

    # concatenate arrays

    STATES_small = np.concatenate(X_small, axis=0)
    STATES_big = np.concatenate([s for s, _ in X_big], axis=0)
    DECKS = np.concatenate([dk for _, dk in X_big], axis=0)

    # make yuge arrays

    ACTIONS = np.concatenate([A for _, A, _, _ in ALL])
    REWARDS = np.concatenate([R for _, _, R, _ in ALL])
    FINAL = np.concatenate([F for _, _, _, F in ALL]).astype(bool)

    # use the same for both?
    Q_small = q_net_small.predict(STATES_small)
    Q_big = q_net_big.predict(STATES_big)

    random_indices = np.random.choice(len(Q_small), len(Q_small), replace=False)

    for i in random_indices:

        Q_small[i, ACTIONS[i]] = REWARDS[i]
        Q_big[i, ACTIONS[i]] = REWARDS[i]

        if not FINAL[i]:

            Q_small[i, ACTIONS[i]] += GAMMA * np.max(Q_small[i+1])
            Q_big[i, ACTIONS[i]] += GAMMA * np.max(Q_big[i+1])

    return STATES_small, STATES_big, Q_small, Q_big, DECKS, X4O, D4O, sum(REWARDS) / n

########################################################################

def stupid_keras_memleak (q_net_small, q_net_big, deck_net, o_net):

    t1 = time()

    q_net_small.save_weights('/home/florian/FF_PROG/HANABINEW/q_net_small_weights.h5')
    q_net_big.save_weights('/home/florian/FF_PROG/HANABINEW/q_net_big_weights.h5')
    deck_net.save_weights('/home/florian/FF_PROG/HANABINEW/deck_net_weights.h5')
    o_net.save_weights('/home/florian/FF_PROG/HANABINEW/o_net_weights.h5')

    wqs = K.batch_get_value(getattr(q_net_small.optimizer, 'weights'))
    wqb = K.batch_get_value(getattr(q_net_big.optimizer, 'weights'))
    wdk = K.batch_get_value(getattr(deck_net.optimizer, 'weights'))
    wdo = K.batch_get_value(getattr(o_net.optimizer, 'weights'))

    with open('/home/florian/FF_PROG/HANABINEW/optiQs.pkl', 'wb') as f:
        pickle.dump(wqs, f)
    with open('/home/florian/FF_PROG/HANABINEW/optiQb.pkl', 'wb') as f:
        pickle.dump(wqb, f)
    with open('/home/florian/FF_PROG/HANABINEW/optiDk.pkl', 'wb') as f:
        pickle.dump(wdk, f)
    with open('/home/florian/FF_PROG/HANABINEW/optiO.pkl', 'wb') as f:
        pickle.dump(wdo, f)

    K.clear_session()

    q_net_small, q_net_big, deck_net, o_net = get_models()

    q_net_small.load_weights('/home/florian/FF_PROG/HANABINEW/q_net_small_weights.h5')
    q_net_big.load_weights('/home/florian/FF_PROG/HANABINEW/q_net_big_weights.h5')
    deck_net.load_weights('/home/florian/FF_PROG/HANABINEW/deck_net_weights.h5')

    o_net.load_weights('/home/florian/FF_PROG/HANABINEW/o_net_weights.h5')

    q_net_small.compile(loss='mse', optimizer=Adam(lr=.0005, clipvalue=2.))
    q_net_big.compile(loss='mse', optimizer=Adam(lr=.0005, clipvalue=2.))
    deck_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.0005))

    o_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.0005))


    with open('/home/florian/FF_PROG/HANABINEW/optiQs.pkl', 'rb') as f:
        wqs = pickle.load(f)
    with open('/home/florian/FF_PROG/HANABINEW/optiQb.pkl', 'rb') as f:
        wqb = pickle.load(f)
    with open('/home/florian/FF_PROG/HANABINEW/optiDk.pkl', 'rb') as f:
        wdk = pickle.load(f)
    with open('/home/florian/FF_PROG/HANABINEW/optiO.pkl', 'rb') as f:
        wdo = pickle.load(f)

    q_net_small.optimizer.set_weights(wqs)
    q_net_big.optimizer.set_weights(wqb)
    deck_net.optimizer.set_weights(wdk)

    o_net.optimizer.set_weights(wdo)

    return q_net_small, q_net_big, deck_net, o_net

########################################################################

LOG_R = []
GAMMA = .999

q_net_small, q_net_big, deck_net, o_net = get_models()

q_net_small.compile(loss='mse', optimizer=Adam(lr=.0005, clipvalue=2.))
q_net_big.compile(loss='mse', optimizer=Adam(lr=.0005, clipvalue=2.))
deck_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.0005))

o_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.0005))

for k in range(10000):

    S_small, S_big, Q_small, Q_big, Dk, X4O, D4O, Rw = play_n_games(100, q_net_small=q_net_small, q_net_big=q_net_big, GAMMA=GAMMA)

    random_indices = np.random.choice(len(Q_small), len(Q_small), replace=False)

    S_small = S_small[random_indices]
    S_big = S_big[random_indices]
    Dk = Dk[random_indices]
    Q_small = Q_small[random_indices]
    Q_big = Q_big[random_indices]
    X4O = X4O[random_indices]
    D4O = D4O[random_indices]

    print ("\nREWARD:", Rw)

    deck_net.trainable = False
    for layer in deck_net.layers:
        layer.trainable = False

    q_net_small.compile(loss='mse', optimizer=Adam(lr=.0005, clipvalue=2.))
    q_net_big.compile(loss='mse', optimizer=Adam(lr=.0005, clipvalue=2.))

    q_net_small.fit(S_small, Q_small, batch_size=16, verbose=(k%5==0))
    q_net_big.fit(S_big, Q_big, batch_size=16, verbose=(k%5==0))

    deck_net.trainable = True
    for layer in deck_net.layers:
        layer.trainable = True

    deck_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.0005))

    deck_net.fit(S_big, Dk, batch_size=16, verbose=(k%5==0))

    o_net.fit(X4O, D4O, batch_size=16, verbose=(k%5==0))

    LOG_R += [Rw]

    # need to save optimizer state
    q_net_small, q_net_big, deck_net, o_net = stupid_keras_memleak(q_net_small, q_net_big, deck_net, o_net)


plt.figure(figsize=(10, 10))
plt.plot(LOG_R, linewidth=3)
plt.show()

# e = .99
# EPSLOG = []
# for _ in range(len(LOG_R)):
#     EPSLOG += [e]
#     e *= .999
#
# plt.figure(figsize=(10, 10))
# # plt.plot(LOG_R)
# plt.plot(EPSLOG)
# plt.show()
#
# # q is randomized!
#
# plt.figure(figsize=(10, 10))
# plt.bar(range(48), Q[0])
# plt.show()
#
#
# # Q is randomized!
# plt.figure(figsize=(20, 20))
# # plt.imshow(np.repeat(S, 20, axis=0), cmap='inferno')
# plt.imshow(Q, cmap='inferno')
# # plt.axis('off')
# plt.show()
#
#
# for i in range(len(Q)):
#     plt.figure(figsize=(10, 10))
#     plt.bar(range(48), Q[i])
#     # plt.ylim((-5, 10))
#     # plt.axis('off')
#     plt.show()
#
#
# # 0.883284994052061
#
#
# S, A, R, F = play_one_game(None)

# S, A, R, F = play_one_game(policy_net)

# is the very last state not saved?
# and is the first state saved twice?
# no but player is 0 in both 1st and 2nd state
# player count off by one so it's showing the wrong card etc
# wtf is going on
# solved ---- need to do "deepcopy" for nested list

def make_game_readable (S, A, R, F):

    player = 0

    for s, a, r, f in zip(S, A, R, F):

        player, stack, discarded, n_inf_tokens, n_fuse_tokens, decks, P = s

        others = list(range(5))
        others.remove(player)

        state, td = encode_state_big(player, stack, discarded, n_inf_tokens, n_fuse_tokens, decks, P)

        q = q_net_big.predict(state.reshape(1, -1))

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

S, A, R, F = play_one_game(q_net_small=q_net_small, q_net_big=q_net_big)


make_game_readable(S, A, R, F)


def get_plot (S, A, R, F):

    probs = []
    vals = []

    player = 0

    for s, a, r, f in zip(S, A, R, F):

        others = list(range(5))
        others.remove(player)

        decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens = s

        state, plb = encode_state(player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)

        p = policy_net.predict(state.reshape(1, -1)).reshape(-1)
        q = q_net.predict(state.reshape(1, -1))[0].reshape(-1)

        probs += [p.reshape(1, -1)]
        vals += [q.reshape(1, -1)]

        player = (player + 1) % 5

    return np.concatenate(probs, axis=0), np.concatenate(vals, axis=0)

P, Q = get_plot (S, A, R, F)

plt.figure(figsize=(20, 20))
# plt.imshow(np.repeat(S, 20, axis=0), cmap='inferno')
plt.imshow(P, cmap='inferno')
# plt.axis('off')
plt.show()

# wq = q_net.get_weights()
#
# len(wq)
# wq[-4][-12:,:]
#
#
# # Q is randomized!
# plt.figure(figsize=(20, 20))
# # plt.imshow(np.repeat(S, 20, axis=0), cmap='inferno')
# plt.imshow(wq[-2], cmap='inferno')
# # plt.axis('off')
# plt.showzz
