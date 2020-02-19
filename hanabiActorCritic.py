from matplotlib import pyplot as plt
from copy import copy, deepcopy
import sys
from random import shuffle, randint, random
from numpy.random import choice
import numpy as np
from itertools import product
from time import time, sleep
from multiprocessing import Pool, cpu_count
from itertools import tee
from keras.layers import Dense, Input, Flatten, BatchNormalization, Add, Concatenate, Lambda, Reshape, Activation, Softmax, Multiply, Dot
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import binary_crossentropy
import gym

#################################################################################################################################
#################################################################################################################################

# ADVANTAGE ACTOR CRITIC

def policy_gradient_loss (y_true, y_pred):

    # y_true contains values (advantages etc.) at index action,
    # zeros elsewhere

    y_pred = K.clip(y_pred, 1e-9, 1-1e-9)

    return - K.sum(K.log(y_pred) * y_true)

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

# for now: always 5 players, always 4 cards

all_cards = list(product(range(5), (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)))

# actions: 4 + 4 + n_players * 4 * 2

def play_one_game (net=None, verbose=False):

    S = []
    A = []
    R = []
    F = []

    all_cards = list(product(range(5), (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)))
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

    disclosed = np.zeros((20, 10)) # 'negative' information

    final_round = False
    moves_left = 5

    S += [(deepcopy(decks), copy(stack), copy(discarded), np.copy(disclosed), n_inf_tokens, n_fuse_tokens)]

    # PLAY

    while 1:

        r = 0

        if moves_left < 1:
            break

        if final_round:
            moves_left -= 1

        valid_actions = np.ones((48,), dtype=bool)
        others = list(range(5))
        others.remove(player)

        if n_inf_tokens < 1:

            valid_actions[8:] = 0

        else:

            for i in range(40):
                if i%10 < 5:
                    valid_actions[8+i] = int(any(i%10 == ss for ss, _ in decks[others[i // 10]]))
                else:
                    valid_actions[8+i] = int(any((i%10)-4 == vv for _, vv in decks[others[i // 10]]))


        actions = np.arange(48)[valid_actions]

        # if random() < EPSILON:
        #
        #     action = choice(actions)
        #
        # else:
        #
        #     state, _ = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)
        #
        #     q = np.squeeze(q_net.predict(state.reshape(1, -1)))
        #     q = q[valid_actions]
        #
        #     action = choice(actions[q == np.max(q)])

        # choose action using policy head

        state, _ = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)

        p = np.squeeze(net.predict(state.reshape(1, -1)))

        p = p[valid_actions]
        p /= np.sum(p)

        # # explore
        # p += .01
        # p /= np.sum(p)

        action = choice(actions, p=p)

        if action < 4:

            # play card w index action

            if valid_card (decks[player][action], stack):

                r = 1

                stack += [decks[player][action]]

                if final_round:
                    decks[player][action] = (-1, -1)
                else:
                    decks[player][action] = rest.pop()

                if stack[-1][1] == 5:
                    # completing a suit retrieves one information token
                    # idiot
                    # n_inf_tokens = (n_inf_tokens + 1) % 8
                    n_inf_tokens = min(n_inf_tokens + 1, 8)

                if not rest:
                    final_round = True

            else:

                discarded += [decks[player][action]]

                if final_round:
                    decks[player][action] = (-1, -1)
                else:
                    decks[player][action] = rest.pop()

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

            if final_round:
                decks[player][action%4] = (-1, -1)
            else:
                decks[player][action%4] = rest.pop()

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

        # don't know anything about new card
        if action < 8:
            disclosed[player*4+(action%4), :] = 0

        S += [(deepcopy(decks), copy(stack), copy(discarded), np.copy(disclosed), n_inf_tokens, n_fuse_tokens)]
        A += [action]
        R += [r]
        F += [0]

        player = (player + 1) % 5

    # because 'break'
    A += [action]
    R += [r]
    F += [1]

    score = len(stack)

    if verbose: print (score)

    return S, A, R, F

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

def one_hot(k, n):

    x = np.zeros(n,)
    x[k] = 1

    return x

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

def decks2vec2 (decks):

    # alternative encoding

    d = np.zeros((20, 25))

    for i, deck in enumerate(decks):
        for j, (s, v) in enumerate(deck):
            if s != -1:
                d[4*i+j, 5 * s + v - 1] = 1

    return d

def stack2vec (stack):

    d = np.zeros((25, 10))

    for i, (s, v) in enumerate(stack):
        d[i, s] = 1
        d[i, 4+v] = 1

    return d

def discarded2vec (discarded):

    d = np.zeros((50, 10))

    for i, (s, v) in enumerate(discarded):
        d[i, s] = 1
        d[i, 4+v] = 1

    return d

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

    return state, true_decks

def encode_states (S):

    return [encode_state(player%5, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)[0] for player, (decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens) in enumerate(S)]

######################################################################

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

######################################################################

def get_models ():

    inp_1 = Input(shape=(1196,))

    ######################################################################

    # playable / crucial / irrelevant

    plb = Lambda(lambda x: x[:, 1121:1146])(inp_1)
    irv = Lambda(lambda x: x[:, 1146:1171])(inp_1)
    crc = Lambda(lambda x: x[:, 1171:1196])(inp_1)

    ######################################################################

    # pred deck

    inp_d = Input(shape=(1121,))

    x_d = Dense(256, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp_d)

    x_d = Dense(100, kernel_initializer='he_normal', bias_initializer='zeros', activation='linear')(x_d)
    x_d = Reshape((4, 25))(x_d)

    deck_pred = Softmax(axis=-1)(x_d)

    ########################################################################

    deck_net = Model(inputs=inp_d, outputs=deck_pred)

    ########################################################################

    inp_d_2 = Lambda(lambda x: x[::, :1121])(inp_1)

    deck_p = deck_net(inp_d_2)

    x_d_plb = Dot(axes=-1)([deck_p, plb])
    x_d_irv = Dot(axes=-1)([deck_p, irv])
    x_d_crc = Dot(axes=-1)([deck_p, crc])

    deck_p = Flatten()(deck_p)

    x = Dense(256, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp_1)

    x = Concatenate()([x, deck_p, x_d_plb, x_d_irv, x_d_crc])

    x = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

    out_q = Dense(48, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='linear', name='qval')(x)
    out_v = Dense(1, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='linear', name='value')(x)

    out_gotc = Dense(1, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='sigmoid', name='gotc')(x)
    out_msas = Dense(1, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='sigmoid', name='msas')(x)
    out_satf = Dense(1, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='sigmoid', name='satf')(x)

    out_policy = Dense(48, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='softmax')(x)

    q_net = Model(inputs=inp_1, outputs=[out_q, out_v, out_gotc, out_msas, out_satf])
    policy_net = Model(inputs=inp_1, outputs=out_policy)

    return q_net, policy_net, deck_net

########################################################################

def _encode_state (S):

    player = S[0]
    s, dk = encode_state(*S)

    return s.reshape(1, -1), dk[4*player:4*player+4].reshape(1, 4, 25)

def play_n_games (n, net=None, GAMMA=.999):

    ALL = [play_one_game(net=net) for _ in range(n)]

    _S = [S for S, _, _, _ in ALL]
    S = []

    for s in _S:
        S += [(player%5, *ss) for player, ss in enumerate(s)]

    # maximum still achievable score

    MSAS = []
    for s in S:
        MSAS += [max_still_achievable_score(s[3])]

    MSAS = np.array(MSAS) / 25.

    # score achieved thus far

    SATF = []
    for s in S:
        SATF += [len(s[2])]

    SATF = np.array(SATF) / 25.

    # encode w multiprocessing

    n_workers = min(n, cpu_count())
    pool = Pool(processes=n_workers)

    X = list(pool.imap(_encode_state, S, min(100, n // n_workers)))

    pool.close()
    pool.join()

    # concatenate arrays

    STATES = np.concatenate([s for s, _ in X], axis=0)
    TrDk_p = np.concatenate([dk for _, dk in X], axis=0)

    # game outcomes

    game_outcome = []

    for _, _, R, _ in ALL:

        game_outcome += [sum([max(0, r) for r in R])] * len(R)

    game_outcome = np.array(game_outcome) / 25.

    # make yuge arrays

    ACTIONS = np.concatenate([A for _, A, _, _ in ALL])
    REWARDS = np.concatenate([R for _, _, R, _ in ALL])
    FINAL = np.concatenate([F for _, _, _, F in ALL]).astype(bool)

    Q, V, _, _, _ = q_net.predict(STATES)

    P = np.zeros_like(Q) # policy gradient

    random_indices = np.random.choice(len(Q), len(Q), replace=False)

    for i in random_indices:

        Q[i, ACTIONS[i]] = REWARDS[i]
        V[i] = REWARDS[i]

        if not FINAL[i]:

            Q[i, ACTIONS[i]] += GAMMA * np.max(Q[i+1])
            V[i] += GAMMA * V[i+1]

        P[i, ACTIONS[i]] = Q[i, ACTIONS[i]] - V[i]

        # # not sure
        # std = np.std(Q[i])
        # P[i, ACTIONS[i]] = (Q[i, ACTIONS[i]] - np.mean(Q[i])) / (std if std > 0 else 1)

    return STATES, Q, V, P, MSAS, SATF, TrDk_p, game_outcome, sum(REWARDS) / n

########################################################################

def stupid_keras_memleak (q_net, policy_net, deck_net):

    q_net.save_weights('/home/florian/FF_PROG/HANABINEW/q_net_weights.h5')
    policy_net.save_weights('/home/florian/FF_PROG/HANABINEW/policy_net_weights.h5')
    deck_net.save_weights('/home/florian/FF_PROG/HANABINEW/deck_net_weights.h5')

    K.clear_session()

    q_net, policy_net, deck_net = get_models()

    q_net.load_weights('/home/florian/FF_PROG/HANABINEW/q_net_weights.h5')
    policy_net.load_weights('/home/florian/FF_PROG/HANABINEW/policy_net_weights.h5')
    deck_net.load_weights('/home/florian/FF_PROG/HANABINEW/deck_net_weights.h5')

    q_net.compile(loss='mse', optimizer=Adam(lr=.0001, clipvalue=2.))
    policy_net.compile(loss=policy_gradient_loss, optimizer=Adam(lr=.0001))
    deck_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.0001))

    return q_net, policy_net, deck_net

########################################################################

LOG_R = []
GAMMA = .999

q_net, policy_net, deck_net = get_models()

for k in range(10000):

    S, Q, V, P, MSAS, SATF, T, G, Rw = play_n_games(100, net=policy_net, GAMMA=GAMMA)

    random_indices = np.random.choice(len(Q), len(Q), replace=False)

    S = S[random_indices]
    Q = Q[random_indices]
    V = V[random_indices]
    T = T[random_indices]
    G = G[random_indices]
    P = P[random_indices]

    MSAS = MSAS[random_indices]
    SATF = SATF[random_indices]

    print ("\nREWARD:", Rw)

    deck_net.trainable = False

    for layer in deck_net.layers:
        layer.trainable = False

    q_net.compile(loss='mse', optimizer=Adam(lr=.0005, clipvalue=2.))
    policy_net.compile(loss=policy_gradient_loss, optimizer=Adam(lr=.0001))

    q_net.fit(S, [Q, V, G, MSAS, SATF], batch_size=16, verbose=(k%5==0))
    policy_net.fit(S, P, batch_size=16, verbose=(k%5==0))

    deck_net.trainable = True

    for layer in deck_net.layers:
        layer.trainable = True

    deck_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.0001))

    deck_net.fit(S[::, :1121], T, batch_size=16, verbose=(k%5==0))

    LOG_R += [Rw]

    np.save('/home/florian/FF_PROG/HANABINEW/logr.npy', LOG_R)

    q_net, policy_net, deck_net = stupid_keras_memleak(q_net, policy_net, deck_net)


S, A, R, F = play_one_game(policy_net)

def make_game_readable (S, A, R, F):

    player = 0

    for s, a, r, f in zip(S, A, R, F):

        others = list(range(5))
        others.remove(player)

        decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens = s

        state, plb = encode_state(player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)

        # play_me = pm_net.predict(state.reshape(1, -1))
        # q = q_net.predict(state.reshape(1, -1)).reshape(-1)
        # q = (10 * q).astype(int).tolist()
        p = policy_net.predict(state.reshape(1, -1)).reshape(-1)
        q, v, g, msas, satf = q_net.predict(state.reshape(1, -1))
        print (v, g, msas, satf)

        p = (100 * p).astype(int).tolist()

        dk = deck_net.predict(state[:1121].reshape(1, -1))
        dk = (100 * dk).astype('int')
        # dk = test_net.predict(state.reshape(1, -1))
        # dk = [list(d.reshape(-1)) for d in dk]


        print ("\n"+"*"*100)
        print ("PLAYER " + str(player+1) + "'s TURN\n")
        print ("DECKS:          ", decks)
        print ("DECKP:          ", dk)
        print ("STACK:          ", stack)
        print ("DISCARDED:      ", discarded)
        print ("DISCLOSED:      ")
        print (disclosed[player*4:player*4+4])
        print ("FINAL:          ", f)
        # print ("DISCLOSED:      ", disclosed)
        print ("n_inf_tokens:   ", n_inf_tokens)
        print ("n_fuse_tokens:  ", n_fuse_tokens)
        print ("len rest:       ", 50 - len(discarded) - len(stack) - 20)
        # print ("playable:       ", plb)
        print ("play me true:  ", [int(valid_card(card, stack)) for card in decks[player]])
        print ("-"*100)
        print ("p:              ", p)
        print ("-"*100)
        print ("action:         ", a, "(q = " + str(p[a]) + ")")
        print ("reward:         ", r)

        # q = np.exp(2*q)
        # q /= np.sum(q)
        # print (q)


        if a < 4:
            print ("Play card", a, ":", decks[player][a])
        elif a < 8:
            print ("Discard card", a-4, ":", decks[player][a-4])
        else:

            p = (a - 8) // 10
            p = others[p]

            k = (a - 8) % 10

            if k < 5:
                inds = [i for i, (ss, _) in enumerate(decks[p]) if ss == k]
                print ("Hint: Player", (p+1), "suit", k, " --- inds:", inds)
            else:
                inds = [i for i, (_, vv) in enumerate(decks[p]) if vv == k-4]
                print ("Hint: Player", (p+1), "value", k-4, " --- inds:", inds)



        print ("\n" + "*"*100 + "\n")

        player = (player + 1) % 5

make_game_readable(S, A, R, F)
