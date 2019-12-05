from matplotlib import pyplot as plt
from copy import copy, deepcopy
import sys
from random import shuffle, randint, choice, random
import numpy as np
from itertools import product
from time import time, sleep
from multiprocessing import Pool, cpu_count
from itertools import tee

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

#################################################################################################################################

# for now: always 5 players, always 4 cards

all_cards = list(product(range(5), (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)))

# actions: 4 + 4 + n_players * 4 * 2

def play_one_game (net=None, verbose=False, EPSILON=.1):

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

        if net == None or random() < EPSILON:

            action = choice(actions)

        else:

            state, _ = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)

            q = np.squeeze(q_net.predict(state.reshape(1, -1)))
            q = q[valid_actions]

            # q /= np.sum(q)
            # q = np.exp(1.5 * q)
            # q /= np.sum(q)
            # action = np.random.choice(actions, p=q)

            action = actions[np.argmax(q)]



        # # blatant cheating
        #
        # useful_hint = 8 + np.arange(40)[give_useful_hint(player, decks, disclosed, stack).astype(bool)]
        #
        # # print ([h for h in useful_hint if not h in list(actions[8:])])
        #
        # pm = play_me (player, decks, disclosed, discarded, stack)
        #
        # if max(pm) > 0:
        #
        #     actions = np.arange(4)[np.array(pm, dtype=bool)]
        #     action = choice(actions)
        #
        # elif n_inf_tokens > 0 and random() < -1.5 and len(useful_hint) > 0:
        #
        #     if net == None:
        #
        #         action = choice(useful_hint)
        #
        #     else:
        #
        #         state, _ = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)
        #         q = np.squeeze(q_net.predict(state.reshape(1, -1)))
        #         q = q[useful_hint]
        #         q /= np.sum(q)
        #         q = np.exp(.8 * q)
        #         q /= np.sum(q)
        #
        #         action = np.random.choice(useful_hint, p=q)
        #
        # elif net == None:
        #
        #     action = choice(actions)
        #
        # else:
        #
        #     state, _ = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)
        #     q = np.squeeze(q_net.predict(state.reshape(1, -1)))
        #     q = q[valid_actions]
        #     q /= np.sum(q)
        #     q = np.exp(.8 * q)
        #     q /= np.sum(q)
        #
        #
        #
        #     action = np.random.choice(actions, p=q)
        #     # action = np.argmax(q)

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
                    n_inf_tokens = (n_inf_tokens + 1) % 8

                if not rest:
                    final_round = True

            else:

                # r = -10

                discarded += [decks[player][action]]

                if final_round:
                    decks[player][action] = (-1, -1)
                else:
                    decks[player][action] = rest.pop()

                # cause of death: old age
                n_fuse_tokens -= 1

                if n_fuse_tokens <= 0:
                    r = -1 # * sum(R)
                    break

                if not rest:
                    final_round = True

        elif action < 8:

            # r = -1

            # discard card w index action-4

            discarded += [decks[player][action%4]]

            if final_round:
                decks[player][action%4] = (-1, -1)
            else:
                decks[player][action%4] = rest.pop()

            n_inf_tokens = (n_inf_tokens + 1) % 8

            if not rest:
                final_round = True

        else:

            # r = -1

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

        # S += [(decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)]
        S += [(deepcopy(decks), copy(stack), copy(discarded), np.copy(disclosed), n_inf_tokens, n_fuse_tokens)]
        A += [action]
        R += [r]
        F += [0]

        player = (player + 1) % 5

    # because 'break'k
    # S += [(deepcopy(decks), copy(stack), copy(discarded), np.copy(disclosed), n_inf_tokens, n_fuse_tokens)]
    A += [action]
    R += [r]
    F += [1]


    score = len(stack)

    if verbose: print (score)

    return S, A, R, F

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

    # plb = np.zeros(25,)
    # irv = np.zeros(25,)
    # crc = np.zeros(25,)
    #
    # for i, card in enumerate(all_cards_):
    #
    #     if valid_card(card, stack):
    #
    #         plb[i] = 1
    #
    #     else:
    #
    #         s, v = card
    #
    #         if v == 5


    plb = np.array([valid_card(card, stack) for card in all_cards_nd], dtype=int)
    irv = np.array([irrelevant_card(card, stack, discarded) for card in all_cards_nd], dtype=int)
    crc = np.array([crucial_card(card, stack, discarded) for card in all_cards_nd], dtype=int)


    # plb = np.array([valid_card(card, stack) for card in all_cards_nd], dtype=int)
    # irv = np.array([irrelevant_card(card, stack, discarded) for card in all_cards_nd], dtype=int)
    # crc = np.array([crucial_card(card, stack, discarded) for card in all_cards_nd], dtype=int)

    true_decks = decks2vec2(decks)

    decks = decks2vec(decks)

    enc_disclosed = np.copy(disclosed)
    enc_disclosed = np.concatenate([enc_disclosed[player*4:player*4+4], enc_disclosed[:player*4], enc_disclosed[player*4+4:]])

    # !!! sanity check / pre-training: player sees owns deck
    # enc_disclosed[:4] = np.copy(decks[player*4:player*4+4])

    # awkward reordering
    decks = np.concatenate([decks[:player*4], decks[player*4+4:]])

    stack = stack2vec(stack)
    discarded = discarded2vec(discarded)
    ninf = one_hot(n_inf_tokens-1, 8)
    nfuse = one_hot(n_fuse_tokens-1, 3)

    state = np.concatenate([a.reshape(-1) for a in [decks, stack, discarded, ninf, nfuse, enc_disclosed, plb, irv, crc]], axis=0)

    # state = np.copy(decks[player*4:player*4+4]).reshape(-1)

    return state, true_decks

def encode_states (S):

    return [encode_state(player%5, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)[0] for player, (decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens) in enumerate(S)]

########################################################################################################################

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

########################################################################################################################

from keras.layers import Dense, Input, Flatten, BatchNormalization, Add, Concatenate, Lambda, Reshape, Activation, Softmax, Multiply, Dot
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import binary_crossentropy


inp_1 = Input(shape=(1196,))

########################################################################

# playable / crucial / irrelevant

plb = Lambda(lambda x: x[:, 1121:1146])(inp_1)
irv = Lambda(lambda x: x[:, 1146:1171])(inp_1)
crc = Lambda(lambda x: x[:, 1171:1196])(inp_1)

########################################################################

# pred deck

inp_d = Input(shape=(1121,))

x_d = Dense(512, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp_d)
x_d = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x_d)

x_d = Dense(100, kernel_initializer='he_normal', bias_initializer='zeros', activation='linear')(x_d)
x_d = Reshape((4, 25))(x_d)

deck_pred = Softmax(axis=-1)(x_d)

########################################################################

deck_net = Model(inputs=inp_d, outputs=deck_pred)
deck_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.001))

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

out = Dense(48, kernel_initializer='zeros', bias_initializer='zeros', activation='linear')(x)

q_net = Model(inputs=inp_1, outputs=out)
q_net.compile(loss='mse', optimizer=Adam(lr=.0001, clipvalue=2.))

# q_net.compile(loss='huber', optimizer=Adam(lr=.001, clipvalue=3.))
# deck_net = Model(inputs=inp_1, outputs=deck_pred)
# deck_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.001))

# multiprocessing: runs in background even after kernel shutdown--->everything becomes slow

########################################################################################################################

test_net = Model(inputs=inp_1, outputs=[x_d_plb, x_d_irv, x_d_crc])
test_net.compile(loss='mse', optimizer=Adam())

########################################################################################################################

def _process (SARF):

    S, A, _, _ = SARF

    S_p = []
    S_r_0 = []
    S_r_1 = []

    TrDk_p = []
    TrDk_r = []

    for player, (s_0, s_1, a) in enumerate(zip(S, S[1:], A)):

        player = player % 5

        if a > 7:

            receive = (a - 8) // 10

            if receive >= player:
                receive += 1

            s_r_0, dk = encode_state(receive, *s_0)
            s_r_1, dk = encode_state(receive, *s_1)

            S_r_0 += [s_r_0.reshape(1, -1)]
            S_r_1 += [s_r_1.reshape(1, -1)]

            TrDk_r += [dk[receive*4:receive*4+4].reshape(1, 4, 25)]

        s, dk = encode_state(player, *s_0)

        S_p += [s.reshape(1, -1)]
        TrDk_p += [dk[player*4:player*4+4].reshape(1, 4, 25)]

    if A[-1] > 7:

        s_r_0, dk = encode_state(receive, *S[-2])

        # final action will almost never be a hint
        # most games end because of fuse tokens (----> play card)
        # probably never gets to the point of running out of cards yet

        S_r_0 += [s_r_0.reshape(1, -1)]
        S_r_1 += [s_r_0.reshape(1, -1)]

        TrDk_r += [dk[receive*4:receive*4+4].reshape(1, 4, 25)]

    player = (player + 1) % 5

    s, dk = encode_state(player, *S[-1])
    S_p += [s.reshape(1, -1)]
    TrDk_p += [dk[player*4:player*4+4].reshape(1, 4, 25)]


    return S_p, S_r_0, S_r_1, TrDk_p, TrDk_r

########################################################################################################################

def play_n_games (n, nonet=True, GAMMA=.99, EPSILON=.1, hint_reward=False):

    if nonet:

        n_workers = min(n, cpu_count())
        pool = Pool(processes=n_workers)

        ALL = list(pool.imap_unordered(play_one_game, (None for _ in range(n)), min(500, n // n_workers)))

        pool.close()
        pool.join()

    else:

        ALL = [play_one_game(q_net) for _ in range(n)]

    if hint_reward:

        _S = [S for S, _, _, _ in ALL]
        S = []

        for s in _S:
            # why would s not be a list?
            S += [(player%5, *ss) for player, ss in enumerate(s)]

        n_workers = min(n, cpu_count())
        pool = Pool(processes=n_workers)

        # needs to be ordered! ACTIONS! REWARDS! FINAL!
        X = list(pool.imap(_process, ALL, min(100, n // n_workers)))

        pool.close()
        pool.join()

        S_p = []
        S_r_0 = []
        S_r_1 = []
        TrDk_p = []
        TrDk_r = []

        for s_p, s_r_0, s_r_1, trdk_p, trdk_r in X:

            S_p += s_p
            S_r_0 += s_r_0
            S_r_1 += s_r_1
            TrDk_p += trdk_p
            TrDk_r += trdk_r

        STATES = np.concatenate(S_p, axis=0)

        STATES_r_0 = np.concatenate(S_r_0, axis=0)
        STATES_r_1 = np.concatenate(S_r_1, axis=0)

        TrDk_p = np.concatenate(TrDk_p, axis=0)
        TrDk_r = np.concatenate(TrDk_r, axis=0)

    else:

        _S = [S for S, _, _, _ in ALL]
        S = []

        for s in _S:
            # why would s not be a list?
            S += [(player%5, *ss) for player, ss in enumerate(s)]

        n_workers = min(n, cpu_count())
        pool = Pool(processes=n_workers)

        X = list(pool.imap(_encode_state, S, min(100, n // n_workers)))

        pool.close()
        pool.join()

        STATES = np.concatenate([s for s, _ in X], axis=0)
        TrDk_p = np.concatenate([dk for _, dk in X], axis=0)

    # make yuge arrays

    ACTIONS = np.concatenate([A for _, A, _, _ in ALL])
    REWARDS = np.concatenate([R for _, _, R, _ in ALL])
    FINAL = np.concatenate([F for _, _, _, F in ALL]).astype(bool)

    # problem: will always be > 0 --- normalize somehow; also rescale

    if hint_reward:

        PrDk_0 = deck_net.predict(STATES_r_0[::, :1121])
        PrDk_1 = deck_net.predict(STATES_r_1[::, :1121])

        ################################################################

        plb = STATES_r_0[::, -3*25:-2*25]
        crc = STATES_r_0[::, -25:]

        plb_pred_0 = np.zeros((len(plb), 4))
        plb_pred_1 = np.zeros((len(plb), 4))
        plb_true = np.zeros((len(plb), 4))

        crc_pred_0 = np.zeros((len(plb), 4))
        crc_pred_1 = np.zeros((len(plb), 4))
        crc_true = np.zeros((len(plb), 4))

        for i in range(4):

            plb_pred_0[:, i] = np.sum(plb * PrDk_0[:, i, :], axis=-1)
            plb_pred_1[:, i] = np.sum(plb * PrDk_1[:, i, :], axis=-1)
            plb_true[:, i] = np.sum(plb * TrDk_r[:, i, :], axis=-1)

            crc_pred_0[:, i] = np.sum(crc * PrDk_0[:, i, :], axis=-1)
            crc_pred_1[:, i] = np.sum(crc * PrDk_1[:, i, :], axis=-1)
            crc_true[:, i] = np.sum(crc * TrDk_r[:, i, :], axis=-1)

        # log safety
        crc_true = np.clip(crc_true, 1e-5, 1-1e-5)
        plb_true = np.clip(plb_true, 1e-5, 1-1e-5)

        E_plb_0 = - np.mean(plb_pred_0 * np.log(plb_true), axis=-1)
        E_plb_1 = - np.mean(plb_pred_1 * np.log(plb_true), axis=-1)
        E_crc_0 = - np.mean(crc_pred_0 * np.log(crc_true), axis=-1)
        E_crc_1 = - np.mean(crc_pred_1 * np.log(crc_true), axis=-1)

        E_plb_delta = (E_plb_0 / (E_plb_1 + 1e-10)) - 1
        E_crc_delta = (E_crc_0 / (E_crc_1 + 1e-10)) - 1

        E_delta = E_plb_delta + E_crc_delta

        E_delta = np.tanh(E_delta)
        E_delta = np.clip(E_delta, 0, 1)

        ################################################################
        #
        # E_0 = - np.mean(TrDk_r * np.log(PrDk_0), axis=(1, 2))
        # E_1 = - np.mean(TrDk_r * np.log(PrDk_1), axis=(1, 2))
        #
        # E_delta = (E_0 / (E_1 + 1e-10)) - 1
        #
        # mE = np.mean(E_delta)
        # E_delta[E_delta<mE] = 0
        #
        # E_delta = np.clip(E_delta, 0, 1)

        a = np.arange(len(ACTIONS))[ACTIONS>7]
        b = ACTIONS[ACTIONS>7]

        # it's a problem that a and E sometimes don't match!
        # 9577 9639
        # 9713 9771
        # 9497 9562
        # a always shorter than E
        # also always large value, not 1 or something

        if not a.shape[0] == E_delta.shape[0]:
            print (len(S_r_0))
            print ('sr0', STATES_r_0.shape)
            print ('b', b.shape)
            print (a.shape[0], E_delta.shape[0])

        # normalize across hints?

    Q = q_net.predict(STATES)

    random_indices = np.random.choice(len(Q), len(Q), replace=False)

    for i in random_indices:
        Q[i, ACTIONS[i]] = REWARDS[i]
        if not FINAL[i]:
            Q[i, ACTIONS[i]] += GAMMA * np.max(Q[i+1])


    if hint_reward:

        # print (E_delta)
        # print (Q)

        Q[a, b] += .02 * E_delta[:len(a)]


    return STATES, Q, TrDk_p, sum(REWARDS) / n

########################################################################################################################

# q_net.save('/home/florian/FF_PROG/HANABI/qnetpres.h5')
# deck_net.save('/home/florian/FF_PROG/HANABI/decknetpres.h5')

_ = play_n_games(1,hint_reward=True)

def _encode_state (S):

    player = S[0]
    s, dk = encode_state(*S)

    return s.reshape(1, -1), dk[4*player:4*player+4].reshape(1, 4, 25)

# random games

print ("Play some random games with multiprocessing ...\n")

K = 80

tic = time()

for k in range(K):

    ETA = ((time() - tic) / (k + 1)) * (K - k - 1)

    h = int(ETA // 3600)
    m = int((ETA - 3600 * h) // 60)
    s = int(ETA - 3600 * h - 60 * m)

    ETA = str(h) + "h " + str(m) + "m " + str(s) + "s"

    sys.stdout.write(str(k+1) + " / " + str(K) + "   ETA:  " + ETA + "           \r")
    sys.stdout.flush()

    S, Q, T, _ = play_n_games(1000, nonet=True, GAMMA=.99, hint_reward=True)

    random_indices = np.random.choice(len(Q), len(Q), replace=False)

    S = S[random_indices]
    Q = Q[random_indices]
    T = T[random_indices]

    deck_net.trainable = False
    for layer in deck_net.layers:
        layer.trainable = False
    q_net.compile(loss='mse', optimizer=Adam(lr=.0001))
    q_net.fit(S, Q, batch_size=32, verbose=0)

    deck_net.trainable = True
    for layer in deck_net.layers:
        layer.trainable = True
    deck_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.001))
    deck_net.fit(S[:, :1121], T, batch_size=32, verbose=0)

    # _, Q, _, _ = play_n_games(1, nonet=False, GAMMA=.99)
    #
    # QQ = np.zeros((80, 48))
    # QQ[:len(Q), :] = Q
    #
    # plt.figure(figsize=(20, 20))
    # plt.imshow(QQ, cmap='inferno', vmin=-10, vmax=100)
    # # plt.axis('off')
    # plt.show()

print ("\nPlay with epsilon-soft ...\n")

LOG_R = []
GAMMA = .99
EPSILON = .99
EPSILON_MIN = .1
EPSILON_DECAY = .999

for _ in range(100000):

    print ("EPSILON:", EPSILON)

    S, Q, T, Rw = play_n_games(1000, nonet=False, GAMMA=GAMMA, EPSILON=EPSILON, hint_reward=True)

    EPSILON *= EPSILON_DECAY
    EPSILON = max(EPSILON_MIN, EPSILON)

    random_indices = np.random.choice(len(Q), len(Q), replace=False)

    S = S[random_indices]
    Q = Q[random_indices]
    T = T[random_indices]

    print ("\nREWARD:", Rw)

    deck_net.trainable = False

    for layer in deck_net.layers:
        layer.trainable = False

    q_net.compile(loss='mse', optimizer=Adam(lr=.0001))

    q_net.fit(S, Q, batch_size=128, verbose=1)

    deck_net.trainable = True

    for layer in deck_net.layers:
        layer.trainable = True

    deck_net.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.001))

    deck_net.fit(S[::, :1121], T, batch_size=32, verbose=1)

    LOG_R += [Rw]

# np.save('/home/florian/FF_PROG/HANABI/logr.npy', LOG_R)
e = .99
EPSLOG = []
for _ in range(len(LOG_R)):
    EPSLOG += [e]
    e *= .999

plt.figure(figsize=(10, 10))
# plt.plot(LOG_R)
plt.plot(EPSLOG)
plt.show()

# q is randomized!

plt.figure(figsize=(10, 10))
plt.bar(range(48), Q[0])
plt.show()

S, Q, T, Rw = play_n_games(1000, nonet=False, GAMMA=.99, EPSILON=0, hint_reward=True)
Rw

# Q is randomized!
plt.figure(figsize=(20, 20))
# plt.imshow(np.repeat(S, 20, axis=0), cmap='inferno')
plt.imshow(Q, cmap='inferno')
# plt.axis('off')
plt.show()


for i in range(len(Q)):
    plt.figure(figsize=(10, 10))
    plt.bar(range(48), Q[i])
    # plt.ylim((-5, 10))
    # plt.axis('off')
    plt.show()


# 0.883284994052061


S, A, R, F = play_one_game(None)
S, A, R, F = play_one_game(q_net, EPSILON=0)

def make_game_readable (S, A, R, F):

    player = 0

    for s, a, r, f in zip(S, A, R, F):

        others = list(range(5))
        others.remove(player)

        decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens = s

        state, plb = encode_state(player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)

        # play_me = pm_net.predict(state.reshape(1, -1))
        q = q_net.predict(state.reshape(1, -1)).reshape(-1)
        q = (10 * q).astype(int).tolist()

        # dk = deck_net.predict(state[:1121].reshape(1, -1))

        dk = test_net.predict(state.reshape(1, -1))
        dk = [list(d.reshape(-1)) for d in dk]


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
        print ("q:              ", q)
        print ("-"*100)
        print ("action:         ", a, "(q = " + str(q[a]) + ")")
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

wq = q_net.get_weights()

len(wq)
wq[-4][-12:,:]


# Q is randomized!
plt.figure(figsize=(20, 20))
# plt.imshow(np.repeat(S, 20, axis=0), cmap='inferno')
plt.imshow(wq[-2], cmap='inferno')
# plt.axis('off')
plt.show()
