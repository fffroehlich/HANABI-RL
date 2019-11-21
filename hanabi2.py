from matplotlib import pyplot as plt
from copy import copy
import sys
from random import shuffle, randint, choice, random
import numpy as np
from itertools import product

#################################################################################################################################

# TODO:

n_suits = 3
n_players = 3

#################################################################################################################################

# for now: always 5 players, always 4 cards

all_cards = list(product(range(5), (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)))

# actions: 4 + 4 + n_players * 4 * 2


def p_choice (a):

    b = np.copy(a)
    b += np.amin(b)
    b /= np.sum(b)
    b = np.cumsum(b)

    r = random()

    i = 0
    while r > b[i]:
        i += 1

    return i

def play_one_game (net=None, verbose=False):

    S = []
    A = []
    R = []
    F = []

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

    disclosed = np.zeros((20, 2)) # val, suit

    final_round = False
    moves_left = 5

    S += [(copy(decks), copy(stack), copy(discarded), np.copy(disclosed), n_inf_tokens, n_fuse_tokens)]

    # PLAY

    while 1:

        r = 0

        if moves_left < 1:
            break

        if final_round:
            moves_left -= 1

        valid_actions = np.ones((48,), dtype=bool)
        # valid_actions[:4] = 0

        if n_inf_tokens < 1:
            # should it be 7: ?
            valid_actions[8:] = 0
        else:
            valid_actions[8*player+8:8*player+16] = 0

        actions = np.arange(48)[valid_actions]

        if net == None:

            action = choice(actions)

        else:

            state = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)
            q = np.squeeze(net.predict(state.reshape(1, -1)))

            q = q[valid_actions]
            action = actions[p_choice(q)]

        if action < 4:

            # play card w index action

            if valid_card (decks[player][action], stack):

                r = 1

                stack += [decks[player][action]]

                if not final_round:
                    decks[player][action] = rest.pop()

                if stack[-1][1] == 5:
                    # completing a suit retrieves one information token
                    n_inf_tokens = (n_inf_tokens + 1) % 8

                if not rest:
                    final_round = True

            else:

                # r = -1

                discarded += [decks[player][action]]

                if not final_round:
                    decks[player][action] = rest.pop()

                n_fuse_tokens -= 1
                if n_fuse_tokens <= 0:
                    break

                if not rest:
                    final_round = True

        elif action < 8:

            # discard card w index action-4

            discarded += [decks[player][action%4]]

            if not final_round:
                decks[player][action%4] = rest.pop()

            n_inf_tokens = (n_inf_tokens + 1) % 8

            if not rest:
                final_round = True

        else:

            # give hint

            if n_inf_tokens < 1:
                print ("INVALID ACTION")

            p = action // 8 - 1

            if p == player:
                print ("INVALID ACTION")

            s, v = decks[p][action % 4]

            if action % 8 < 4:
                # suit
                inds = [i for i, (ss, _) in enumerate(decks[p]) if ss == s]

                for i in inds:
                    disclosed[4*p+i, 0] = 1

            else:
                # value
                inds = [i for i, (_, vv) in enumerate(decks[p]) if vv == v]

                for i in inds:
                    disclosed[4*p+i, 1] = 1


            n_inf_tokens -= 1

        if action < 8:
            disclosed[player*4+(action%4), :] = 0

        # S += [(decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)]
        S += [(copy(decks), copy(stack), copy(discarded), np.copy(disclosed), n_inf_tokens, n_fuse_tokens)]
        A += [action]
        R += [r]
        F += [0]

        player = (player + 1) % 5

    # this is not really the final state but the next one is so don't add maxQ_next
    F[-1] = 1

    score = len(stack)

    if verbose: print (score)

    return S, A, R, F

S, A, R, F = play_one_game()

len(S)
len(A)
len(R)
len(F)


# possible problem: %8 etc but indices 0 - 47, not 1 - 48

def one_hot(k, n):

    x = np.zeros(n,)
    x[k] = 1

    return x

def decks2vec (decks):

    d = np.zeros((20, 10))
    i = 0

    for deck in decks:
        for s, v in deck:
            d[i, s] = 1
            d[i, 4+v] = 1
            i += 1

    return d

def stack2vec (stack):

    d = np.zeros((50, 10))

    for i, (s, v) in enumerate(stack):
        d[i, s] = 1
        d[i, 4+v] = 1

    return d

def hand2vec (player, decks, disclosed):

    d = np.zeros((4, 10))

    for i in range(4):

        s, v = decks[player][i]

        if disclosed[player*4+i, 0]:
            d[i, s] = 1
        if disclosed[player*4+i, 1]:
            d[i, v + 4] = 1

    return d

def encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens):

    hand = hand2vec(player, decks, disclosed)
    decks = decks2vec(decks)

    decks[player*4:player*4+4, :] = 0

    stack = stack2vec(stack)
    discarded = stack2vec(discarded)
    ninf = one_hot(n_inf_tokens-1, 8)
    nfuse = one_hot(n_fuse_tokens-1, 3)

    state = np.concatenate([a.reshape(-1) for a in [hand, decks, stack, discarded, ninf, nfuse, disclosed]], axis=0)

    return state

def encode_states (S):

    return [encode_state(player%5, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens) for player, (decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens) in enumerate(S)]

# also 'disclosed' directly

########################################################################################################################

from keras.layers import Dense, Input, Flatten, BatchNormalization, Add
from keras.models import Model
from keras.optimizers import Adam

inp = Input(shape=(1291,))

x = Dense(512, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp)
x = Dense(256, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

for _ in range(8):

    x_1 = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

    x = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x_1)
    x = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

    x = Add()([x_1, x])

    # x = BatchNormalization()(x)

out = Dense(48, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='linear')(x)

# should out really be softmax?
# maybe that's why DoKo didn't work

net = Model(inputs=inp, outputs=out)
net.compile(loss='mse', optimizer=Adam(lr=.01))

# net.fit_generator(data_gen_tn(64), epochs=2000, steps_per_epoch=200)

########################################################################################################################

def play_n_games (n, nonet=True, GAMMA=.7):

    # print ("\nPlaying ...\n")

    ALL = []
    for _ in range(n):

        # sys.stdout.write('\r'+str(_))
        # sys.stdout.flush()

        S, A, R, F = play_one_game(None if nonet else net)
        S = encode_states(S[:-1])

        ALL += [(S, A, R, F)]

    # make yuge arrays
    STATES = np.concatenate([S for S, _, _, _ in ALL])
    ACTIONS = np.concatenate([A for _, A, _, _ in ALL])
    REWARDS = np.concatenate([R for _, _, R, _ in ALL])
    FINAL = np.concatenate([F for _, _, _, F in ALL])

    Q = net.predict(STATES)

    Q[range(len(ACTIONS)), ACTIONS] = REWARDS

    a = np.arange(len(ACTIONS))[~FINAL]
    b = ACTIONS[~FINAL]
    Q[a, b] += GAMMA * np.max(Q[1:, :][~FINAL[1:]])

    random_indices = np.random.choice(len(Q), len(Q), replace=False)

    print ("\nREWARD:", sum(REWARDS))

    return STATES[random_indices], Q[random_indices]


# some random games
for _ in range(20):
    S, Q = play_n_games(1500, nonet=True, GAMMA=.05)
    net.fit(S, Q, batch_size=32, verbose=0)



LOG_R = []
LOG_G = []

GAMMA = .05
GAMMA_MAX = .8

for _ in range(100000):
    print ("GAMMA:", GAMMA)
    S, Q = play_n_games(200, nonet=False, GAMMA=GAMMA)
    net.fit(S, Q, verbose=0)
    GAMMA *= 1.005
    GAMMA = min(GAMMA, GAMMA_MAX)


plt.figure(figsize=(10, 10))
plt.bar(range(48), Q[-1])
plt.show()

S, A, R, F = play_one_game(net)
S


def make_game_readable (S, A, R, F):

    player = 0

    for s, a, r in zip(S, A, R):

        decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens = s

        state = encode_state(player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)
        q = net.predict(state.reshape(1, -1))

        print ("*"*50)
        print ("PLAYER " + str(player+1) + "'s TURN\n")
        print ("DECKS:          ", decks)
        print ("STACK:          ", stack)
        print ("DISCARDED:      ", discarded)
        # print ("DISCLOSED:      ", disclosed)
        print ("n_inf_tokens:   ", n_inf_tokens)
        print ("n_fuse_tokens:  ", n_fuse_tokens)
        print ("\n" + "*"*50 + "\n")


        player = (player + 1) % 5


make_game_readable(S, A, R, F)


# problem: if you do it like this, the rewards (q-values) are only ever passed on in the next batch
# problem: in current implementation, if action random, hint much more likely than play/discard
# (easy to fix)
# does this reward structure make sense? or do it separately for each player?
# 'disclosed' is wrong: player knows either suit or value
# missing: who did what when (essential)
# could also show previous states (simple)
# idiot. own hand is still visible (though it might be useful to train like that for a while?).
# predict own hand (or which cards, irrespective of order)
# index problem 0-47 v 1-48
# double check if S, A, R, final, ect. is correct
# make as much information EXPLICITLY available as possible
# start with simpler version / task
# experience replay
# where do these occasional absurdly high losses come from?
# batch norm in dqn? (you're not even training in batches so no)
# don't forget how hard the task is even at the most basic level---will take time
# >>>> need to learn to give hints at beginning (which card)
# >>>> need to learn to USE hints and how
# probably learns to never play cards at the moment?
# am i handling invalid actions correctly? doesn't this lead to incorrect q value updates? no.
# batchnorm seems to have caused exploding q values, or too many layers
# train with all cards disclosed; fix weights, train other net
# experiment with rewards --- should be able to achieve non-negative R simply by never playing a card ...
# reward from playing a card is extremely unlikely. double check all code
# CURRENTLY OWN HAND VISIBLE (sanity check)
# learning rate
# experiment: rewards, lr, show own hand, limit valid actions
# limit valid actions in beginning
# r = -1 for lost cards not always useful; risky play can be good towards end of game
# IDIOT!! stack/discarded are updated ... fucking lists
