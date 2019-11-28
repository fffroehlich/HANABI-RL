from matplotlib import pyplot as plt
from copy import copy, deepcopy
import sys
from random import shuffle, randint, choice, random
import numpy as np
from itertools import product
from time import time, sleep
from multiprocessing import Pool, cpu_count
from itertools import tee


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

        if net == None:

            action = choice(actions)

        else:

            state, _ = encode_state (player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)
            q = np.squeeze(q_net.predict(state.reshape(1, -1)))
            q = q[valid_actions]
            # q -= np.amin(q)
            q /= np.sum(q)
            # q += .1
            # q /= np.sum(q)
            # q = np.exp(1.5 * q)
            q = np.exp(.8 * q)
            q /= np.sum(q)

            # plt.bar(range(len(q)), q)
            # plt.show()

            action = np.random.choice(actions, p=q)
            # action = np.argmax(q)

        if action < 4:

            # play card w index action

            if valid_card (decks[player][action], stack):

                r = 10

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

                # r = -1

                discarded += [decks[player][action]]

                if final_round:
                    decks[player][action] = (-1, -1)
                else:
                    decks[player][action] = rest.pop()

                n_fuse_tokens -= 1

                if n_fuse_tokens <= 0:
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

            n_inf_tokens = (n_inf_tokens + 1) % 8

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

        # S += [(decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)]
        S += [(deepcopy(decks), copy(stack), copy(discarded), np.copy(disclosed), n_inf_tokens, n_fuse_tokens)]
        A += [action]
        R += [r]
        F += [0]

        player = (player + 1) % 5

    # because 'break'
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
    playable = np.array([valid_card(card, stack) for card in all_cards_nd], dtype=int)

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

    # !!!
    # enc_disclosed[4:] = 0
    # discarded = np.zeros_like(discarded)
    # decks = np.zeros_like(decks)

    state = np.concatenate([a.reshape(-1) for a in [decks, stack, discarded, ninf, nfuse, enc_disclosed]], axis=0)


    # state = np.copy(decks[player*4:player*4+4]).reshape(-1)

    return state, playable

def encode_states (S):

    return [encode_state(player%5, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens) for player, (decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens) in enumerate(S)]

########################################################################################################################

possible_values = [list(range(1, i)) for i in range(2, 7)]
single_suit = [[[(i, j) for j in k] for k in possible_values] for i in range(5)]
all_stacks = list(product(*single_suit))

all_stacks2 = []

for stack in all_stacks:

    stack2 = []

    for s in stack:

        if isinstance(s, list):
            stack2 += s
        else:
            stack2 += [s]

    all_stacks2 += [stack2]

all_stacks = all_stacks2

del all_stacks2

len(all_stacks)
all_stacks

def shuffle_list (l):

    random_indices = np.random.choice(len(l), len(l), replace=False)

    return [l[i] for i in random_indices]

def stack_gen (n=32):

    shuffle(all_stacks)
    i = 0

    all_cards_nd = list(product(range(5), range(1, 6)))

    def playable (stack):

        return np.array([valid_card(card, stack) for card in all_cards_nd], dtype=int)


    while True:

        stack = copy(all_stacks[i:i+n])

        plb = np.concatenate([playable(s).reshape(1, -1) for s in stack], axis=0)

        stack = [shuffle_list(s) for s in stack]
        stack = np.concatenate([stack2vec(s).reshape((1, 25*10)) for s in stack], axis=0)

        s = np.zeros((n, 1121))
        s[:, 160:160+250] = stack

        i += n
        if i > len(all_stacks) - n:
            i = 0
            shuffle(all_stacks)

        yield s, plb

for s, p in stack_gen():
    break

########################################################################################################################

from keras.layers import Dense, Input, Flatten, BatchNormalization, Add, Concatenate, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K

inp_1 = Input(shape=(1121,))

x_1 = Lambda(lambda x: x[:, 160:160+250])(inp_1)
x_1 = Dense(64, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x_1)

playable = Dense(25, kernel_initializer='he_normal', bias_initializer='zeros', activation='sigmoid')(x_1)


inp_2 = Concatenate()([inp_1, playable])

x = Dense(1024, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(inp_2)

x = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)
x = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)

# out = Dense(48, kernel_initializer='glorot_normal', bias_initializer='zeros', activation='linear')(x)
out = Dense(48, kernel_initializer='zeros', bias_initializer='zeros', activation='linear')(x)

q_net = Model(inputs=inp_1, outputs=out)
q_net.compile(loss='mse', optimizer=Adam(lr=.001, clipvalue=3.))

p_net = Model(inputs=inp_1, outputs=playable)
p_net.compile(loss='binary_crossentropy', optimizer=Adam(lr=.001))


########################################################################################################################

# p_net.fit_generator(stack_gen(64), steps_per_epoch=500, verbose=1, epochs=50)

p_net = load_model('/home/florian/FF_PROG/HANABI/saved_models/p_net.h5')

p_net.trainable = False

for layer in p_net.layers:
    layer.trainable = False

########################################################################################################################

def play_n_games (n, nonet=True, GAMMA=.7):

    ALL = []
    for _ in range(n):

        S, A, R, F = play_one_game(None if nonet else q_net)
        SP = encode_states(S)
        S = [s for s, _ in SP]
        P = [p for _, p in SP]

        ALL += [(S, A, R, F, P)]

    # make yuge arrays
    STATES = np.concatenate([S for S, _, _, _, _ in ALL])
    ACTIONS = np.concatenate([A for _, A, _, _, _ in ALL])
    REWARDS = np.concatenate([R for _, _, R, _, _ in ALL])
    FINAL = np.concatenate([F for _, _, _, F, _ in ALL])
    PLAYABLE = np.concatenate([P for _, _, _, _, P in ALL])

    # !!!
    FINAL = FINAL.astype(bool)

    Q = q_net.predict(STATES)

    random_indices = np.random.choice(len(Q), len(Q), replace=False)
    for i in random_indices:
        Q[i, ACTIONS[i]] = REWARDS[i]
        if not FINAL[i]:
            Q[i, ACTIONS[i]] += GAMMA * np.max(Q[i+1])

    random_indices = np.random.choice(len(Q), len(Q), replace=False)

    return STATES[random_indices], Q[random_indices], PLAYABLE[random_indices], sum(REWARDS) / n

# q_net.save('/home/florian/FF_PROG/HANABI/q_net_lotsarandom.h5')
# p_net.save('/home/florian/FF_PROG/HANABI/p_net.h5')

def _encode_state (S):

    return encode_state(*S)[0].reshape(1, -1)

def play_n_games_MP (n, GAMMA=.99):

    n_workers = cpu_count()
    pool = Pool(processes=n_workers)

    ALL = list(pool.imap_unordered(play_one_game, (None for _ in range(n)), min(50, n // n_workers)))

    S = [S for S, _, _, _ in ALL]
    S2 = []

    for s in S:
        if isinstance(s, list):
            s = [(player%5, *ss) for player, ss in enumerate(s)]
            S2 += s
        else:
            S2 += [s]

    # needs to be ordered
    STATES = np.concatenate(list(pool.imap(_encode_state, S2, min(50, n // n_workers))), axis=0)

    pool.close()
    pool.join()

    ACTIONS = np.concatenate([A for _, A, _, _ in ALL])
    REWARDS = np.concatenate([R for _, _, R, _ in ALL])
    FINAL = np.concatenate([F for _, _, _, F in ALL])


    FINAL = FINAL.astype(bool)

    Q = q_net.predict(STATES)

    random_indices = np.random.choice(len(Q), len(Q), replace=False)
    for i in random_indices:
        Q[i, ACTIONS[i]] = REWARDS[i]
        if not FINAL[i]:
            Q[i, ACTIONS[i]] += GAMMA * np.max(Q[i+1])


    random_indices = np.random.choice(len(Q), len(Q), replace=False)

    return STATES[random_indices], Q[random_indices]


# random games

print ("Play some random games with multiprocessing ...\n")

K = 2000

tic = time()

for k in range(K):

    ETA = ((time() - tic) / (k + 1)) * (K - k - 1)

    h = int(ETA // 3600)
    m = int((ETA - 3600 * h) // 60)
    s = int(ETA - 3600 * h - 60 * m)

    ETA = str(h) + "h " + str(m) + "m " + str(s) + "s"

    sys.stdout.write(str(k+1) + " / " + str(K) + "   ETA:  " + ETA + "           \r")
    sys.stdout.flush()

    S, Q = play_n_games_MP(1000, GAMMA=.99)
    q_net.fit(S, Q, batch_size=128, verbose=0)


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

for _ in range(100000):

    S, Q, P, Rw = play_n_games(1000, nonet=False, GAMMA=GAMMA)
    print ("\nREWARD:", Rw)
    # p_net.fit(S, P, batch_size=8, verbose=1 if _%10==0 else 0)
    q_net.fit(S, Q, batch_size=64, verbose=1 if _%10==0 else 0)

    LOG_R += [Rw]



plt.figure(figsize=(10, 10))
plt.plot(LOG_R)
plt.show()

# q is randomized!
plt.figure(figsize=(10, 10))
plt.bar(range(48), Q[0])
plt.show()


S, Q, P, Rw = play_n_games(3, nonet=False, GAMMA=GAMMA)
np.min(Q)
np.max(Q)

print (list(np.sum(S, axis=0)))


plt.figure(figsize=(20, 20))
plt.imshow(Q, cmap='inferno')
# plt.axis('off')
plt.show()


for i in range(len(Q)):
    plt.figure(figsize=(10, 10))
    plt.bar(range(48), Q[i])
    # plt.ylim((-5, 10))
    # plt.axis('off')
    plt.show()

S, A, R, F = play_one_game(q_net)
R
A = np.array(A)
R = np.array(R, dtype=bool)
A[R]

def make_game_readable (S, A, R, F):


    player = 0

    for s, a, r, f in zip(S, A, R, F):

        others = list(range(5))
        others.remove(player)

        decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens = s

        state, plb = encode_state(player, decks, stack, discarded, disclosed, n_inf_tokens, n_fuse_tokens)
        q = q_net.predict(state.reshape(1, -1)).reshape(-1)
        q = (10 * q).astype(int).tolist()

        print ("\n"+"*"*100)
        print ("PLAYER " + str(player+1) + "'s TURN\n")
        print ("DECKS:          ", decks)
        print ("STACK:          ", stack)
        print ("DISCARDED:      ", discarded)
        print ("DISCLOSED:      ")
        print (disclosed[player*4:player*4+4])
        print ("FINAL:          ", f)
        # print ("DISCLOSED:      ", disclosed)
        print ("n_inf_tokens:   ", n_inf_tokens)
        print ("n_fuse_tokens:  ", n_fuse_tokens)
        print ("len rest:       ", 50 - len(discarded) - len(stack) - 20)
        print ("playable:       ", plb)
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
                print ("Hint: Player", p, "suit", k, " --- inds:", inds)
            else:
                inds = [i for i, (_, vv) in enumerate(decks[p]) if vv == k-4]
                print ("Hint: Player", p, "value", k-4, " --- inds:", inds)



        print ("\n" + "*"*100 + "\n")

        player = (player + 1) % 5

make_game_readable(S, A, R, F)



# learning rate
# beta (seems fine)
# WHY ARE Q VALS FOR DISCARDING CARDS SO HIGH?
# > something wrong with code, e.g., 'final'?
# > or mapping S-Q incorrect?
# > NO! This is bound to happen with Q-learning!
# > because of the max operation, simply staying alive accumulates reward (will to live)
# > rescale rewards?
# > lower GAMMA?
# > small negative reward each episode?
# > reward = 0 for discard but + gamma * max(Q_next)
# > initialize output layer weights to zeros
# > same as setting GAMMA=small in beginning?

# should input be [0, 1] instead of [-1, 1]? (don't think so)
# assign roles in beginning of game: play / discard / hint
# one final state was not updated ... (random_indices = ... len(Q)-1)
# (one out of 1000 games)
# now trying lower learning rate, one more layer

# weighted rules / handrcrafted heuristics
