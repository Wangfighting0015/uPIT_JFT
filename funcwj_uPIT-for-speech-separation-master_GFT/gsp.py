import numpy as np
import torch
import torch as th

def speechadj(Number):
    adjspeech = np.zeros(shape=(Number, Number))
    for i in range(Number):
        for j in range(Number):
            if i - j == 1:
                adjspeech[i][j] = 1
            else:
                adjspeech[i][j] = 0
    adjspeech[0][Number - 1] = 1

    return adjspeech


def graph_frouier_basis(Number):
    adjspeech = np.zeros(shape=(Number, Number))
    for i in range(Number):
        for j in range(Number):
            if i - j == 1:
                adjspeech[i][j] = 1
            else:
                adjspeech[i][j] = 0
    adjspeech[0][Number - 1] = 1
    adjspeech = adjspeech + adjspeech.T
    adjspeech = adjspeech + np.eye(adjspeech.shape[0])
    #print(adjspeech)
    U2, S, VT2 = np.linalg.svd(adjspeech)
    U = np.linalg.inv(U2)
    #U = tf.cast(U, tf.float32)
    #U = np.transpose(U)
    return U, S

'''def graph_frequncy(Number):
    adjspeech = np.zeros(shape=(Number, Number))
    for i in range(Number):
        for j in range(Number):
            if i - j == 1:
                adjspeech[i][j] = 1
            else:
                adjspeech[i][j] = 0
    adjspeech[0][Number - 1] = 1
    adjspeech = adjspeech + adjspeech.T
    adjspeech = adjspeech + np.eye(adjspeech.shape[0])
    print(adjspeech)
    U2, S, VT2 = np.linalg.svd(adjspeech)
    #print(S.shape)
    return S'''


def graph_frouier_basisA2(Number):
    A = np.ones(Number)
    W = np.diag(A)
    adj1 = circshift(W, 0, -1)
    adj2 = circshift(W, 0, -2)
    adj3 = circshift(W, 0, -3)
    adj = adj1 +adj2 +adj3
    #adjspeech = np.zeros(shape=(Number, Number))
    #####print(adj)
    U21, S1, VT21 = np.linalg.svd(adj)
    U2 = np.linalg.inv(U21)
    return U2, S1

'''def graph_freguencyA2(Number):
    A = np.ones(Number)
    W = np.diag(A)
    adj1 = circshift(W, 0, -1)
    adj2 = circshift(W, 0, -2)
    adj3 = circshift(W, 0, -3)
    adj = adj1 +adj2 +adj3
    #adjspeech = np.zeros(shape=(Number, Number))
    print(adj)
    U21, S2, VT21 = np.linalg.svd(adj)
    U1 = np.linalg.inv(U21)
    return S2'''

def graph_frouier_basisA3(Number):
    W = np.eye(Number)
    W1 = -1*np.diagflat(np.ones(Number-1, int), 1)
    W2 = 1 * np.diagflat(np.ones(Number-1, int), -1)
    adj13 = W + W1 + W2
    adj13[Number-1,0] = -1
    #print(adj13)
    U23, S3, VT23 = np.linalg.svd(adj13)
    U3 = np.linalg.inv(U23)
    return U3, S3

'''def graph_freguencyA3(Number):
    W = np.eye(Number)
    print(Number)
    W1 = -1*np.diagflat(np.ones(Number-1,int), 1)
    W2 = 1 * np.diagflat(np.ones(Number-1, int), -1)
    adj = W + W1 + W2
    adj[Number-1, 0] = -1
    print(adj.shape)
    U23, S3, VT23 = np.linalg.svd(adj)
    U3 = np.linalg.inv(U23)
    return S3'''

def graph_frouier_basisA4(Number):
    adjspeech_one = torch.FloatTensor(torch.zeros(Number, Number))
    lam = 0.8
    for i in range(Number):
        for j in range(Number):
                adjspeech_one[i][j] =(lam)**(np.abs(i-j))
                if adjspeech_one[i][j] < 0.1:
                   adjspeech_one[i][j] = 0
    #print(adjspeech_one)
    U24, S4, VT24 = np.linalg.svd(adjspeech_one)
    U4 = np.linalg.inv(U24)
    return U4, S4

'''def graph_freguencyA4(Number):
    A = np.ones(Number)
    W = np.diag(A)
    adj1 = circshift(W, 0, -1)
    adj2 = circshift(W, 0, -2)
    adj3 = circshift(W, 0, -3)
    adj = adj1 +adj2 +adj3
    #adjspeech = np.zeros(shape=(Number, Number))
    print(adj)
    U24, S4, VT24 = np.linalg.svd(adj)
    U1 = np.linalg.inv(U24)
   return S4'''


def circshift(a,downshift,rightshift):
    row, col = np.shape(a)
    downshift = ((downshift % row) + row) % row
    rightshift = ((rightshift % col) + col) % col
    b = np.zeros(shape=(row, col))
    for i in range(0,row):
        newrow = (i + downshift) % row
        for j in range (0, col):
            newcol = (j + rightshift) % col
            b[newrow, newcol]= a[i,j]
    return b


def inv_graph_frouier_basis(Number):
    adjspeech = np.zeros(shape=(Number, Number))
    for i in range(Number):
        for j in range(Number):
            if i - j == 1:
                adjspeech[i][j] = 1
            else:
                adjspeech[i][j] = 0
    adjspeech[0][Number - 1] = 1
    adjspeech = adjspeech + adjspeech.T
    adjspeech = adjspeech + np.eye(adjspeech.shape[0])
    V, Ss, V2 = np.linalg.svd(adjspeech)
    #U = np.linalg.inv(U2)
    #U = tf.cast(U, tf.float32)
    #U = np.transpose(U)
    return V

'''def GFT_STFT(STFT):
    nn1 = 257
    U = graph_frouier_basis(nn1)
    #print(U.shape)
    firstdimention = STFT.shape[0]
    framenumber = STFT.shape[1]
    if firstdimention == 1:
       GFT_STFT = np.zeros(shape=(firstdimention, framenumber, nn1))
       for i in range(framenumber):
            GFT_STFT[:, i, :] = np.transpose(np.dot(U, np.transpose(STFT[:, i, :])))

       #GFT_STFT = tf.cast(tf.convert_to_tensor(GFT_STFT), tf.float32)

    elif firstdimention == 8:
        # firstdimention1 = STFT.shape[0]
        # print(firstdimention1)
        # samplenumber = STFTGFT_STFT = np.zeros(shape=(firstdimention, nn1))
        GFT_STFT = np.zeros(shape=(firstdimention, framenumber, nn1))
        # print(STFT.shape)
        for i in range(firstdimention):
            for j in range(framenumber):
                GFT_STFT[i,j :] = np.transpose(np.dot(U, np.transpose(STFT[i,j, :])))
            # print(GFT_STFT[i, :].shape)
        GFT_STFT = tf.cast(tf.convert_to_tensor(GFT_STFT), tf.float32)
    else:
        GFT_STFT = np.zeros(shape=(firstdimention, nn1))
        for i in range(firstdimention):
            print((U).shape)
            print((STFT).shape)
            GFT_STFT[i, :] = np.transpose(np.dot(U, np.transpose(STFT[i, :])))
            print(GFT_STFT[i, :].shape)
        GFT_STFT = tf.cast(tf.convert_to_tensor(GFT_STFT), tf.float32)
    return GFT_STFT


def inverce_GFT_STFT(STFT):
    nn1 = 257
    U = graph_frouier_basis(nn1)
    V = U
    firstdimention = STFT.shape[0]
    inverse_GFT_STFT = np.zeros(shape=(firstdimention, nn1))
    for i in range(firstdimention):
        inverse_GFT_STFT[i, :] = np.transpose(np.dot(V,np.transpose(STFT[i, :])))
    inverse_GFT_STFT = tf.cast(tf.convert_to_tensor(inverse_GFT_STFT), tf.float32)
    #print(inverse_GFT_STFT.dtype)
    return inverse_GFT_STFT'''


#def inverse_graph_fourier_transform(Nn)

'''print(STFT.shape)
        nn = STFT.shape[1]
        nn1 = STFT.shape[2]
        JFT = np.zeros(shape=(nn, nn1))
        for i in range(nn):
            #nt = tf.convert_to_tensor(STFT[:, i, :])
            #nt1 = tf.convert_to_tensor(np.transpose(nt))
            #print(nt1)
            GFT_STFT = tf.convert_to_tensor(np.transpose(np.dot(tf.convert_to_tensor(np.transpose(U)), tf.convert_to_tensor(np.transpose(STFT[:,i,:])))))
            #print(GFT_STFT.dtype)
            #STFT1 = tf.convert_to_tensor(np.transpose(
                #np.dot(tf.convert_to_tensor(np.transpose(U)), tf.convert_to_tensor((STFT[:, i, :])))))
            JFT[i, :] = GFT_STFT
        #print(JFT.dtype)
        JFT = tf.expand_dims(tf.convert_to_tensor(JFT), 0)'''
