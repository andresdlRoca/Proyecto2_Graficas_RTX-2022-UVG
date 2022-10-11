import math
from unittest import result

def matrixMult(A, B):
    result = [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0,0,0,0]]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def multiplyVectbyNum(vector, num):
    result = [0,0,0]
    for x in vector:
        result[int(x)] = x * num
    return result

def add(m1, m2):
    result= [0,0,0]
    for i in range(len(m1)):
        result[i] = m1[i] + m2[i]

    return result

def subtract(m1, m2):
    result= [0,0,0]
    for i in range(len(m1)):
        result[i] = m1[i] - m2[i]

    return result

def mulvects(m1, m2):
    result= [0,0,0]
    for i in range(len(m1)):
        result[i] = m1[i] * m2[i]

    return result

def dividevects(m1, m2):
    result= [0,0,0]
    for i in range(len(m1)):
        result[i] = m1[i] / m2[i]

    return result

def convertMatrix(row, col, data):
    mat = []
    for i in range(row):
        rowList = []
        for j in range(col):
            rowList.append(data[row * i + j])
        mat.append(rowList)

    return mat

def identity(num):
    matrix = []
    for row in range(0, num):
        matrix.append([])
        for col in range(0, num):
            if (row == col):
                matrix[row].append(1)
            else:
                matrix[row].append(0)
    return matrix

def dot(A, B):
    return sum([x*y for x,y in zip(A, B)])

def matMultVect(M, v):
    return [dot(r,v) for r in M]

def subtractArrays(A,B):
     dims = isinstance(A,list) + 2 * isinstance(B,list)
     if dims == 3:
         return [ subtractArrays(ra,rb) for ra,rb in zip(A,B) ]
     if dims == 2:
         return [ subtractArrays(A,rb) for rb in B ]
     if dims == 1: 
         return [ subtractArrays(ra,B) for ra in A ]
     return A-B

def crossProduct(A, B):
    Res = [A[1]*B[2] - A[2]*B[1],
         A[2]*B[0] - A[0]*B[2],
         A[0]*B[1] - A[1]*B[0]]

    return Res


def norm(list):
    # dist = math.sqrt(((list[0] - list[1]) ** 2)
    #                 + ((list[1] - list[2]) ** 2)
    #                 + ((list[2] - list[0]) ** 2))
    
    dist = math.sqrt((list[0] ** 2) + (list[1] ** 2) + (list[2] ** 2))
    
    return dist

def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a

def MatInv(a):
    tmp = [[] for _ in a]
    for i,row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i])//2:])
    return ret