import math


def Cal_MCC(mat):
    TP = mat[0][0]
    TN = mat[1][1]
    FP = mat[0][1]
    FN = mat[1][0]
    bunshi = TP*TN - FP*FN
    bunbo = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + 1e-09
    return bunshi / bunbo

def F1(c_mat):
    c_precision = c_mat[0][0] / (1e-09 + c_mat[0][0] + c_mat[0][1])
    c_recall = c_mat[0][0] / (1e-09 + c_mat[0][0] + c_mat[1][0])
    c_f1 = (2 * c_recall * c_precision) / (1e-09 + c_recall + c_precision)
    nc_precision = c_mat[1][1] / (1e-09 + c_mat[1][1] + c_mat[1][0])
    nc_recall = c_mat[1][1] / (1e-09 + c_mat[1][1] + c_mat[0][1])
    nc_f1 = (2 * nc_recall * nc_precision) / (1e-09 + nc_recall + nc_precision)
    return (c_f1 + nc_f1) / 2

def F1_p(c_mat):
    c_precision = c_mat[0][0] / (1e-09 + c_mat[0][0] + c_mat[0][1])
    c_recall = c_mat[0][0] / (1e-09 + c_mat[0][0] + c_mat[1][0])
    c_f1 = (2 * c_recall * c_precision) / (1e-09 + c_recall + c_precision)
    return c_f1


q = int(input())
w = int(input())
e = int(input())
r = int(input())
a = [[q,w],[e,r]]
print(Cal_MCC(a))
print(F1_p(a))
print((q+r)/(q+w+e+r))

