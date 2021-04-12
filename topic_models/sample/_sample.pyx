# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free, srand, rand, RAND_MAX
import numpy as np
cimport numpy as np

from topic_models.utils.bigdouble cimport BigDouble, normalize_probs

cpdef _lda_train(int[:,:] n_mk, int[:,:] n_kt, int [:] n_kt_sum,
                 int[:] W, int[:] Z, int[:] N_m, int[:] I_m,
                 double alpha, double beta, int n_fixed):
    cdef:
        int m, i, num_z_change = 0
        int M = n_mk.shape[0]
        int K = n_kt.shape[0]
        int T = n_kt.shape[1]
        double *theta_estimate = <double *>malloc(K*sizeof(double))
        double *phi_t_estimate = <double *>malloc(K*sizeof(double))
        double *pval = <double *>malloc(K*sizeof(double))

    for m in range(n_fixed, M):
        for i in range(N_m[m]):
            num_z_change += _lda_sample_word_topic(n_mk, n_kt, n_kt_sum, W, Z, I_m, N_m, m, i, alpha, beta, theta_estimate, phi_t_estimate, pval)

    free(theta_estimate)
    free(phi_t_estimate)
    free(pval)
    return num_z_change


cdef _lda_sample_word_topic(int[:,:] n_mk, int[:,:] n_kt, int[:] n_kt_sum,
                            int[:] W, int[:] Z, int[:] I_m, int[:] N_m,
                            int m, int i,
                            double alpha, double beta,
                            double *theta_estimate, double *phi_t_estimate, double *pval):
    cdef:
        int k, t
        double cum
        int w = W[I_m[m] + i]
        int z = Z[I_m[m] + i]
        int z_orig = z
        int K = n_kt.shape[0]
        int T = n_kt.shape[1]

    dec(n_mk[m, z])
    dec(n_kt[z, w])
    dec(n_kt_sum[z])

    cum = N_m[m] + K * alpha
    for k in range(K):
        theta_estimate[k] = (n_mk[m,k] + alpha) / cum

    for k in range(K):
        cum = n_kt_sum[k]
        phi_t_estimate[k] = (n_kt[k, w] + beta)/(cum + T * beta)
        pval[k] = theta_estimate[k] * phi_t_estimate[k]

    cum = 0
    for k in range(K):
        cum += pval[k]
    for k in range(K):
        pval[k] /= cum


    cum = <double>rand() / <double>RAND_MAX

    z = K - 1
    for k in range(K):
        cum -= pval[k]
        if cum < 0:
            z = k
            break

    Z[I_m[m]+i] = z
    inc(n_kt[z, w])
    inc(n_mk[m, z])
    inc(n_kt_sum[z])
    if z != z_orig:
        return 1
    else:
        return 0

cpdef _dmm_train(int[:,:] n_kt, int[:] n_kt_sum, int[:] M_k, int[:] I_m, int[:] W_m,
                 int[:] W_m_freq, int[:] Z, int[:] N_m, int[:] U_m, double alpha,
                 double beta, int n_fixed):
    cdef:
        int m, i1, i2, num_z_change = 0
        int M = N_m.shape[0]
        int K = n_kt.shape[0]
        int T = n_kt.shape[1]
        double *pval = <double *>malloc(K*sizeof(double))
        BigDouble *probs = <BigDouble *>malloc(K*sizeof(BigDouble))

    for m in range(n_fixed, M):
        i1 = I_m[m]
        i2 = i1 + U_m[m]
        num_z_change += _dmm_sample_doc_topic(n_kt, n_kt_sum, M_k, W_m[i1:i2], W_m_freq[i1:i2],
                            Z, N_m[m], U_m[m], alpha, beta, m, probs, pval)
    
    free(probs)
    free(pval)
    return num_z_change

cdef _dmm_sample_doc_topic(int[:,:] n_kt, int[:] n_kt_sum, int[:] M_k,
              int[:] W_m, int[:] W_m_freq, int[:] Z, int N_m, int U_m, double alpha, 
              double beta, int m, BigDouble *probs, double *pval):
    # n_kt 是第k个topic中单词t出现的次数
    # M_k 是各个topic对应的文档数目向量，长度为K
    # W_m 是第m篇文档对应的word_id_list，长度为第m篇文档中unique word token数
    # W_m_freq 对应W_m中各word_id出现的次数
    # Z 是全部m篇文档对应的topic编号向量，长度为M
    # N_m 第m篇文档的长度
    # U_m 第m篇文档中unique word token数
    # probs是以BigDouble形式储存的第m篇文档落入各个topic的概率数组的指针
    # pval是对probs做了归一化后的概率数组的指针
    cdef:
        int k, t, i
        double cum, tmp
        int z = Z[m]
        int z_orig = z
        int K = n_kt.shape[0]
        int T = n_kt.shape[1]
        BigDouble denominator
    
    n_kt_sum[z] -= N_m
    dec(M_k[z])
    for i in range(U_m):
        n_kt[z, W_m[i]] -= W_m_freq[i]

    for k in range(K):
        probs[k] = BigDouble(M_k[k] + alpha, 0)
        denominator = BigDouble(1, 0)
        for i in range(U_m):
            for j in range(W_m_freq[i]):
                tmp = n_kt[k, W_m[i]] + beta + j
                probs[k].imul(tmp)
        for i in range(N_m):
            denominator.imul(n_kt_sum[k] + T * beta + i)
        probs[k].idiv(&denominator)
    normalize_probs(probs, pval, K)

    cum = <double>rand() / <double>RAND_MAX

    z = K - 1
    for k in range(K):
        cum -= pval[k]
        if cum < 0:
            z = k
            break
        
    Z[m] = z
    n_kt_sum[z] += N_m
    inc(M_k[z])
    for i in range(U_m):
        n_kt[z, W_m[i]] += W_m_freq[i]

    if z != z_orig:
        return 1
    else:
        return 0


cdef _lda_predict_word(double[:,:] phi, int[:] doc, int[:] n_z, int n, int N,
                       double alpha, double *theta_estimate, double *pval):
    cdef:
        int k, t, z
        int w_i = doc[n]
        int K = phi.shape[0]
        int T = phi.shape[1]
        double cum = N + K * alpha

    cum = N + K * alpha
    for k in range(K):
        theta_estimate[k] = (n_z[k] + alpha) / cum
        pval[k] = theta_estimate[k] * phi[k, w_i]

    cum = 0
    for k in range(K):
        cum += pval[k]
    for k in range(K):
        pval[k] /= cum

    cum = <double>rand() / <double>RAND_MAX

    z = K - 1
    for k in range(K):
        cum -= pval[k]
        if cum < 0:
            z = k
            break
    return z


cpdef _lda_predict_doc(double[:,:] phi, int[:] doc, int[:] z, int[:] n_z, double alpha):
    cdef:
        int n, z_i_old, z_i_new
        int N = len(doc)
        int K = phi.shape[0]
        double *theta_estimate = <double *>malloc(K*sizeof(double))
        double *pval = <double *>malloc(K*sizeof(double))

    for n in range(N):
        z_i_old = z[n]
        dec(n_z[z_i_old])
        z_i_new = _lda_predict_word(phi, doc, n_z, n, N, alpha,
                                theta_estimate, pval)
        inc(n_z[z_i_new])
        z[n] = z_i_new

    free(theta_estimate)
    free(pval)
    return z

def demo():
    cdef BigDouble bd[3]
    cdef double result[3]
    result[0] = 0.2
    bd[0] = BigDouble(2, 3)
    bd[1] = BigDouble(2, 3)
    bd[2] = BigDouble(2, 3)
    normalize_probs(bd, result, 3)
    print(bd[0].repr().decode())
    print(result[0])