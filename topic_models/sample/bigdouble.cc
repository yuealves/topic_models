#include "bigdouble.h"
#include <cstdio>
#include <cassert>
#include <cmath>
#include <string>
#include <stdexcept>

BigDouble::BigDouble(): val(0), exp(0) {}
BigDouble::BigDouble(double val): val(val), exp(0) { rescale(); }
BigDouble::BigDouble(double val, int exp): val(val), exp(exp) { rescale(); }

bool BigDouble::operator>(BigDouble rhs) const {
    if (val !=0 && rhs.val ==0) {
        return true;
    }
    if (val == 0) {
        return false;
    }
    if (exp > rhs.exp) {
        return true;
    }
    if (exp < rhs.exp) {
        return false;
    }
    return val > rhs.val;
}
BigDouble &BigDouble::operator+=(BigDouble rhs) {
    if (rhs.val == 0) return *this;
    if (val == 0) {
        val = rhs.val;
        exp = rhs.exp;
        return *this;
    }
    int exp_diff = exp - rhs.exp;
    while (exp_diff > 0) {
        rhs.val /= LARGE_DOUBLE;
        exp_diff--;
    }
    while (exp_diff < 0) {
        val /= LARGE_DOUBLE;
        exp_diff++;
        exp++;
    }
    val += rhs.val;
    return *this;
}

BigDouble &BigDouble::operator*=(BigDouble rhs) {
    val *= rhs.val;
    exp += rhs.exp;
    if (val > LARGE_DOUBLE) {
        val /= LARGE_DOUBLE;
        exp++;
    }
    if (exp >= EXP_MAX) {
      throw std::overflow_error("Value overflow:" + repr());
    }
    return *this;
}

void BigDouble::imul(BigDouble *rhs) {
    (*this) *= *rhs;
}

void BigDouble::imul(double val) {
    this->val *= val;
    if (this->val > LARGE_DOUBLE) {
        this->val /= LARGE_DOUBLE;
        exp++;
    }
    if (exp >= EXP_MAX) {
      throw std::overflow_error("Value overflow:" + repr());
    }
}

void BigDouble::idiv(BigDouble *rhs) {
    (*this) /= *rhs;
}

BigDouble &BigDouble::operator/=(BigDouble rhs) {
    val /= rhs.val;
    exp -= rhs.exp;
    if (val < 1) {
        val *= LARGE_DOUBLE;
        exp--;
    }
    assert(exp > EXP_MIN);
    if (exp <= EXP_MIN) {
      throw std::underflow_error("Value underflow:" + repr());
    }
    return *this;
}

BigDouble BigDouble::operator*(BigDouble rhs) const {
    BigDouble result = *this;
    result *= rhs;
    return result;
}

BigDouble BigDouble::operator/(BigDouble rhs) const {
    BigDouble result = *this;
    result /= rhs;
    return result;
}

BigDouble::operator double() const {
    double result = val;
    int e = exp;
    while (e > 0) {
        result *= LARGE_DOUBLE;
        e--;
    }
    while (e < 0) {
        result /= LARGE_DOUBLE;
        e++;
    }
    return result;
}

std::string BigDouble::repr() {
    return "(" + std::to_string(val) +", " + std::to_string(exp) + ")";
}

void BigDouble::rescale() {
    // rescale to [1, LARGE_DOUBLE)
    // assert(!std::isnan(val) && !std::isinf(val) && val >=0);
    if (std::isnan(val) || std::isinf(val)) {
        throw std::range_error("Invalid value:" + repr());
    }
    if (val == 0) {
        exp = 0;
        return;
    }
    while (val >= LARGE_DOUBLE) {
        val /= LARGE_DOUBLE;
        exp++;
    }
    while (val < 1) {
        val *= LARGE_DOUBLE;
        exp--;
    }
}

double BigDouble::get_val() const {
    return val;
}

int BigDouble::get_exp() const {
    return exp;
}

void BigDouble::set_val(double val) {
    this->val = val;
}

void BigDouble::set_exp(int exp) {
    this->exp = exp;
}

void normalize_probs(BigDouble *probs, double *result, int K) {
    BigDouble sum = 0;
    for (int i = 0; i < K; i++) {
        sum += probs[i];
    }
    for (int i = 0; i < K; i++) {
        result[i] = (double) (probs[i] / sum);
    }
    return;
}