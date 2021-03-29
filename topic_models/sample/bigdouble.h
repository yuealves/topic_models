#ifndef __BIGDOUBLE_H
#define __BIGDOUBLE_H

#include <climits>
#include <string>


class BigDouble {
public:
  double get_large_double();
  double get_small_double();
  BigDouble();
  BigDouble(double val);
  BigDouble(double val, int exp);
  BigDouble &operator+=(BigDouble rhs);
  BigDouble &operator*=(BigDouble rhs);
  BigDouble &operator/=(BigDouble rhs);
  void imul(BigDouble *rhs);
  void idiv(BigDouble *rhs);
  bool operator>(BigDouble rhs) const;
  BigDouble operator*(BigDouble rhs) const;
  BigDouble operator/(BigDouble rhs) const;
  explicit operator double() const;
  void print() const;
  std::string repr();


private:
  constexpr static double LARGE_DOUBLE = 1e100;
  constexpr static double SMALL_DOUBLE = 1 / LARGE_DOUBLE;
  constexpr static int EXP_MAX = INT_MAX / 2;
  constexpr static int EXP_MIN = INT_MIN / 2;
  double val;
  int exp;

  void rescale();
};
#endif // __BIGDOUBLE_H
