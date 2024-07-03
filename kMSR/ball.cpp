#include "header/ball.h"

bool Ball::contains(const Point &p) const {
  return center.distanceTo(p) <= radius;
}