#include "header/ball.h"

Ball::Ball(int d) : center(Point(d)), radius(0.0) {}
Ball::Ball(const Point& center, double radius)
    : center(center), radius(radius) {}

bool Ball::contains(const Point& p) const {
  return center.distanceTo(p) <= radius;
}