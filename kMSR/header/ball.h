// ball.h

#ifndef BALL_H
#define BALL_H

#include "point.h"

class Ball {
 public:
  Ball() : center({0, 0}), radius(0.0) {}
  Ball(const Point& center, double radius) : center(center), radius(radius) {}

  bool contains(const Point& p) const;

  const Point& getCenter() const { return center; }
  double getRadius() const { return radius; }

  void setCenter(const Point& newCenter) { center = newCenter; }
  void setRadius(double newRadius) { radius = newRadius; }

 private:
  Point center;
  double radius;
};

#endif  // BALL_H