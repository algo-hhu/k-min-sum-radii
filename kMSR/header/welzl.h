// welzl.h

#ifndef WELZL_H
#define WELZL_H

#include "ball.h"
#include "point.h"

Ball welzlHelper(std::vector<Point>& points, std::vector<Point> support, int n);
Ball makeBallThreePoints(const Point& A, const Point& B, const Point& C);
Ball makeBallTwoPoints(const Point& A, const Point& B);
Ball findMinEnclosingBall(const std::vector<Point>& points);

#endif  // WELZL_H