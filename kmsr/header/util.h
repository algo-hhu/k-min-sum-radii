// util.h

#ifndef UTIL_H
#define UTIL_H

#include "ball.h"
#include "cluster.h"
#include "point.h"

bool containsPoint(const Point& p, const std::vector<Ball>& balls);

double logBase(double x, double b);

bool containsAllPoints(const std::vector<Point>& points,
                       const std::vector<Ball>& balls);

double cost(std::vector<Cluster>& cluster);

std::vector<Cluster> mergeCluster(std::vector<Cluster> &clusters);

std::vector<Cluster> assignPointsToCluster(const std::vector<Point> &points,
                                      const std::vector<Point> &centers, int k);

Point computeCentroid(const std::vector<Point> &points);
#endif  // UTIL_H
