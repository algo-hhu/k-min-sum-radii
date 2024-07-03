// cluster.h

#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>

#include "point.h"

class Cluster {
 public:
  Cluster() {}
  Cluster(const std::vector<Point>& points) : points(points) {}

  void addPoint(const Point& p) { points.push_back(p); }
  const std::vector<Point>& getPoints() const { return points; }
  void setPoints(const std::vector<Point>& newPoints) { points = newPoints; }
  void merge(const Cluster& other);

 private:
  std::vector<Point> points;
};

#endif  // CLUSTER_H