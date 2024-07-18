#include "../header/cluster.h"

void Cluster::merge(const Cluster& other) {
  points.insert(points.end(), other.points.begin(), other.points.end());
}
