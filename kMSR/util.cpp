#include "header/util.h"

#include <limits>
#include <iostream>

#include "header/welzl.h"

// Calculates the logarithm of a number 'x' with base 'b'.
double logBase(double x, double b) { return log(x) / log(b); }
// Checks if two clusters overlap or touch
bool clustersOverlap(const Cluster &c1, const Cluster &c2) {
  // Calculate the minimum enclosing balls
  std::vector<Point> p1 = c1.getPoints();
  std::vector<Point> p2 = c2.getPoints();

  if (p1.size() == 0 || p2.size() == 0) return false;
  Ball b1 = findMinEnclosingBall(p1);
  Ball b2 = findMinEnclosingBall(p2);

  // Calculate the Euclidean distance between the centers of the two balls
  double distance = Point::distance(b1.getCenter(), b2.getCenter());

  // Calculate the sum of the radii of the two balls
  double radiusSum = b1.getRadius() + b2.getRadius();

  // Check if the distance between the centers is less than or equal to the sum
  // of the radii
  return distance <= radiusSum;
}

// Assigns each point in the 'points' vector to the nearest center in the
// 'centers' vector
std::vector<Cluster> assignPointsToCluster(const std::vector<Point> &points,
                                           const std::vector<Point> &centers,
                                           int k) {
  int n = points.size();
  std::vector<Cluster> clusters(k);

  // Create clusters based on the centers
  for (int i = 0; i < n; i++) {
    int closestCenter = -1;
    double minDist = std::numeric_limits<double>::max();

    // Find the nearest center for each point
    for (int j = 0; j < k; j++) {
      double dist = points[i].distanceTo(centers[j]);
      if (dist < minDist) {
        minDist = dist;
        closestCenter = j;
      }
    }
    // Add the current point to its nearest cluster
    clusters[closestCenter].addPoint(points[i]);
  }
  return clusters;
}

// Computes the centroid of the cluster
Point computeCentroid(const std::vector<Point> &points) {
  int dimension = points[0].getCoordinates().size();
  std::vector<double> centroidCoords(dimension, 0.0);

  // Sum of the coordinates of all points in the cluster
  for (const Point &p : points) {
    for (int i = 0; i < dimension; i++) {
      centroidCoords[i] += p.getCoordinates()[i];
    }
  }

  // Calculate the mean of the coordinates
  for (int i = 0; i < dimension; i++) {
    centroidCoords[i] /= points.size();
  }
  return Point(centroidCoords);
}

// Checks if every point in the 'points' list is contained in at least one ball
// in 'balls'.
bool containsAllPoints(const std::vector<Point> &points,
                       const std::vector<Ball> &balls) {
  for (const Point &p : points) {
    bool isContained = false;
    for (const Ball &b : balls) {
      if (b.contains(p)) {
        isContained = true;
        break;
      }
    }
    if (!isContained) {
      return false;  // Early exit if a point is not contained in any ball
    }
  }

  return true;  // All points are contained in at least one ball
}

// Checks if the point 'p' is contained in at least one ball in the 'balls'
// list.
bool containsPoint(const Point &p, const std::vector<Ball> &balls) {
  for (const Ball &b : balls) {
    if (b.contains(p)) {
      return true;
    }
  }
  return false;
}

// Calculates the total cost for all clusters based on the radius of the
// smallest enclosing ball.
double cost(std::vector<Cluster> &cluster) {
  double result = 0;

  for (Cluster &c : cluster) {
    if (!c.getPoints().empty()) {
      result += findMinEnclosingBall(c.getPoints()).getRadius();
    }
  }
  return result;
}

// Merges overlapping or touching clusters
std::vector<Cluster> mergeCluster(std::vector<Cluster> &clusters) {
  bool changed;

  // Repeat the merge process until no more clusters are merged
  do {
    changed = false;
    std::vector<Cluster> mergedClusters;
    std::vector<bool> merged(clusters.size(), false);

    for (size_t i = 0; i < clusters.size(); i++) {
      if (merged[i]) {
        continue;  // Skip already merged clusters
      }
      Cluster currentCluster = clusters[i];
      merged[i] = true;

      for (size_t j = i + 1; j < clusters.size(); j++) {
        if (merged[j]) {
          continue;  // Skip already merged clusters
        }
        if (clustersOverlap(currentCluster, clusters[j])) {
          currentCluster.merge(clusters[j]);
          merged[j] = true;
          changed = true;  // There was a change
        }
      }
      mergedClusters.push_back(
          currentCluster);  // Add the merged cluster to the merged clusters
    }

    clusters = mergedClusters;  // Update the cluster list

  } while (changed);

  return clusters;
}