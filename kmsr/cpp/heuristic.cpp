#include <omp.h>

#include <iostream>
#include <random>

#include "../header/ball.h"
#include "../header/cluster.h"
#include "../header/k_MSR.h"
#include "../header/point.h"
#include "../header/welzl.h"
#include "../header/yildirim.h"

using namespace std;

// Assigns each point in the 'points' vector to the nearest center in the 'centers' vector
vector<Cluster> assignPointsToCluster(const vector<Point> &points,
                                      const vector<Point> &centers, int k)
{
  int n = points.size();
  vector<Cluster> clusters(k);

  // Create clusters based on the centers
  for (int i = 0; i < n; i++)
  {
    int closestCenter = -1;
    double minDist = numeric_limits<double>::max();

    // Find the nearest center for each point
    for (int j = 0; j < k; j++)
    {
      double dist = points[i].distanceTo(centers[j]);
      if (dist < minDist)
      {
        minDist = dist;
        closestCenter = j;
      }
    }
    // Add the current point to its nearest cluster
    clusters[closestCenter].addPoint(points[i]);
  }
  return clusters;
}

// Checks if two clusters overlap or touch
bool clustersOverlap(const Cluster &c1, const Cluster &c2)
{
  // Calculate the minimum enclosing balls
  vector<Point> p1 = c1.getPoints();
  vector<Point> p2 = c2.getPoints();

  if (p1.size() == 0 || p2.size() == 0)
    return false;
  Ball b1 = findMinEnclosingBall(p1);
  Ball b2 = findMinEnclosingBall(p2);

  // Calculate the Euclidean distance between the centers of the two balls
  double distance = Point::distance(b1.getCenter(), b2.getCenter());

  // Calculate the sum of the radii of the two balls
  double radiusSum = b1.getRadius() + b2.getRadius();

  // Check if the distance between the centers is less than or equal to the sum of the radii
  return distance <= radiusSum;
}

// Merges overlapping or touching clusters
vector<Cluster> mergeCluster(vector<Cluster> &clusters)
{
  bool changed;

  // Repeat the merge process until no more clusters are merged
  do
  {
    changed = false;
    vector<Cluster> mergedClusters;
    vector<bool> merged(clusters.size(), false);

    for (size_t i = 0; i < clusters.size(); i++)
    {
      if (merged[i])
      {
        continue; // Skip already merged clusters
      }
      Cluster currentCluster = clusters[i];
      merged[i] = true;

      for (size_t j = i + 1; j < clusters.size(); j++)
      {
        if (merged[j])
        {
          continue; // Skip already merged clusters
        }
        if (clustersOverlap(currentCluster, clusters[j]))
        {
          currentCluster.merge(clusters[j]);
          merged[j] = true;
          changed = true; // There was a change
        }
      }
      mergedClusters.push_back(currentCluster); // Add the merged cluster to the merged clusters
    }

    clusters = mergedClusters; // Update the cluster list

  } while (changed);

  return clusters;
}

// Computes the centroid of the cluster
Point computeCentroid(const vector<Point> &points)
{
  int dimension = points[0].getCoordinates().size();
  vector<double> centroidCoords(dimension, 0.0);

  // Sum of the coordinates of all points in the cluster
  for (const Point &p : points)
  {
    for (int i = 0; i < dimension; i++)
    {
      centroidCoords[i] += p.getCoordinates()[i];
    }
  }

  // Calculate the mean of the coordinates
  for (int i = 0; i < dimension; i++)
  {
    centroidCoords[i] /= points.size();
  }

  return Point(centroidCoords);
}

vector<Cluster> gonzales(vector<Point> &points, int k)
{
  srand(1234);
  int n = points.size();
  vector<Point> centers;
  centers.push_back(points[rand() % n]);

  // Find the remaining k-1 centers
  for (int i = 1; i < k; i++)
  {
    int nextCenter = -1;
    double maxDist = -1.0;

    // Find the point that is farthest from its nearest center
    for (int j = 0; j < n; j++)
    {
      double dist = numeric_limits<double>::max();
      for (Point center : centers)
      {
        dist = min(dist, points[j].distanceTo(center));
      }
      if (dist > maxDist)
      {
        maxDist = dist;
        nextCenter = j;
      }
    }
    centers.push_back(points[nextCenter]);
  }

  // Assign the points to the centers and create clusters
  vector<Cluster> clusters = assignPointsToCluster(points, centers, k);

  // Merge overlapping or touching clusters
  return mergeCluster(clusters);
}

vector<Cluster> kMeansPlusPlus(vector<Point> &points, int k)
{
  int n = points.size();
  vector<Point> centers;
  mt19937 gen(1234);
  uniform_int_distribution<> dis(0, n - 1);

  // Choose the first center randomly
  centers.push_back(points[dis(gen)]);

  // Choose the remaining centers based on the distance distribution
  for (int i = 1; i < k; i++)
  {
    vector<double> dist(n, numeric_limits<double>::max());

    for (int j = 0; j < n; j++)
    {
      for (const Point &center : centers)
      {
        dist[j] = min(dist[j], points[j].distanceTo(center));
      }
    }

    // Calculate the probability distribution for selecting the next center
    vector<double> distSquared(n);
    double sumDist = 0.0;
    for (int j = 0; j < n; j++)
    {
      distSquared[j] = dist[j] * dist[j];
      sumDist += distSquared[j];
    }

    uniform_real_distribution<> disReal(0, sumDist);
    double r = disReal(gen);
    double cumulativeDist = 0.0;

    for (int j = 0; j < n; j++)
    {
      cumulativeDist += distSquared[j];
      if (cumulativeDist >= r)
      {
        centers.push_back(points[j]);
        break;
      }
    }
  }

  vector<Cluster> clusters;
  bool changed = true;
  while (changed)
  {
    changed = false;

    // Assign the points to the centers and create clusters
    clusters = assignPointsToCluster(points, centers, k);

    // Update the centers based on the clusters
    for (int i = 0; i < k; i++)
    {
      Point newCenter = computeCentroid(clusters[i].getPoints());
      if (newCenter != centers[i])
      {
        centers[i] = newCenter;
        changed = true;
      }
    }
  }

  // Merge overlapping or touching clusters
  return mergeCluster(clusters);
}

vector<Cluster> heuristic(vector<Point> &points, int k)
{
  int n = points.size();
  vector<Cluster> bestCluster;
  bestCluster.push_back(
      Cluster(points)); // Initialize with all points in one cluster
  vector<vector<double>> distances(n, vector<double>(n, 0));

  // Calculation of distances between all points
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      distances[i][j] = Point::distance(points[i], points[j]);
    }
  }

#pragma omp parallel
  {
    vector<Cluster> localBestCluster =
        bestCluster; // Local variable for the best clusters in each thread
    double localBestCost =
        cost(localBestCluster); // Cost of the local best clusters

#pragma omp for
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        vector<Point> centers;
        Point largestCenter = points[i];
        centers.push_back(largestCenter);
        double radius = distances[i][j];

        // Find k centers
        while (static_cast<int>(centers.size()) != k)
        {
          int nextCenter = -1;
          double maxDist = -1.0;

          for (int h = 0; h < n; h++)
          {
            double dist = numeric_limits<double>::max();
            for (const Point &center : centers)
            {
              if (center == largestCenter)
              {
                dist = min(dist, points[h].distanceTo(center) - radius);
              }
              else
              {
                dist = min(dist, points[h].distanceTo(center));
              }
            }
            if (dist > maxDist)
            {
              maxDist = dist;
              nextCenter = h;
            }
          }
          centers.push_back(points[nextCenter]);
        }

        // Assign the points to the nearest centers
        vector<Cluster> cluster = assignPointsToCluster(points, centers, k);
        double clusterCost =
            cost(cluster); // Calculate the cost of the current cluster

        // Update local best clusters if the current cluster is better
        if (clusterCost < localBestCost)
        {
          localBestCluster = cluster;
          localBestCost = clusterCost;
        }
      }
    }
#pragma omp critical
    {
      if (localBestCost < cost(bestCluster))
      {
        bestCluster = localBestCluster;
      }
    }
  }

  // Merge overlapping or touching clusters
  return mergeCluster(bestCluster);
}
