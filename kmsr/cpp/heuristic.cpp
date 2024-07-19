#ifdef _OPENMP
  #include <omp.h>
#endif

#include <iostream>
#include <random>

#include "../header/ball.h"
#include "../header/util.h"
#include "../header/cluster.h"
#include "../header/k_MSR.h"
#include "../header/point.h"
#include "../header/welzl.h"
#include "../header/yildirim.h"

using namespace std;

vector<Cluster> gonzales(vector<Point> &points, int k, int seed)
{
  srand(seed);
  size_t n = points.size();
  vector<Point> centers;
  centers.push_back(points[rand() % n]);

  // Find the remaining k-1 centers
  for (int i = 1; i < k; i++)
  {
    int nextCenter = -1;
    double maxDist = -1.0;

    // Find the point that is farthest from its nearest center
    for (size_t j = 0; j < n; j++)
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

vector<Cluster> kMeansPlusPlus(vector<Point> &points, int k, int seed)
{
  size_t n = points.size();
  vector<Point> centers;
  mt19937 gen(seed);
  uniform_int_distribution<> dis(0, n - 1);

  // Choose the first center randomly
  centers.push_back(points[dis(gen)]);

  // Choose the remaining centers based on the distance distribution
  for (int i = 1; i < k; i++)
  {
    vector<double> dist(n, numeric_limits<double>::max());

    for (size_t j = 0; j < n; j++)
    {
      for (const Point &center : centers)
      {
        dist[j] = min(dist[j], points[j].distanceTo(center));
      }
    }

    // Calculate the probability distribution for selecting the next center
    vector<double> distSquared(n);
    double sumDist = 0.0;
    for (size_t j = 0; j < n; j++)
    {
      distSquared[j] = dist[j] * dist[j];
      sumDist += distSquared[j];
    }

    uniform_real_distribution<> disReal(0, sumDist);
    double r = disReal(gen);
    double cumulativeDist = 0.0;

    for (size_t j = 0; j < n; j++)
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
      if (clusters[i].getPoints().empty())
      {
        continue;
      }
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

vector<Cluster> heuristik(vector<Point> &points, int k)
{
  size_t n = points.size();
  vector<Cluster> bestCluster;
  bestCluster.push_back(
      Cluster(points)); // Initialize with all points in one cluster
  vector<vector<double>> distances(n, vector<double>(n, 0));

// Calculation of distances between all points
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < n; j++)
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
    for (size_t i = 0; i < n; i++)
    {
      for (size_t j = 0; j < n; j++)
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

          for (size_t h = 0; h < n; h++)
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
