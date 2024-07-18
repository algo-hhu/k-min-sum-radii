#include <cmath>
#include <vector>

#include "../header/yildirim.h"

using namespace std;

// Function to find the furthest point from a given point
int findFurthestPoint(const std::vector<Point> &points, const Point &p)
{
  int furthestIndex = 0;
  double maxDistSquared = 0.0;
  for (size_t i = 0; i < points.size(); i++)
  {
    // Calculate the squared distance between the current point and p
    double distSquared =
        Point::distance(points[i], p) * Point::distance(points[i], p);
    if (distSquared > maxDistSquared)
    {
      maxDistSquared = distSquared;
      furthestIndex = i;
    }
  }
  return furthestIndex;
}

// Function to calculate the weighted sum of the squares of the coordinates
double phi(const vector<Point> &points, const vector<double> &u)
{
  double sum = 0.0;
  for (size_t i = 0; i < points.size(); i++)
  {
    if (u[i] > 0)
    {
      // Calculate the weighted sum of the squares of the coordinates
      const vector<double> &coords = points[i].getCoordinates();
      for (size_t j = 0; j < coords.size(); j++)
      {
        sum += u[i] * coords[j] * coords[j];
      }
    }
  }
  return sum;
}

// Main function to calculate the Minimum Enclosing Ball
Ball findMEB(const vector<Point> &points, double epsilon)
{
  // Initialize the two furthest points alpha and beta
  int alpha = findFurthestPoint(points, points[0]);
  int beta = findFurthestPoint(points, points[alpha]);

  // Initialize the weights u
  vector<double> u(points.size(), 0);
  u[alpha] = 0.5;
  u[beta] = 0.5;

  // Calculate the initial center c
  Point c(vector<double>(points[0].getCoordinates().size(), 0.0));
  for (size_t i = 0; i < points.size(); i++)
  {
    c = c + (points[i] * u[i]);
  }

  // Calculate the initial gamma value
  double gamma = phi(points, u);

  // Find the furthest point kappa from the current center c
  int kappa = findFurthestPoint(points, c);

  // Calculate the initial delta value
  double delta = (Point::distance(points[kappa], c) *
                  Point::distance(points[kappa], c) / gamma) -
                 1;

  // Iteratively refine the center and weights
  while (delta > ((1 + epsilon) * (1 + epsilon)) - 1)
  {
    double lambda = delta / (2 * (1 + delta));

    // Update the weights u
    for (size_t i = 0; i < points.size(); i++)
    {
      u[i] = (1 - lambda) * u[i] + (static_cast<int>(i) == kappa ? lambda : 0);
    }

    // Update the center c
    vector<double> newCoordinates = c.getCoordinates();
    const vector<double> &kappaCoordinates = points[kappa].getCoordinates();
    for (size_t j = 0; j < newCoordinates.size(); j++)
    {
      newCoordinates[j] =
          (1 - lambda) * newCoordinates[j] + lambda * kappaCoordinates[j];
    }
    c.setCoordinates(newCoordinates);

    // Calculate the new gamma value
    gamma = phi(points, u);

    // Find the new furthest point kappa
    kappa = findFurthestPoint(points, c);

    // Calculate the new delta value
    delta = (Point::distance(points[kappa], c) *
             Point::distance(points[kappa], c) / gamma) -
            1;
  }

  // Calculate the final radius
  double radius = sqrt((1 + delta) * gamma);
  return Ball(c, radius);
}
