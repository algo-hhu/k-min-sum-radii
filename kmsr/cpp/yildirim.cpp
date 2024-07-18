#include <cmath>
#include <vector>

#include "../header/yildirim.h"

using namespace std;

// Funktion zur Bestimmung des am weitesten entfernten Punkts von einem
// gegebenen Punkt
int findFurthestPoint(const std::vector<Point> &points, const Point &p)
{
  int furthestIndex = 0;
  double maxDistSquared = 0.0;
  for (size_t i = 0; i < points.size(); i++)
  {
    // Berechne die quadrierte Distanz zwischen dem aktuellen Punkt und p
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

// Funktion zur Berechnung der gewichteten Summe der Quadrate der Koordinaten
double phi(const vector<Point> &points, const vector<double> &u)
{
  double sum = 0.0;
  for (size_t i = 0; i < points.size(); i++)
  {
    if (u[i] > 0)
    {
      // Berechne die gewichtete Summe der Quadrate der Koordinaten
      const vector<double> &coords = points[i].getCoordinates();
      for (size_t j = 0; j < coords.size(); j++)
      {
        sum += u[i] * coords[j] * coords[j];
      }
    }
  }
  return sum;
}

// Hauptfunktion zur Berechnung des Minimum Enclosing Ball
Ball findMEB(const vector<Point> &points, double epsilon)
{
  // Initialisierung der beiden am weitesten entfernten Punkte alpha und beta
  int alpha = findFurthestPoint(points, points[0]);
  int beta = findFurthestPoint(points, points[alpha]);

  // Initialisierung der Gewichtungen u
  vector<double> u(points.size(), 0);
  u[alpha] = 0.5;
  u[beta] = 0.5;

  // Berechnung des initialen Zentrums c
  Point c(vector<double>(points[0].getCoordinates().size(), 0.0));
  for (size_t i = 0; i < points.size(); i++)
  {
    c = c + (points[i] * u[i]);
  }

  // Berechnung des initialen gamma-Werts
  double gamma = phi(points, u);

  // Bestimmung des am weitesten entfernten Punkts kappa vom aktuellen Zentrum c
  int kappa = findFurthestPoint(points, c);

  // Berechnung des initialen delta-Werts
  double delta = (Point::distance(points[kappa], c) *
                  Point::distance(points[kappa], c) / gamma) -
                 1;

  // Iterative Verfeinerung des Zentrums und der Gewichtungen
  while (delta > ((1 + epsilon) * (1 + epsilon)) - 1)
  {
    double lambda = delta / (2 * (1 + delta));

    // Aktualisierung der Gewichtungen u
    for (size_t i = 0; i < points.size(); i++)
    {
      u[i] = (1 - lambda) * u[i] + (static_cast<int>(i) == kappa ? lambda : 0);
    }

    // Aktualisierung des Zentrums c
    vector<double> newCoordinates = c.getCoordinates();
    const vector<double> &kappaCoordinates = points[kappa].getCoordinates();
    for (size_t j = 0; j < newCoordinates.size(); j++)
    {
      newCoordinates[j] =
          (1 - lambda) * newCoordinates[j] + lambda * kappaCoordinates[j];
    }
    c.setCoordinates(newCoordinates);

    // Berechnung des neuen gamma-Werts
    gamma = phi(points, u);

    // Bestimmung des neuen am weitesten entfernten Punkts kappa
    kappa = findFurthestPoint(points, c);

    // Berechnung des neuen delta-Werts
    delta = (Point::distance(points[kappa], c) *
             Point::distance(points[kappa], c) / gamma) -
            1;
  }

  // Berechnung des finalen Radius
  double radius = sqrt((1 + delta) * gamma);

  return Ball(c, radius);
}
