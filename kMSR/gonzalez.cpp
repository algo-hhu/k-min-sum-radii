#include "header/gonzalez.h"

#include <algorithm>

#include "header/point.h"

using namespace std;

double gonzalezrmax(const vector<Point> &points, int k, int seed) {
  srand(seed);
  vector<Point> centers;

  // Wähle den ersten Punkt zufällig aus den Eingabepunkten
  centers.push_back(points[rand() % points.size()]);

  vector<double> distances(points.size());

  // Berechne den maximalen Abstand für die initiale Zuordnung
  for (int i = 0; i < points.size(); i++) {
    distances[i] = Point::distance(points[i], centers[0]);
  }

  // Iteriere, um die restlichen k - 1 Zentren zu finden
  while (centers.size() < k) {
    // Wähle das nächste Zentrum basierend auf dem größten Abstand
    int nextCenterIndex = distance(
        distances.begin(), max_element(distances.begin(), distances.end()));
    Point nextCenter = points[nextCenterIndex];
    centers.push_back(nextCenter);

    // Aktualisiere die Abstände für jeden Punkt zu seinem nächsten Zentrum
    for (int i = 0; i < points.size(); i++) {
      distances[i] = min(distances[i], Point::distance(points[i], nextCenter));
    }
  }

  // Berechne den maximalen Abstand der Punkte zu ihren Zentren
  double maxDistance = 0.0;
  for (int i = 0; i < points.size(); i++) {
    maxDistance = max(maxDistance, distances[i]);
  }

  return maxDistance;
}