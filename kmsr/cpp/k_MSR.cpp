#include <omp.h>

#include <random>

#include "../header/k_MSR.h"
#include "../header/gonzales.h"
#include "../header/welzl.h"
#include "../header/yildirim.h"

using namespace std;

// Berechnet den Logarithmus einer Zahl 'x' zur Basis 'b'.
double logBase(double x, double b) { return log(x) / log(b); }

// Prüft, ob jeder Punkt in der Liste 'points' von mindestens einem Ball in
// 'balls' enthalten ist.
bool containsAllPoints(const vector<Point> &points, const vector<Ball> &balls) {
  for (const Point &p : points) {
    bool isContained = false;
    for (const Ball &b : balls) {
      if (b.contains(p)) {
        isContained = true;
        break;
      }
    }
    if (!isContained) {
      return false;  // Frühes Beenden, wenn ein Punkt von keinem Ball enthalten
                     // ist
    }
  }

  return true;  // Alle Punkte sind von mindestens einem Ball enthalten
}

// Überprüft, ob der Punkt 'p' von mindestens einem Ball in der Liste 'balls'
// enthalten ist.
bool containsPoint(const Point &p, const vector<Ball> &balls) {
  for (const Ball &b : balls) {
    if (b.contains(p)) {
      return true;
    }
  }
  return false;
}

// Berechnet die Gesamtkosten für alle Cluster basierend auf dem Radius des
// kleinsten einschließenden Balls.
double cost(vector<Cluster> &cluster) {
  double result = 0;

  for (Cluster &c : cluster) {
    if (!c.getPoints().empty()) {
      result += findMinEnclosingBall(c.getPoints()).getRadius();
    }
  }
  return result;
}

// Berechnet ein Vektor von Vektoren von Radien für einen gegebenen maximalen
// Radius, eine Anzahl von Bällen k und eine Genauigkeit epsilon.
vector<vector<double>> getRadii(double rmax, int k, double epsilon) {
  vector<vector<double>> result;
  vector<int> indices(k - 1, 0);
  vector<double> set;

  // Berechnung der Anzahl der Radien, die benötigt werden, um eine ausreichende
  // Abdeckung sicherzustellen
  int limit = ceil(logBase((k / epsilon), (1 + epsilon)));

  // Erstelle das Set der Radien, deren Permutationen gebildet werden.
  for (int i = 0; i <= limit; i++) {
    set.push_back(pow((1 + epsilon), i) * (epsilon / k) * rmax);
  }

  // Erstelle alles möglichen Permutationen von 'set' mit 'rmax' als erstem
  // Element.
  while (true) {
    vector<double> current;

    // Der maximale Radius wird immer als erster Radius in der Kombination
    // hinzugefügt
    current.push_back(rmax);

    for (int idx : indices) {
      current.push_back(set[idx]);
    }
    result.push_back(current);

    int next = k - 2;
    while (next >= 0 && ++indices[next] == static_cast<int>(set.size())) {
      indices[next] = 0;
      next--;
    }
    if (next < 0) {
      break;
    }
  }

  return result;
}

vector<vector<double>> getRandomRadii(double rmax, int k, double epsilon,
                                      int numRadiiVectors) {
  vector<double> set;

  // Berechnung der Anzahl der Radien, die benötigt werden, um eine ausreichende
  // Abdeckung sicherzustellen
  int limit = ceil(logBase((k / epsilon), (1 + epsilon)));

  // Erstelle das Set der Radien, deren Permutationen gebildet werden.
  for (int i = 0; i <= limit; i++) {
    set.push_back(pow((1 + epsilon), i) * (epsilon / k) * rmax);
  }

  vector<vector<double>> result(numRadiiVectors);

  // Initialisiert einen Mersenne Twister-Generator mit der Seed von 'rd'.
  mt19937 gen(1234);

  // Definiert eine Gleichverteilung für Ganzzahlen zwischen 0 und set.size()-1.
  uniform_int_distribution<> distrib(0, set.size() - 1);

  // Erzeugt numVectors viele Vektoren.
  for (int i = 0; i < numRadiiVectors; i++) {
    vector<double> currentVector(k);
    currentVector[0] = rmax;

    // Füllt den Vektor mit zufälligen Werten, die durch den Zufallsgenerator
    // bestimmt werden.
    for (size_t j = 1; j < currentVector.size(); j++) {
      currentVector[j] = set[distrib(gen)];
    }
    result[i] = currentVector;
  }

  return result;
}

// Generiert eine Liste von Vektoren, die jeweils zufällige Ganzzahlen zwischen
// 0 und k-1 enthalten.
vector<vector<int>> getU(int n, int k, double epsilon, int numUVectors) {
  // Berechnet die Länge jedes Vektors basierend auf den gegebenen Parametern k
  // und epsilon.
  int length =
      min(n, static_cast<int>((32 * k * (1 + epsilon)) / (pow(epsilon, 3))));

  vector<vector<int>> result(numUVectors);

  // Initialisiert einen Mersenne Twister-Generator mit der Seed von 'rd'.
  mt19937 gen(1234);

  // Definiert eine Gleichverteilung für Ganzzahlen zwischen 0 und k-1.
  uniform_int_distribution<> distrib(0, k - 1);

  // Erzeugt numVectors viele Vektoren.
  for (int i = 0; i < numUVectors; i++) {
    vector<int> currentVector(length);

    // Füllt den Vektor mit zufälligen Werten, die durch den Zufallsgenerator
    // bestimmt werden.
    for (int &value : currentVector) {
      value = distrib(gen);
    }
    result[i] = currentVector;
  }

  return result;
}

// Erstellt 'k' Bälle, die alle übergebenen Punkte beinhalten.
vector<Ball> selection(const vector<Point> &points, int k, const vector<int> &u,
                       const vector<double> &radii, double epsilon) {
  vector<Ball> balls(k, Ball(points.front().getCoordinates().size()));
  vector<vector<Point>> Si(k);
  double lambda = 1 + epsilon + 2 * sqrt(epsilon);

  for (size_t i = 0; i < u.size(); i++) {
    bool addedPoint = false;

    // Füge den ersten ersten Punkt in 'points', der nicht von 'X' oder 'R'
    // enthalten ist zu 'S_ui' hinzu.
    for (Point p : points) {
      if (!containsPoint(p, balls)) {
        Si[u[i]].push_back(p);
        addedPoint = true;
        break;
      }
    }

    // Wenn kein Punkt hinzugefügt wurde, breche den Vorgang ab und gib die
    // Bälle zurück.
    if (!addedPoint) {
      return balls;
    }

    // Wenn die Größe von 'S_ui' größer oder gleich 2 ist, finde den Ball,
    // der alle Punkte in 'S_ui' einschließt, und vergrößer seinen Radius um den
    // Faktor Lambda.
    if (Si[u[i]].size() >= 2) {
      Ball b = findMEB(Si[u[i]], epsilon);
      b.setRadius(b.getRadius() * lambda);
      balls[u[i]] = b;

    } else {
      balls[u[i]] = Ball(Si[u[i]][0], (epsilon / (1 + epsilon)) * radii[u[i]]);
    }
  }
  return balls;
}

// Hauptfunktion, die die Cluster berechnet.
vector<Cluster> clustering(const vector<Point> &points, int k, double epsilon,
                           int numUVectors, int numRadiiVectors) {
  vector<Cluster> bestCluster(k);
  double rmax = gonzalesrmax(points, k);

  // Berechnung der Radien und u-Werte basierend auf 'rmax', 'k' und 'epsilon'.
  vector<vector<double>> radii =
      getRandomRadii(rmax, k, epsilon, numRadiiVectors);
  vector<vector<int>> u = getU(points.size(), k, epsilon, numUVectors);

  // Initialisiere das 'bestCluster', indem alle Punkte Teil eines Clusters
  // sind.
  bestCluster[0].setPoints(points);
  double bestCost = cost(bestCluster);

#pragma omp parallel for collapse(2) schedule(dynamic) \
    shared(bestCluster, bestCost)
  // Berechne für alle Kombinationen von 'radii' und 'u' die Cluster mit den
  // geringsten Kosten.
  for (size_t i = 0; i < radii.size(); i++) {
    for (size_t j = 0; j < u.size(); j++) {
      vector<double> r = radii[i];
      vector<int> ui = u[j];

      // Berechne Bälle basierend auf den Radien und 'u'-Werten.
      vector<Ball> localBalls = selection(points, k, ui, r, epsilon);

      // Überprüfe, ob alle Punkte von den ausgewählten Bällen abgedeckt werden.
      if (containsAllPoints(points, localBalls)) {
        // Erstelle Cluster basierend auf den ausgewählten Bällen.
        vector<Cluster> localCluster(k);
        for (Point p : points) {
          for (int c = 0; c < k; c++) {
            if (localBalls[c].contains(p)) {
              localCluster[c].addPoint(p);
              break;
            }
          }
        }

        // Berechnung der Kosten für die lokalen Cluster.
        double localCost = cost(localCluster);

#pragma omp critical
        {
          // Aktualisierung des besten Clusters und Kostenwerts, falls ein
          // besserer gefunden wird.
          if (localCost < bestCost) {
            bestCost = localCost;
            bestCluster = localCluster;
          }
        }
      }
    }
  }

  return bestCluster;
}
