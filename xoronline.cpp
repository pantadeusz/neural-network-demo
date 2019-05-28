#include <cmath>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

using namespace std;

class neuron_t {
public:
// pary: waga,neuron
  vector<pair<double, shared_ptr<neuron_t>>> inputs;
// funkcja aktywacji tego konkretnego neuronu
  function<double(double)> activation_function;
  // oblicz poziom wzbudzenia neuronu
  virtual double calculate() {
    double sum_inputs = 0;
    for (auto &input : inputs) {
      sum_inputs += input.first * input.second->calculate();
    }
    return activation_function(sum_inputs);
  };
  // inicjalizacja na wartości domyślne
  neuron_t() {
    activation_function = [](double x) {
      return 1.0 / (sqrt(1.0 + 10 * x * x));
    };
  }
};

// neuron symulujący wejście
class input_neuron_t : public neuron_t {
public:
  double value; // wartość wejścia
  virtual double calculate() { return value; }
  input_neuron_t(double value) { this->value = value; }
};

int main() {
    // konfiguracja generatora liczb losowych
  random_device rd{};
  mt19937 gen{rd()};
  normal_distribution<> distr{0.0, 1.0};

  // konfiguracja neuronów

  auto output_neuron = make_shared<neuron_t>();
  auto hidden0 = make_shared<neuron_t>();
  auto hidden1 = make_shared<neuron_t>();
  auto input0 = make_shared<input_neuron_t>(1.0);
  auto input1 = make_shared<input_neuron_t>(1.0);
  auto bias = make_shared<input_neuron_t>(1.0);

 // połączenie ich w sieć
  output_neuron->inputs.push_back({0.5, bias});
  output_neuron->inputs.push_back({0.2, hidden0});
  output_neuron->inputs.push_back({0.3, hidden1});

  hidden0->inputs.push_back({-0.5, bias});
  hidden0->inputs.push_back({0.4, input0});
  hidden0->inputs.push_back({0.5, input1});

  hidden1->inputs.push_back({0.5, bias});
  hidden1->inputs.push_back({33.6, input0});
  hidden1->inputs.push_back({0.7, input1});

  // pobranie wag z poszczególnych neuronów z sieci neuronowej
  auto get_nn_weights = [&]() {
    list<double> weights;
    auto pbck = [&](auto neuron) {
      for (auto [w, neuron] : neuron->inputs)
        weights.push_back(w);
    };
    pbck(output_neuron);
    pbck(hidden0);
    pbck(hidden1);
    return weights;
  };
  // zapisanie wag do sieci neuronowej
  auto set_nn_weights = [&](list<double> weights) {
    auto updtneuron = [&](auto &neuron) {
      for (auto &[w, neuron] : neuron->inputs) {
        w = weights.front();
        weights.pop_front();
      }
    };
    updtneuron(output_neuron);
    updtneuron(hidden0);
    updtneuron(hidden1);
  };

  // przykład do którego ma się dostosować
  vector<tuple<double, double, double>> truth_table_xor = {
      {0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}};

  // funkcja kosztu - oblicza roznice miedzy wynikiem a oczekiwaną wartością
  auto loss_function = [&](auto weights) {
    double loss = 0;
    set_nn_weights(weights);
    for (auto &[a, b, r] : truth_table_xor) {
      input0->value = a;
      input1->value = b;
      auto v = output_neuron->calculate();
      loss += (r - v) * (r - v);
    }
    return sqrt(loss);
  };

  // inicjalizacja wektora wag w neuronach na losowe
  // powtarzamy wielokrotnie w celu uzyskania dobrego punktu startu
  auto best_weights_taken = get_nn_weights();
  for (int i = 0; i < 1000; i++) {
    list<double> weights_taken = best_weights_taken;
    for (auto &w : weights_taken) {
      w = 1.0 * distr(gen);
    }
    if (loss_function(weights_taken) < loss_function(best_weights_taken)) {
      cout << loss_function(weights_taken) << " < "
           << loss_function(best_weights_taken) << endl;
      best_weights_taken = weights_taken;
    }
  }
  set_nn_weights(best_weights_taken);

  // algorytm wspinaczkowy
  for (int i = 0; i < 10000; i++) {
    auto weights_taken = best_weights_taken;
    for (auto &w : weights_taken) {
      w += 0.1 * distr(gen);
    }
    if (loss_function(weights_taken) < loss_function(best_weights_taken)) {
      cout << loss_function(weights_taken) << " < "
           << loss_function(best_weights_taken) << endl;
      best_weights_taken = weights_taken;
    }
  }
  set_nn_weights(best_weights_taken);

  // wypisanie wag
  best_weights_taken = get_nn_weights();
  for (double w : best_weights_taken) {
    cout << w << " ";
  }
  cout << endl;

  // wypisanie jak sobie radzi nasza sieć
  for (auto &[a, b, r] : truth_table_xor) {
    input0->value = a;
    input1->value = b;
    cout << input0->calculate() << " " << input1->calculate() << " --> "
         << output_neuron->calculate() << "     expected " << r
         << " diff: " << (output_neuron->calculate() - r) << endl;
  }

  return 0;
}
