#include <cmath>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

using namespace std; // bad practice, but good for simple examples

class neuron_t {
public:
  vector<pair<double, shared_ptr<neuron_t>>> _input;
  function<double(double)> activation_f;
  virtual double action_potential() = 0;
  neuron_t() {
    activation_f = [](double x) {
      return 1.0 / (sqrt(1.0 + 10 * x * x));

      // return 1.0 / (1.0 + exp(-x));

      // if (x < 0) return -1.0;
      // return 1.0;
    };
  }
};

class const_neuron_t : public neuron_t {
public:
  double _value;
  const_neuron_t(double v) : _value(v){};
  virtual double action_potential() { return _value; };
};

class simple_neuron_t : public neuron_t {
public:
  virtual double action_potential() {
    double sum;
    for (auto &a : _input) {
      sum += a.first * a.second->action_potential();
    }
    return activation_f(sum);
  };
  simple_neuron_t() {}
  simple_neuron_t(
      vector<pair<double, shared_ptr<neuron_t>>> inputs) {
    _input = inputs;
  }
};

list<double> get_weights(vector<shared_ptr<neuron_t>> &nn) {
  list<double> ret;
  for (int i = 0; i < nn.size(); i++) {
    for (int j = 0; j < nn[i]->_input.size(); j++) {
      ret.push_back(nn[i]->_input[j].first);
    }
  }
  return ret;
}

void set_weights(vector<shared_ptr<neuron_t>> &nn,
                 list<double> weights) {
  for (int i = 0; i < nn.size(); i++) {
    for (int j = 0; j < nn[i]->_input.size(); j++) {
      nn[i]->_input[j].first = weights.front();
      weights.pop_front();
    }
  }
}

int main() {
  const vector<tuple<double, double, double>> truth_table = {
      {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};

  random_device rd{};
  mt19937 gen{rd()};
  normal_distribution<> distr{0.0, 1.0};

  shared_ptr<neuron_t> output = make_shared<simple_neuron_t>();
  auto input1 = make_shared<const_neuron_t>(0.0);
  auto input2 = make_shared<const_neuron_t>(0.0);
  auto bias = make_shared<const_neuron_t>(0.0);
  shared_ptr<neuron_t> hidden1 = make_shared<simple_neuron_t>();
  shared_ptr<neuron_t> hidden2 = make_shared<simple_neuron_t>();

  vector<shared_ptr<neuron_t>> all_neurons_to_learn = {
      hidden1, hidden2, output};

  hidden1->_input.push_back({0.5, input1});
  hidden1->_input.push_back({0.5, input2});
  hidden1->_input.push_back({0.5, bias});

  hidden2->_input.push_back({0.5, input1});
  hidden2->_input.push_back({0.5, input2});
  hidden2->_input.push_back({0.5, bias});

  output->_input.push_back({0.5, hidden1});
  output->_input.push_back({0.5, hidden1});

  auto load_learned_fun = [&]() {
    list<double> weights_0 = {262.547, -262.551, 2.27988,  -0.458649,
                                   2.51047, 15.2769,  -21.5358, -6.25552};
    set_weights(all_neurons_to_learn, weights_0);
  };

  auto loss_function = [&]() {
    double diffNew = 0.0;
    for (auto &[x, y, v] : truth_table) {
      input1->_value = x;
      input2->_value = y;
      diffNew += sqrt((v - output->action_potential()) *
                           (v - output->action_potential()));
    }
    return diffNew;
  };

  auto initialization_fun = [&]() {
    pair<double, list<double>> best_weights = {
        1000.0, get_weights(all_neurons_to_learn)};
    for (int i = 0; i < 100; i++) {
      auto new_weights = get_weights(all_neurons_to_learn);
      for (auto &e : new_weights) {
        e = distr(gen);
      }
      set_weights(all_neurons_to_learn, new_weights);
      auto nl = loss_function();
      if (nl < best_weights.first) {
        best_weights = {nl, new_weights};
      }
    }
  };

  auto learning_fun = [&](int max_iter) {
    for (int i = 0; i < 100000; i++) {
      double diffPrev;
      double diffNew;
      auto prev_weights = get_weights(all_neurons_to_learn);
      auto new_weights = get_weights(all_neurons_to_learn);

      set_weights(all_neurons_to_learn, prev_weights);
      diffPrev = loss_function();
      for (auto &e : new_weights) {
        e = e + 0.1 * distr(gen);
      }
      set_weights(all_neurons_to_learn, new_weights);
      diffNew = loss_function();
      set_weights(all_neurons_to_learn,
                  (diffPrev > diffNew) ? new_weights : prev_weights);
    }
  };

  auto print_results = [&]() {
    for (auto &[x, y, v] : truth_table) {
      input1->_value = x;
      input2->_value = y;
      cout << "input: " << input1->action_potential() << " ";
      cout << "input: " << input2->action_potential() << " ";
      cout << "Output: " << output->action_potential() << " vs expected "
                << v << endl;
    }
    auto print_neuron = [](auto name, auto neuron) {
      cout << name;
      for (int j = 0; j < neuron->_input.size(); j++) {
        cout << " [" << j << "]=" << neuron->_input[j].first;
      }
      cout << endl;
    };
    print_neuron("hidden1", hidden1);
    print_neuron("hidden2", hidden2);
    print_neuron("output", output);
    auto weights = get_weights(all_neurons_to_learn);
    cout << "weights = {";
    for (auto e : weights)
      cout << e << ",";
    cout << "}" << endl;
  };

  load_learned_fun();
  //initialization_fun();
  //learning_fun(1000);
  print_results();

  return 0;
}