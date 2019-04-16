

#include <map>
#include <vector>
#include <tuple>
#include <string>
#include <functional>
#include <iostream>
#include <set>
#include <cmath>

using namespace std;

struct neuron_t
{
    vector<pair<string, double>> inputs;
    double value;
};

int main()
{
    map<string, function<double(double)>> nn_f;
    map<string, neuron_t> nn{
        {"A", {{}, 0.0}},
        {"B", {{}, 1.0}},
        {"1", {{}, 1.0}},
        {"X", {{}, 0.0}},
        {"Y", {{}, 0.0}},
        {"Result", {{}, 0.0}},
    };

    nn["X"].inputs = {{"A", 1}, {"B", 1}, {"1", -1}};
    nn["Y"].inputs = {{"A", -1}, {"B", 1}, {"1", -1}};
    nn["Result"].inputs = {{"X", 1}, {"Y", -2}, {"1", 1}};
    for (auto &[k, v] : nn)
    {
        nn_f[k] = [](double x) {
            return x;//exp(x)/(exp(x)+1);
        };
    }

    // liczymy
    map<string, neuron_t> new_nn = nn;
    auto count_nn = [&]() {
        for (auto &[k, v] : nn)
        {
            double s = 0;
            if (nn[k].inputs.size() != 0)
            {
                for (auto n : nn[k].inputs)
                {
                    s = s + nn[n.first].value * n.second;
                }
                new_nn[k].value = nn_f[k](s);
            }
        }
        nn = new_nn;
    };
    count_nn();
    for (auto &[k, v] : nn)
    {
        cout << k << ": " << nn[k].value << endl;
    }
    count_nn();
    
    for (auto &[k, v] : nn)
    {
        cout << k << ": " << nn[k].value << endl;
    }
    return 0;
}
