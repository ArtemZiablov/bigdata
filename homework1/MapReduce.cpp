#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>

using namespace std;

mutex mtx;

// метод для знаходження мінімума та максимума у підмасиві
pair<int, int> findMinMaxPart(const vector<int>& data, int start, int end) {
    auto result = minmax_element(data.begin() + start, data.begin() + end);
    return {*result.first, *result.second};
}

// MapReduce метод
pair<int, int> findMinMaxMapReduce(const vector<int>& data, int numThreads) {
    vector<thread> threads;
    vector<pair<int, int>> results(numThreads);

    int dataSize = data.size();
    int chunkSize = dataSize / numThreads;

    // map - розбиваємо на підзадачі
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = min((i + 1) * chunkSize, dataSize);

        // запускаємо кожен потік
        threads.emplace_back([&results, &data, i, start, end]() {
            results[i] = findMinMaxPart(data, start, end);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // reduce - об'єднуємо результати з усіх потоків
    // ініціалізуємо min та max
    int minNum = results[0].first;
    int maxNum = results[0].second;

    // знаходимо мінімальний з мінімальних, та максимальний з максимальних
    for (int i = 1; i < numThreads; ++i) {
        lock_guard<mutex> lock(mtx);
        minNum = min(minNum, results[i].first);
        maxNum = max(maxNum, results[i].second);
    }

    return {minNum, maxNum};
}

// "стандартний" метод
pair<int, int> findMinMaxStandard(const vector<int>& data) {
    auto result = minmax_element(data.begin(), data.end());
    return {*result.first, *result.second};
}

int main() {
    // створюємо дууже великий набір чисел (один мільярд чисел)
    const int size = 1000000000;
    vector<int> data(size);

    for (int i = 0; i < size; ++i) {
        data[i] = rand() % 100000 - 50000;
    }

    int numThreads = thread::hardware_concurrency();

    // викликаємо MapReduce метод
    auto start = chrono::high_resolution_clock::now();
    auto [minValue, maxValue] = findMinMaxMapReduce(data, numThreads);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    cout << "MapReduce: min = " << minValue << ", max = " << maxValue << endl;
    cout << "Time: " << duration.count() << " sec." << endl;

    // викликаємо "стандартный" метод
    auto start2 = chrono::high_resolution_clock::now();
    auto [minValue2, maxValue2] = findMinMaxStandard(data);
    auto end2 = chrono::high_resolution_clock::now();

    chrono::duration<double> duration2 = end2 - start2;
    cout << "Standart: min = " << minValue2 << ", max = " << maxValue2 << endl;
    cout << "Time: " << duration2.count() << " sec." << endl;

    return 0;
}
