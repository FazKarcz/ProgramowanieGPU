#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

using namespace std;
std::mutex mtx;

int counter;

void zwiekszanie(int id) {
    for (int i = 0; i < 10000; i++) {
        mtx.lock();
        ++counter;
        cout << "Watek " << id << " zwiekszyl licznik: " << counter << std::endl;
        mtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void zmniejszanie(int id) {
    for (int i = 0; i < 10000; i++) {
        mtx.lock();
        --counter;
        cout << "Watek " << id << " zmniejszyl licznik: " << counter << std::endl;
        mtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main()
{
    std::cout << "Uruchamianie watkow..." << std::endl;

    std::thread thread1(zwiekszanie, 1);
    std::thread thread2(zmniejszanie, 2);

    thread1.join();
    thread2.join();

    std::cout << "Wszystkie watki zakonczyly prace. Koncowa wartosc licznika: " << counter << std::endl;

    return 0;
}

