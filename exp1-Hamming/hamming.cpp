#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <math.h>

using namespace std;

double p[2005];
ofstream OutFile("./results/power4_8736_results.txt");

double cal_phi_n(double n) {
    clock_t start, end, start_io, end_io;
    start = clock();
    double io_time = 0, phi = 0;
    for (int _x = 1001; _x <= n; _x++) {
        double x = _x * 0.001;
        double pre = ((x - 1) * 1.0 / x) * p[int((x - 1.0) * 1000)] + 1.0 / (x * x);
        start_io = clock();
        OutFile << fixed << setprecision(3) << x << "\t";
        OutFile << fixed << setprecision(13) << pre << endl;
        printf("%5.3f %16.12f\n", x, pre);
        end_io = clock();
        io_time += (double)(end_io - start_io) * 1000 / CLOCKS_PER_SEC;
    }
    end = clock();
    double all_time = (double)(end - start) * 1000 / CLOCKS_PER_SEC;
    return all_time - io_time;
}

double power4_func(int k_n, int num_x, bool use_acc) {
    clock_t start, end, start_io, end_io;
    start = clock();
    double io_time = 0;
    for (int _x = 0; _x <= num_x; _x++) {
        double x = _x * 0.001, sum = 0.0;
        for (int k = 1; k <= k_n; k++)
            sum += ((2 - x) * (1 - x)) * 1.0 / (k * (k + x) * (k + 2) * (k + 1));
        sum += (1 - x) * 1.0 / 4.0 + 1.0;
        if (use_acc)
            p[int(x * 1000)] = sum;
        start_io = clock();
        OutFile << fixed << setprecision(3) << x << "\t";
        OutFile << fixed << setprecision(13) << sum << endl;
        printf("%5.3f %16.12f\n", x, sum);
        end_io = clock();
        io_time += (double)(end_io - start_io) * 1000 / CLOCKS_PER_SEC;
    }
    end = clock();
    double all_time = (double)(end - start) * 1000 / CLOCKS_PER_SEC;
    return all_time - io_time;
}

double basic_func(int k_n, int num_x) {
    clock_t start, end, start_io, end_io;
    start = clock();
    double io_time = 0;
    for (int _x = 0; _x <= num_x; _x++) {
        double x = _x * 0.001, sum = 0.0;
        for (int k = 1; k <= k_n; k++)
            sum += 1.0 / (k * (k + x));
        start_io = clock();
        OutFile << fixed << setprecision(3) << x << "\t";
        OutFile << fixed << setprecision(12) << sum << endl;
        printf("%5.3f %16.12f\n", x, sum);
        end_io = clock();
        io_time += (double)(end_io - start_io) * 1000 / CLOCKS_PER_SEC;
    }
    end = clock();
    double all_time = (double)(end - start) * 1000 / CLOCKS_PER_SEC;
    return all_time - io_time;
}

// Generate the results with original Hamming function. Algorithm Basic
void gen_basic_results(int k_n, int num_x) {
    double basic_time = basic_func(k_n, num_x - 1);
    printf("Basic Run time(ms): %.0f\n", basic_time);
}

// Generate the results in the range of error 0.5e-12. Algorithm Power4
void gen_results_in_error(int k_n, int num_x) {
    double power4_time = power4_func(k_n, num_x - 1, false);
    printf("Power4 Run time(ms): %.0f\n", power4_time);
}

// Accelerate calculating results but out of the error range 0.5e-12.
void acc_gen_results_out_error(int k_n, int num_x) {
    int limit_num = 1000;
    double power4_time = power4_func(k_n, limit_num, true);
    double dp_time = cal_phi_n(num_x - 1);
    printf("Power4 Run time(ms): %.0f\n", power4_time);
    printf("DP Run time(ms): %.0f\n", dp_time);
    printf("Acc Run time(ms): %.0f\n", power4_time + dp_time);
}

int main() {
    int power4_n = 8736, num_x = 2001;
    // If you want to compare, remember reserve 13 decimal places for output.
    gen_results_in_error(8736, num_x);
    OutFile.close();
    return 0;
}