#include <iostream>

#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2

#include <windows.h>
#include <time.h>

using namespace std;

//生成测试样例 100,500,1000,2000,3000
const int n = 3000;
float m[n][n];
void reset() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			m[i][j] = 0;
		}
		m[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			m[i][j] = rand();
	}
	for (int k = 0; k < n; k++) {
		for (int i = k + 1; i < n; i++) {
			for (int j = 0; j < n; j++) {
				m[i][j] += m[k][j];
			}
		}
	}
}

//串行
void Ord(){
	for (int k = 0; k < n; k++){
		for (int j = k + 1; j < n; j++){
			m[k][j] = m[k][j] / m[k][k];
		}
		m[k][k] = 1.0;

		for (int i = k + 1; i < n; i++){
			for (int j = k + 1; j < n; j++){
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}
			m[i][k] = 0;
		}
	}
}

//SSE未对齐
void SSE(){
	__m128 va, vt, vx, vaij, vaik, vakj;
	for (int k = 0; k < n; k++){
		vt = _mm_set_ps(m[k][k], m[k][k], m[k][k], m[k][k]);
		int j;
		for (j = k + 1; j + 4 <= n; j += 4){
			va = _mm_loadu_ps(&(m[k][j]));
			va = _mm_div_ps(va, vt);
			_mm_store_ps(&(m[k][j]), va);
		}

		for (; j < n; j++){
			m[k][j] = m[k][j] / m[k][k];

		}
		m[k][k] = 1.0;

		for (int i = k + 1; i < n; i++){
			vaik = _mm_set_ps(m[i][k], m[i][k], m[i][k], m[i][k]);

			for (j = k + 1; j + 4 <= n; j += 4){
				vakj = _mm_loadu_ps(&(m[k][j]));
				vaij = _mm_loadu_ps(&(m[i][j]));
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);

				_mm_store_ps(&m[i][j], vaij);
			}

			for (; j < n; j++){
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}

			m[i][k] = 0;
		}
	}
}

//AVX未对齐
void AVX(){
	__m256 va, vt, vx, vaij, vaik, vakj;
	for (int k = 0; k < n; k++){
		vt = _mm256_set_ps(m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k]);
		int j;
		for (j = k + 1; j + 8 <= n; j += 8){
			va = _mm256_loadu_ps(&(m[k][j]));
			va = _mm256_div_ps(va, vt);
			_mm256_store_ps(&(m[k][j]), va);
		}

		for (; j < n; j++){
			m[k][j] = m[k][j] / m[k][k];

		}
		m[k][k] = 1.0;

		for (int i = k + 1; i < n; i++){
			vaik = _mm256_set_ps(m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]);

			for (j = k + 1; j + 8 <= n; j += 8){
				vakj = _mm256_loadu_ps(&(m[k][j]));
				vaij = _mm256_loadu_ps(&(m[i][j]));
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);

				_mm256_store_ps(&m[i][j], vaij);
			}

			for (; j < n; j++){
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}

			m[i][k] = 0;
		}
	}
}

//SSE对齐
void SSE_update() {
	for (int k = 0; k < n; k++) {
		//1.处理不对齐部分
		int pre1 = 4 - (k + 1) % 4;
		for (int j = k + 1; j < k + 1 + pre1; j++) {
			m[k][j] /= m[k][k];
		}
		//2.处理对齐部分
		float tmp1[4] = { m[k][k] ,m[k][k] ,m[k][k] ,m[k][k] };
		__m128 tmp_kk = _mm_load_ps(tmp1);
		int num1 = pre1 + k + 1;
		for (int j = pre1 + k + 1; j + 4 <= n; j += 4, num1 = j) {
			__m128 tmp_kj = _mm_load_ps(m[k] + j);
			tmp_kj = _mm_div_ps(tmp_kj, tmp_kk);
			_mm_store_ps(m[k] + j, tmp_kj);
		}
		//3.处理剩余部分
		for (int j = num1; j < n; j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			//1.处理不对齐部分
			int pre2 = 4 - (k + 1) % 4;
			for (int j = k + 1; j < k + 1 + pre2; j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			//2.处理对齐部分
			float tmp2[4] = { m[i][k] , m[i][k] , m[i][k] , m[i][k] };
			__m128 tmp_ik = _mm_load_ps(tmp2);
			int num2 = pre2 + k + 1;
			for (int j = pre2 + k + 1; j + 4 <= n; j += 4, num2 = j) {
				__m128 tmp_ij = _mm_load_ps(m[i] + j);
				__m128 tmp_kj = _mm_load_ps(m[k] + j);
				tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
				_mm_store_ps(m[i] + j, tmp_ij);
			}
			//3.处理剩余部分
			for (int j = num2; j < n; j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}

//AVX对齐
void AVX_update() {
	for (int k = 0; k < n; k++) {
		//1.处理不对齐部分
		int pre1 = 8 - (k + 1) % 8;
		for (int j = k + 1; j < k + 1 + pre1; j++) {
			m[k][j] /= m[k][k];
		}
		//2.处理对齐部分
		float tmp1[8] = { m[k][k] ,m[k][k] ,m[k][k] ,m[k][k],m[k][k],m[k][k],m[k][k],m[k][k] };
		__m256 tmp_kk = _mm256_load_ps(tmp1);
		int num1 = pre1 + k + 1;
		for (int j = k + 1 + pre1; j + 8 <= n; j += 8, num1 = j) {
			__m256 tmp_kj = _mm256_load_ps(m[k] + j);
			tmp_kj = _mm256_div_ps(tmp_kj, tmp_kk);
			_mm256_store_ps(m[k] + j, tmp_kj);
		}
		//3.处理剩余部分
		for (int j = num1; j < n; j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			//1.处理不对齐部分
			int pre2 = 8 - (k + 1) % 8;
			for (int j = k + 1; j < k + 1 + pre2; j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			//2.处理对齐部分
			float tmp2[8] = { m[i][k] , m[i][k] , m[i][k] , m[i][k], m[i][k] , m[i][k] , m[i][k] , m[i][k] };
			__m256 tmp_ik = _mm256_load_ps(tmp2);
			int num2 = pre2 + k + 1;
			for (int j = pre2 + k + 1; j + 8 <= n; j += 8, num2 = j) {
				__m256 tmp_ij = _mm256_load_ps(m[i] + j);
				__m256 tmp_kj = _mm256_load_ps(m[k] + j);
				tmp_kj = _mm256_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm256_sub_ps(tmp_ij, tmp_kj);
				_mm256_store_ps(m[i] + j, tmp_ij);
			}
			//3.处理剩余部分
			for (int j = num2; j < n; j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}


int main()
{
	double seconds1 = 0, seconds2 = 0, seconds3 = 0, seconds4 = 0, seconds5 = 0;
	long long head, tail, freq, noww;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	int times = 10;
	for (int i = 0; i < times; i++) {
		reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		Ord();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds1 += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << seconds1 / times << 'ms' << endl;

	for (int i = 0; i < times; i++) {
		reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		SSE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds2 += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << seconds2 / times << 'ms' << endl;

	for (int i = 0; i < times; i++) {
		reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		AVX();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds3 += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << seconds3 / times << 'ms' << endl;

	for (int i = 0; i < times; i++) {
		reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		SSE_update();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds4 += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << seconds4 / times << 'ms' << endl;

	for (int i = 0; i < times; i++) {
		reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		AVX_update();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds5 += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << seconds5 / times << 'ms' << endl;

	return 0;
}


