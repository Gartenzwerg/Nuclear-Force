#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "lib.h"
#include <armadillo>
#include <fstream>

using namespace std;
using namespace arma;

const double m = 939;
const double PI = 3.14159265;
const double constant = 197.3; //h_bar*c in MeV*fm
const double mu = 0.7;
const double Va = -10.463, Vb = -1650.6, Vc = 6484.3;
const double a = 1, b = 4, c = 7;
int N = 50;

//set up the mesh points and weights
void mesh (double x[], double w[], int N);

//potential matrix in momentum space
void potential(double **V, double k[], int N);
void Amatrix(double **A, double **V, double k[], double w[], int N);

int main() {

	double E = 0.5 ;// in MeV
	double k[N+1], w[N]; //in fm^{-1}
	double **V, **A;
	ofstream fout;

	fout.open("sc_50.dat",ios::app);
	mesh(k, w, N);
	V = new double *[N+1];
	A = new double *[N+1];
	for(int i = 0; i < (N+1); ++i) {
		V[i] = new double [N+1];
		A[i] = new double [N+1];
	}
	for(int loopCounter = 0; loopCounter < 300; loopCounter++) {
		k[N] = sqrt(m*E)/constant;
		potential(V, k, N);
		Amatrix(A, V, k, w, N);	
		mat matR(N+1, N+1), matA(N+1, N+1), matV(N+1, N+1);
		for(int i = 0; i < (N+1); i++) {
			for(int j = 0; j < (N+1); j++) {
				matA(i, j) = A[i][j];
				matV(i, j) = V[i][j];
			}
		}

		matR = inv(matA) * matV;

		fout<<E<<"	"<<atan(-matR(N, N)*m*k[N]/(constant*constant))<<endl;
		E += 0.5;
	}
	
}

void mesh (double x[], double w[], int N) {
	gauleg(-1, 1, x, w, N);
	double temp = 0;
	for(int i = 0; i < N; ++i) {
		temp = cos( PI*(1.+x[i])/4. );
		x[i] = tan( PI*(1.+x[i])/4. );
		w[i] = w[i]*(PI/4.)/(temp*temp);
	}
}

void potential(double **V, double k[], int N) {
	double temp = 0, ksum2 = 0, kdiff2 = 0;
	double amu2, bmu2, cmu2;
	amu2 = a*a*mu*mu;
	bmu2 = b*b*mu*mu;
	cmu2 = c*c*mu*mu;
	for(int i = 0; i < (N+1); ++i) {
		for(int j = i; j < (N+1); ++j) {
			temp = 1/(4.0*mu*k[i]*k[j]);
			ksum2 = (k[i] + k[j])*(k[i] + k[j]);
			kdiff2 = (k[i] - k[j])*(k[i] - k[j]);
			V[i][j] = Va*log((ksum2 + amu2)/(kdiff2 + amu2));
			V[i][j] = V[i][j] + Vb*log((ksum2 + bmu2)/(kdiff2 + bmu2));
			V[i][j] = V[i][j] + Vc*log((ksum2 + cmu2)/(kdiff2 + cmu2));
			V[i][j] = temp*V[i][j];
			V[j][i] = V[i][j];
		}
	}
}

void Amatrix(double **A, double **V, double k[], double w[], int N) {
	double u[N+1];
	u[N] = 0;
	double k2 = k[N]*k[N];
	for(int i = 0; i < N; ++i) {
		u[i] = (2.0/PI)*(w[i]*k[i]*k[i])/((k2-k[i]*k[i])*constant*constant/m);
		u[N] -= (2.0/PI)*(w[i]*k2)/((k2-k[i]*k[i])*constant*constant/m);
	}
	for(int i = 0; i < (N+1); ++i) {
		for(int j = 0; j < (N+1); ++j) {
			A[i][j] = -V[i][j]*u[j];
		}
		A[i][i] += 1.;
	}
}