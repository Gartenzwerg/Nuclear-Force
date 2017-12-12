#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "lib.h"
#include <armadillo>
#include <fstream>
#include <random>

using namespace std;
using namespace arma;

const double m = 939;
const double PI = 3.14159265;
const double constant = 197.3; //h_bar*c in MeV*fm
const double mu = 0.7;
const double Va = -10.463, Vb = -1650.6, Vc = 6484.3;
const double a = 1, b = 4, c = 7;
const double L = 0.696*1.4;
int N = 50;
int maxEcount = 100; //MaxEnergy = maxEcount*0.05 (MeV)

//set up the mesh points and weights
void mesh (double x[], double w[], int N);

//potential matrix in momentum space
void Yukawa_potential(double **V, double k[], int N);
void Effective_potential(double **V, double k[], int N, double C[], int C_num);
void Pion_Effective_potential(double **V, double k[], int N, double C[], int C_num);
void Amatrix(double **A, double **V, double k[], double w[], int N);
double errorFunction(double ref_phaseshifts[], double res[], int count);
void reset_matrices(double **M, int N);
void Compute_phseshfits(double **V, double **A, double k[], double w[], int N, double C[], int C_num, double shifts[]);

int main() {

	double E = 0.05 ;// in MeV
	double k[N+1], w[N]; //in fm^{-1}
	double **V, **A;
	double *C = new double [4], *prev_C = new double [4], *curr_C = new double [4], *best_C = new double [4], *attempt_C = new double [4];

	// maxEcount = 800;

	double *ref_phaseshifts = new double [maxEcount], *prev_phaseshifts = new double [maxEcount], *curr_phaseshifts = new double [maxEcount],
			*attempt_phaseshifts = new double [maxEcount];
	int C_num = 2; // number of parameters under consideration
	double prev_error, curr_error, min_error;
	double error_derivative[4];
	// ofstream fout;

	C[0] = attempt_C[0] = prev_C[0] = best_C[0] = -56.0252;	//C0
	C[1] = attempt_C[1] = prev_C[1] = best_C[1] = 18.2039;	//C2
	C[2] = attempt_C[2] = prev_C[2] = best_C[2] = 0;	//C4
	C[3] = attempt_C[3] = prev_C[3] = best_C[3] = 0;	//C4'

	// fout.open("sc_50.dat");

	mesh(k, w, N);

	V = new double *[N+1];
	A = new double *[N+1];
	for(int i = 0; i < (N+1); ++i) {
		V[i] = new double [N+1];
		A[i] = new double [N+1];
	}


	Compute_phseshfits(V, A, k, w, N, C, 0, ref_phaseshifts);
	Compute_phseshfits(V, A, k, w, N, prev_C, C_num, prev_phaseshifts);

	//fitting
	// Greatest Descent + Random perturbations + "active intervention" when out of control
	prev_error = errorFunction(ref_phaseshifts, prev_phaseshifts, maxEcount);
	min_error = prev_error;
	double C_increments[4] = {};
	for(int i=0; i < C_num; ++i){
		curr_C[i] = prev_C[i] - 0.05;
		C_increments[i] = - 0.05;
	}
	
	int unchanged_min_count = 0;

	default_random_engine generator;
  	normal_distribution<double> distribution(0,1);
  	double random_number;

	while(true) {
		double grad_square = 0, attempt_error;

		Compute_phseshfits(V, A, k, w, N, curr_C, C_num, curr_phaseshifts);
		curr_error = errorFunction(ref_phaseshifts, curr_phaseshifts, maxEcount);
		cout<<curr_error<<" "<<min_error<<" "<<prev_error<<endl;

		if(curr_error < min_error) {
			if((min_error-curr_error) > 0.001) unchanged_min_count = 0;
			min_error = curr_error;
			for(int i = 0; i < C_num; ++i) {
				best_C[i] = curr_C[i];
			}
		} else {
			unchanged_min_count++;
		}
		if(unchanged_min_count > 2000 || min_error < 1) break;	//convergence criteria

		if(curr_error < 6*min_error) {

			for(int i=0; i < C_num; ++i) {
				attempt_C[i] = curr_C[i] + C_increments[i];
			}
			Compute_phseshfits(V, A, k, w, N, attempt_C, C_num, attempt_phaseshifts);
			attempt_error = errorFunction(ref_phaseshifts, attempt_phaseshifts, maxEcount);
			for(int i=0; i < C_num; ++i) {
				error_derivative[i] = (attempt_error-prev_error)/(attempt_C[i]-prev_C[i]);
				grad_square += error_derivative[i]*error_derivative[i];
			}

		} else {
			for(int i = 0; i < C_num; ++i) {
				error_derivative[i] = 0;
				C_increments[i] = 0.005*best_C[i];
				curr_C[i] = best_C[i];
				curr_error = min_error;
				grad_square = 1;
			}
		}	

		for(int i=0; i < C_num; ++i) {
			prev_C[i] = curr_C[i];
			random_number = distribution(generator);
			if(curr_error > 2*min_error) {
				C_increments[i] = 0.05*((min_error-curr_error)/grad_square)*error_derivative[i] + 
									+abs(random_number)*(0.1*random_number+1)*C_increments[i];
				curr_C[i] += C_increments[i];
				// cout<<i<<" "<<curr_C[i]<<" "<<random_number<<" "<<C_increments[i]<<endl;
			} else {
				// cout<<curr_error<<" "<<grad_square<<" "<<error_derivative[i]<<" "<<(-curr_error/grad_square)<<endl;
				C_increments[i] = 0.005*(-curr_error/grad_square)*error_derivative[i] + 
									+abs(random_number)*(0.1*random_number+1)*C_increments[i];
				curr_C[i] += C_increments[i];
				// cout<<i<<" "<<curr_C[i]<<" "<<random_number<<" "<<C_increments[i]<<endl;
			}
		}
		prev_error = curr_error;
		// loopCounter++;
		// cout<<prev_error<<endl;
	}

	Compute_phseshfits(V, A, k, w, N, best_C, C_num, curr_phaseshifts);
	cout<<"C0 = "<<best_C[0]<<", C2 = "<<best_C[1]<<", C4 = "<<best_C[2]<<", C4' = "<<best_C[3]<<", error function = "<<min_error<<endl;
	for(int i=0; i<maxEcount; ++i) {
		cout<<(i+1)*0.05<<"		"<<curr_phaseshifts[i]<<"		"<<ref_phaseshifts[i]<<endl;
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

void Yukawa_potential(double **V, double k[], int N) {
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

void Effective_potential(double **V, double k[], int N, double C[], int C_num) {
	double exponentials[N+1];
	double square, powerFour;
	
	for(int i = 0; i < (N+1); ++i) {
		square = (k[i]/L)*(k[i]/L);
		powerFour = square*square;
		exponentials[i] = exp(-powerFour);
	}

	for(int i = 0; i < (N+1); ++i) {
		for(int j = i; j < (N+1); ++j) {
			V[i][j] = C[0];
			if(C_num > 1) {
				V[i][j] += C[1]*(k[i]*k[i]+k[j]*k[j]);
				if(C_num > 2) {
					V[i][j] += C[2]*(k[i]*k[i]*k[i]*k[i] + k[j]*k[j]*k[j]*k[j]);
					V[i][j] += C[3]*k[i]*k[i]*k[j]*k[j];
				}
			}
			V[i][j] *= exponentials[i]*exponentials[j];
			V[j][i] = V[i][j];
		}
	}
}

void Pion_Effective_potential(double **V, double k[], int N, double C[], int C_num) {
	double exponentials[N+1];
	double square, powerFour;
	double temp = 0, ksum2 = 0, kdiff2 = 0, amu2;
	amu2 = a*a*mu*mu;
	
	for(int i = 0; i < (N+1); ++i) {
		square = (k[i]/L)*(k[i]/L);
		powerFour = square*square;
		exponentials[i] = exp(-powerFour);
	}

	for(int i = 0; i < (N+1); ++i) {
		for(int j = i; j < (N+1); ++j) {
			V[i][j] = C[0];
			if(C_num > 1) {
				V[i][j] += C[1]*(k[i]*k[i]+k[j]*k[j]);
				if(C_num > 2) {
					V[i][j] += C[2]*(k[i]*k[i]*k[i]*k[i] + k[j]*k[j]*k[j]*k[j]);
					V[i][j] += C[3]*k[i]*k[i]*k[j]*k[j];
				}
			}
			temp = 1/(4.0*mu*k[i]*k[j]);
			ksum2 = (k[i] + k[j])*(k[i] + k[j]);
			kdiff2 = (k[i] - k[j])*(k[i] - k[j]);

			V[i][j] += temp*Va*log((ksum2 + amu2)/(kdiff2 + amu2));
			V[i][j] *= exponentials[i]*exponentials[j];
			V[j][i] = V[i][j];
		}
	}	
}

double errorFunction(double ref_phaseshifts[], double res[], int count) {
	double difference = 0, difference_sum = 0;
	double energy = 0;

	for(int i = 0; i < count; ++i) {
		energy += 0.05;
		difference = (ref_phaseshifts[i] - res[i]);
		difference_sum += difference*difference/(1+exp((energy-7.54)/1.8));
	}
	return difference_sum;
}	

void reset_matrices(double **M, int N) {
	for(int i=0; i <= N; ++i) {
		for(int j=i; j <= N; ++j) {
			M[i][j] = 0;
			M[j][i] = 0;
		}
	}
}

void Compute_phseshfits(double **V, double **A, double k[], double w[], int N, double C[], int C_num, double shifts[]) {
	double E = 0.05;
	reset_matrices(V, N);
	reset_matrices(A, N);
	for(int loopCounter = 0; loopCounter < maxEcount; loopCounter++) {
		k[N] = sqrt(m*E)/constant;
		if(C_num == 0) {
			Yukawa_potential(V, k, N);
		} else if(C_num > 0) {
			// !!! MANUAL SWITCH !!! //
			// Effective_potential(V, k, N, C, C_num);
			Pion_Effective_potential(V, k, N, C, C_num);
		}
		Amatrix(A, V, k, w, N);	
		mat matR(N+1, N+1), matA(N+1, N+1), matV(N+1, N+1);
		for(int i = 0; i < (N+1); i++) {
			for(int j = 0; j < (N+1); j++) {
				matA(i, j) = A[i][j];
				matV(i, j) = V[i][j];
			}
		}

		matR = inv(matA) * matV;

		// fout<<E<<"	"<<(atan(-matR(N, N)*m*k[N]/(constant*constant)))/PI*180<<endl;
		shifts[loopCounter] = (atan(-matR(N, N)*m*k[N]/(constant*constant)))/PI*180;
		// cout <<E<<"		"<<shifts[loopCounter]<<endl;
		E += 0.05;
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