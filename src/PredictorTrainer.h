/*
 * PredictorTrainer.h
 *
 *  Created on: Jul 31, 2015
 *      Author: lunt
 */

#ifndef SRC_PREDICTORTRAINER_H_
#define SRC_PREDICTORTRAINER_H_

#include <string>

using namespace std;

enum ObjType
{
    SSE,                                            // sum of squared error
    CORR,                                           // Pearson correlation
    CROSS_CORR,                                     // cross correlation (maximum in a range of shifts)
    PGP,                                            // PGP score
    LOGISTIC_REGRESSION,                            // Logistic Regression
    PEAK_WEIGHTED,                                  // SSE with equal weight to peaks and non-peaks
    WEIGHTED_SSE,                                   // User provides weights for sse.
    WEIGHTED_CLASSIFIER,
    T_TEST,
    T_TEST_WSSE,
    T_WSSE_DOT,
    SEQWISE,
    POWER
};

ObjType getObjOption( const string& objOptionStr );
string getObjOptionStr( ObjType objOption );

enum SearchType
{
    UNCONSTRAINED,                                // unconstrained search
    CONSTRAINED                                   // constrained search
};

string getSearchOptionStr( SearchType searchOption );


#include "SeqAnnotator.h"

/*
class PredictorTrainer {
public:
	PredictorTrainer();
	virtual ~PredictorTrainer();

    int nConds() const
    {
        return exprData.nCols();
    }

	double getObj() const { return obj_model; }

	// the objective function to be minimized
	double objFunc( const ExprPar& par ) ;

	// training the model
	int train( const ExprPar& par_init );     // training with the initial values given
											  // training with the initial values and allowing random starts
	int train( const ExprPar& par_init, const gsl_rng* rng );
	int train();                              // automatic training: first estimate the initial values, then train

	// predict expression values of a sequence (across the same conditions)
	int predict( const SiteVec& targetSites, int targetSeqLength, vector< double >& targetExprs, int seq_num ) const;

	// test the model, perfOption = 0: RMSE
	// 	double test( const vector< Sequence  >& testSeqs, const Matrix& testExprData, Matrix& predictions ) const;

	//std::ofstream gene_crm_fout;

	static ObjType objOption;                 // option of the objective function

	// the similarity between two expression patterns, using cross-correlation
	static double exprSimCrossCorr( const vector< double >& x, const vector< double >& y );
	static int maxShift;                      // maximum shift when computing cross correlation
	static double shiftPenalty;               // the penalty for shift (when weighting different positions)

	// the parameters for the optimizer
	static int nAlternations;                 // number of alternations (between two optimization methods)
	static int nRandStarts;                   // number of random starts
	static double min_delta_f_SSE;            // the minimum change of the objective function under SSE
	static double min_delta_f_Corr;           // the minimum change of the objective function under correlation
	static double min_delta_f_CrossCorr;      // the minimum change of the objective function under cross correlation
	static double min_delta_f_PGP;            // the minimum change of the objective function under PGP
	static int nSimplexIters;                 // maximum number of iterations for Simplex optimizer
	static int nGradientIters;                // maximum number of iterations for Gradient optimizer
	vector < bool > indicator_bool;


    vector < double > fix_pars;
    vector < double > free_pars;

private:
    // training data
            const vector< SiteVec >& seqSites;        // the extracted sites for all sequences
            const vector< int >& seqLengths;          // lengths of all sequences
            //TODO: R_SEQ Either remove this dead feature or revive it and make it conditional.
            const vector <SiteVec>& r_seqSites;
            const vector< int >& r_seqLengths;        // lengths of all sequences
            const Matrix& exprData;	// expressions of the corresponding sequences across multiple conditions
            const Matrix& factorExprData;             // [TF] of all factors over multiple conditions
            const vector < int >& axis_start;
            const vector < int >& axis_end;
            const vector < double >& axis_wts;

            // randomly sample parameter values (only those free parameters), the parameters should be initialized
                    int randSamplePar( const gsl_rng* rng, ExprPar& par ) const;

                    // print the parameter values (the ones that are estimated) in a single line
                    void printPar( const ExprPar& par ) const;


                    // objective functions
                    double compRMSE( const ExprPar& par );    // root mean square error between predicted and observed expressions
                    double compAvgCorr( const ExprPar& par ); // the average Pearson correlation
                                                              // the average cross correlation -based similarity
                    double compAvgCrossCorr( const ExprPar& par );
                    double compPGP( const ExprPar& par );     // the average cross correlation -based similarity

                    // minimize the objective function, using the current model parameters as initial values
                                                              // simplex
                    int simplex_minimize( ExprPar& par_result, double& obj_result );
                                                              // gradient: BFGS or conjugate gradient
                    int gradient_minimize( ExprPar& par_result, double& obj_result );
                    //  	int SA_minimize( ExprPar& par_result, double& obj_result ) const;	// simulated annealing
            	int optimize_beta( ExprPar& par_result, double& obj_result); // find the current best beta with one-step otimization.

};


// the objective function and its gradient of ExprPredictor::simplex_minimize or gradient_minimize
double gsl_obj_f( const gsl_vector* v, void* params );
void gsl_obj_df( const gsl_vector* v, void* params, gsl_vector* grad );
void gsl_obj_fdf( const gsl_vector* v, void* params, double* result, gsl_vector* grad );
*/
#endif /* SRC_PREDICTORTRAINER_H_ */
