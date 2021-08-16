#include "ObjFunc.h"

double RMSEObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double squaredErr = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
      double beta = 1.0;
      #ifdef BETAOPTTOGETHER
        if(NULL != par)
          beta = par->getBetaForSeq(i);
        squaredErr += least_square( prediction[i], ground_truth[i], beta, true );
      #else
        squaredErr += least_square( prediction[i], ground_truth[i], beta );
      #endif
  }

    double rmse = sqrt( squaredErr / ( nSeqs * nConds ) );
    return rmse;
}

double AvgCorrObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double totalSim = 0.0;

    for(int i = 0;i<ground_truth.size();i++){

      totalSim += abs( corr(  prediction[i], ground_truth[i] ) );
  }

    return -totalSim/nSeqs;
  }

double PGPObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

        assert(ground_truth.size() == prediction.size());
        int nSeqs = ground_truth.size();
        int nConds = ground_truth[0].size();
        double totalPGP = 0.0;

        for(int i = 0;i<ground_truth.size();i++){
          double beta = 1.0;
          #ifdef BETAOPTTOGETHER
        	beta = par->getBetaForSeq(i);
                totalPGP += pgp(  prediction[i], ground_truth[i], beta, true);
        	#else
        	totalPGP += pgp(  prediction[i], ground_truth[i], beta );
        	#endif
      }

      return totalPGP / nSeqs;
  }

double AvgCrossCorrObjFunc::exprSimCrossCorr( const vector< double >& x, const vector< double >& y )
  {
      vector< int > shifts;
      for ( int s = -maxShift; s <= maxShift; s++ )
      {
          shifts.push_back( s );
      }

      vector< double > cov;
      vector< double > corr;
      cross_corr( x, y, shifts, cov, corr );
      double result = 0, weightSum = 0;
      //     result = corr[maxShift];
      result = *max_element( corr.begin(), corr.end() );
      //     for ( int i = 0; i < shifts.size(); i++ ) {
      //         double weight = pow( shiftPenalty, abs( shifts[i] ) );
      //         weightSum += weight;
      //         result += weight * corr[i];
      //     }
      //     result /= weightSum;

      return result;
  }

double AvgCrossCorrObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){

    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double totalSim = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
        totalSim += exprSimCrossCorr( prediction[i], ground_truth[i] );
    }

  return -totalSim / nSeqs;
  }


double LogisticRegressionObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){
    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double totalLL = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
      vector<double> Y = ground_truth[i];
      vector<double> Ypred = prediction[i];

      double one_sequence_LL = 0.0;

      for(int j = 0;j<Y.size();j++){
        double one_gt = Y[i];
        double pred_prob = logistic(w*(Ypred[j] - bias));
        double singleLL = one_gt*log(pred_prob) + (1.0 - one_gt)*log(1.0 - pred_prob);
        one_sequence_LL += singleLL;
      }

      totalLL += one_sequence_LL;
    }
    return -totalLL;
}

RegularizedObjFunc::RegularizedObjFunc(ObjFunc* wrapped_obj_func, const ExprPar& centers, const ExprPar& l1, const ExprPar& l2)
{
  my_wrapped_obj_func = wrapped_obj_func;
  ExprPar tmp_energy_space = centers.my_factory->changeSpace(centers, ENERGY_SPACE);
  tmp_energy_space.getRawPars(my_centers );

  //It doesn't matter what space these are in, they are just storage for values.
  l1.getRawPars(lambda1 );
  l2.getRawPars(lambda2 );
  cache_pars = vector<double>(my_centers.size(),0.0);
  //cache_diffs(my_centers.size(),0.0);
  //cache_sq_diffs(my_centers.size(),0.0);
}

double RegularizedObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction, const ExprPar* par){

  double objective_value = my_wrapped_obj_func->eval( ground_truth, prediction, par );



  double l1_running_total = 0.0;
  double l2_running_total = 0.0;

  ExprPar tmp_energy_space = par->my_factory->changeSpace(*par, ENERGY_SPACE);
  tmp_energy_space.getRawPars(cache_pars );

  for(int i = 0;i<cache_pars.size();i++){
    double the_diff = abs(cache_pars[i] - my_centers[i]);
    l1_running_total += lambda1[i]*the_diff;
    l2_running_total += lambda2[i]*pow(the_diff,2.0);
  }

  objective_value += l1_running_total + l2_running_total;

  return objective_value;
}

double PeakWeightedObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){
    double on_threshold = 0.5;
    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double squaredErr = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
      double beta = 1.0;
      #ifdef BETAOPTTOGETHER
        if(NULL != par)
          beta = par->getBetaForSeq(i);
        squaredErr += wted_least_square( prediction[i], ground_truth[i], beta, on_threshold, true );
      #else
        squaredErr += wted_least_square( prediction[i], ground_truth[i], beta, on_threshold );
      #endif
  }

    double rmse = sqrt( squaredErr / ( nSeqs * nConds ) );
    return rmse;
}

double Weighted_RMSEObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){
  	//cerr << "Weighted Objective called." << endl;
    #ifndef BETAOPTTOGETHER
        assert(false);
    #endif

    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double squaredErr = 0.0;

    for(int i = 0;i<ground_truth.size();i++){
      double beta = 1.0;
      if(NULL != par){ beta = par->getBetaForSeq(i); }

      for(int j = 0;j<nConds;j++){
          double single_sqr_error = (beta*prediction[i][j] - ground_truth[i][j]);
          single_sqr_error = weights->getElement(i,j)*pow(single_sqr_error,2);
          squaredErr += single_sqr_error;
      }
    }
    //total_weight = 1;
    double rmse = sqrt( squaredErr / total_weight );
    return rmse;
}


void Weighted_ClassifierObjFunc::Fconverter(vector<vector<double> >& f_prediction)
{
  for(int i = 0; i < f_prediction.size(); i ++)
  {
    for(int j = 0; j < f_prediction[0].size(); j++)
    {
      double temp = f_prediction[i][j];
      /*if(temp == 0)
      {
        temp = 0.00001;
      }
      if(temp == 1)
      {
        temp = 0.99999;
      }*/
      f_prediction[i][j] = (1-temp)/double(temp);
    }
  }
}

double Weighted_ClassifierObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,const ExprPar* par)
{
  vector<vector<double> > f_prediction = prediction;
  Fconverter(f_prediction);
  double error = 0;

  #ifndef BETAOPTTOGETHER
    assert(false);
  #endif

  assert(ground_truth.size() == f_prediction.size());
  int nSeqs = ground_truth.size();
  int nConds = ground_truth[0].size();

  for(int i = 0; i < nSeqs; i++)
  {
    for(int j = 0; j < nConds; j++)
    {
      if(ground_truth[i][j] == 0)
      {
        error += weights->getElement(i,j)*log(1 + 1/float(f_prediction[i][j]));
      }
      else if(ground_truth[i][j] == 1)
      {
        error += weights->getElement(i,j)*log(1 + f_prediction[i][j]);
      }
    }
  }
  error = error/total_weight;
  return error;
}

double tTestKindaObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,const ExprPar* par)
{
  double t_error = 0;

  assert(ground_truth.size() == prediction.size());
  int nSeqs = ground_truth.size();
  int nConds = ground_truth[0].size();
  
  for(int j = 0; j < nConds; j++)
  {
    int nExpr = 0;
    int nNotExpr = 0;
    double meanExpr = 0;
    double meanNotExpr = 0;
    double meanSqrExpr = 0;
    double meanSqrNotExpr = 0;

    for(int i = 0; i < nSeqs; i++)
    {
      double beta = 1.0;
      if(NULL != par){ beta = par->getBetaForSeq(i); }
      if(ground_truth[i][j] == 1)
      {
        nExpr += 1;
        meanExpr += beta*prediction[i][j];
        meanSqrExpr += pow(beta*prediction[i][j], 2);
      }
      else if(ground_truth[i][j] == 0)
      {
        nNotExpr += 1;
        meanNotExpr += beta*prediction[i][j];
        meanSqrNotExpr += pow(beta*prediction[i][j], 2); 
      }      
    }
    meanExpr = meanExpr/float(nExpr);
    meanNotExpr = meanNotExpr/float(nNotExpr);
    meanSqrExpr = meanSqrExpr/float(nExpr);
    meanSqrNotExpr = meanSqrNotExpr/float(nNotExpr);
    double varExpr_wt = (meanSqrExpr - pow(meanExpr,2))/float(nExpr);
    double varNotExpr_wt = (meanSqrNotExpr - pow(meanNotExpr,2))/float(nNotExpr);
    double var = varExpr_wt + varNotExpr_wt;
    if(var <= 0)
    {
      var = 0.00000001;
    }
    t_error += (meanNotExpr - meanExpr)/sqrt(var);
  }
  
  return t_error;
}

//combined weighted sse and t-test objective with weight lambda for t-test. 
double t_WSSE_ObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,const ExprPar* par)
{
  double t_error = 0;
  double squaredErr = 0.0;
  double lambda = 0.01;

  assert(ground_truth.size() == prediction.size());
  int nSeqs = ground_truth.size();
  int nConds = ground_truth[0].size();
  
  for(int j = 0; j < nConds; j++)
  {
    int nExpr = 0;
    int nNotExpr = 0;
    double meanExpr = 0;
    double meanNotExpr = 0;
    double meanSqrExpr = 0;
    double meanSqrNotExpr = 0;

    for(int i = 0; i < nSeqs; i++)
    {
      double beta = 1.0;
      if(NULL != par){ beta = par->getBetaForSeq(i); }
      if(ground_truth[i][j] == 1)
      {
        nExpr += 1;
        //meanExpr += beta*prediction[i][j];
        //meanSqrExpr += pow(beta*prediction[i][j], 2);
        meanExpr += prediction[i][j];
        meanSqrExpr += pow(prediction[i][j], 2);
      }
      else if(ground_truth[i][j] == 0)
      {
        nNotExpr += 1;
        //meanNotExpr += beta*prediction[i][j];
        //meanSqrNotExpr += pow(beta*prediction[i][j], 2); 
        meanNotExpr += prediction[i][j];
        meanSqrNotExpr += pow(prediction[i][j], 2); 
      }
      double single_sqr_error = (beta*prediction[i][j] - ground_truth[i][j]);
      single_sqr_error = weights->getElement(i,j)*pow(single_sqr_error,2);
      squaredErr += single_sqr_error;
    }
    meanExpr = meanExpr/float(nExpr);
    meanNotExpr = meanNotExpr/float(nNotExpr);
    meanSqrExpr = meanSqrExpr/float(nExpr);
    meanSqrNotExpr = meanSqrNotExpr/float(nNotExpr);
    double varExpr_wt = (meanSqrExpr - pow(meanExpr,2))/float(nExpr);
    double varNotExpr_wt = (meanSqrNotExpr - pow(meanNotExpr,2))/float(nNotExpr);
    double var = varExpr_wt + varNotExpr_wt;
    if(var <= 0)
    {
      var = 0.00000001;
    }
    //t_error += pow((meanNotExpr - meanExpr),3)/var; 
    t_error += (meanNotExpr - meanExpr)/sqrt(var);
   }
  double wsse = sqrt( squaredErr / total_weight ); 
  //double error = wsse - lambda*t_error;
  double error = wsse + lambda*t_error;
  return error;
}

double t_WSSE_dot_ObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,const ExprPar* par)
{ 
  double t_error = 0;
  double squaredErr = 0.0;
  double lambda = 0.01;
  vector<double> betas;
  double temp = 0.0;
  double norm = 0.0;
  double gt_norm = 0.0;
  double temp_beta = 0.0;

  assert(ground_truth.size() == prediction.size());
  int nSeqs = ground_truth.size();
  int nConds = ground_truth[0].size();

  /*for(int i = 0; i < nSeqs; i++)
  { 
    temp = 0.0;
    norm = 0.0;
    for(int j = 0; j < nConds; j++)
    {
      temp += prediction[i][j]*ground_truth[i][j];
      norm += prediction[i][j]*prediction[i][j];
      //gt_norm += ground_truth[i][j]*ground_truth[i][j];
    }
    //temp_beta = temp/(norm*sqrt(gt_norm));
    temp_beta = temp/norm;
    betas.push_back(temp_beta);
  }*/
  
  for(int j = 0; j < nConds; j++)
  {
    int nExpr = 0;
    int nNotExpr = 0;
    double meanExpr = 0;
    double meanNotExpr = 0;
    double meanSqrExpr = 0;
    double meanSqrNotExpr = 0;

    for(int i = 0; i < nSeqs; i++)
    {
      double beta = 1.0;
      //beta = betas[i];
      if(ground_truth[i][j] == 1)
      {
        nExpr += 1;
        meanExpr += beta*prediction[i][j];
        meanSqrExpr += pow(beta*prediction[i][j], 2);
        //meanExpr += prediction[i][j];
        //meanSqrExpr += pow(prediction[i][j], 2);
      }
      else if(ground_truth[i][j] == 0)
      {
        nNotExpr += 1;
        meanNotExpr += beta*prediction[i][j];
        meanSqrNotExpr += pow(beta*prediction[i][j], 2); 
        //meanNotExpr += prediction[i][j];
        //meanSqrNotExpr += pow(prediction[i][j], 2); 
      }
      double single_sqr_error = (beta*prediction[i][j] - ground_truth[i][j]);
      single_sqr_error = weights->getElement(i,j)*pow(single_sqr_error,2);
      squaredErr += single_sqr_error;
    }
    meanExpr = meanExpr/float(nExpr);
    meanNotExpr = meanNotExpr/float(nNotExpr);
    meanSqrExpr = meanSqrExpr/float(nExpr);
    meanSqrNotExpr = meanSqrNotExpr/float(nNotExpr);
    double varExpr_wt = (meanSqrExpr - pow(meanExpr,2))/float(nExpr);
    double varNotExpr_wt = (meanSqrNotExpr - pow(meanNotExpr,2))/float(nNotExpr);
    double var = varExpr_wt + varNotExpr_wt;
    if(var <= 0)
    {
      var = 0.0000000000001;
    } 
    t_error += (meanNotExpr - meanExpr)/sqrt(var);
   }
  double wsse = sqrt( squaredErr / total_weight ); 
  double error = wsse + lambda*t_error;
  return error;
}

double SeqWise_ObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,const ExprPar* par)
{
  double t_error = 0;
  double squaredErr = 0.0;
  double lambda = 0.01;
  assert(ground_truth.size() == prediction.size());
  int nSeqs = ground_truth.size();
  int nConds = ground_truth[0].size();
  double maxterror = -0.000000001;
  for(int i = 0; i < nSeqs; i++)
  {
    int nExpr = 0;
    int nNotExpr = 0;
    double meanExpr = 0;
    double meanNotExpr = 0;
    double meanSqrExpr = 0;
    double meanSqrNotExpr = 0;
    double varExpr_wt = 0;
    double varNotExpr_wt = 0;

    for(int j = 0; j < nConds; j++)
    {

      double beta = 1.0;
      //beta = betas[i];
      if(ground_truth[i][j] == 1)
      {
        nExpr += 1;
        meanExpr += beta*prediction[i][j];
        meanSqrExpr += pow(beta*prediction[i][j], 2);
        //meanExpr += prediction[i][j];
        //meanSqrExpr += pow(prediction[i][j], 2);
      }
      else if(ground_truth[i][j] == 0)
      {
        nNotExpr += 1;
        meanNotExpr += beta*prediction[i][j];
        meanSqrNotExpr += pow(beta*prediction[i][j], 2); 
        //meanNotExpr += prediction[i][j];
        //meanSqrNotExpr += pow(prediction[i][j], 2); 
      }
      double single_sqr_error = (beta*prediction[i][j] - ground_truth[i][j]);
      single_sqr_error = weights->getElement(i,j)*pow(single_sqr_error,2);
      squaredErr += single_sqr_error;
    }

    meanExpr = meanExpr/float(nExpr);
    meanSqrExpr = meanSqrExpr/float(nExpr);
    varExpr_wt = (meanSqrExpr - pow(meanExpr,2))/float(nExpr);
    if(nNotExpr == 0)
    {
      meanNotExpr = 0;
      meanSqrNotExpr = 0;
      varNotExpr_wt = 0;
    }
    else
    {
      meanNotExpr = meanNotExpr/float(nNotExpr);
      meanSqrNotExpr = meanSqrNotExpr/float(nNotExpr);
      varNotExpr_wt = (meanSqrNotExpr - pow(meanNotExpr,2))/float(nNotExpr); 
    }
    double var = varExpr_wt + varNotExpr_wt;
    if(var <= 0)
    {
      var = 0.00000001;
    }
    double tempTerror = (meanNotExpr - meanExpr)/sqrt(var);
    if(tempTerror < maxterror)
    {
      maxterror = tempTerror; 
    }
    t_error += tempTerror ;
  }
  double wsse = sqrt( squaredErr / total_weight ); 
  double error = wsse + lambda*t_error/maxterror;
  //return t_error/maxterror;
  return t_error;
  //return error;  
}

double PowerObjFunc::eval(const vector<vector<double> >& ground_truth, const vector<vector<double> >& prediction,
  const ExprPar* par){
    //cerr << "Weighted Objective called." << endl;
    #ifndef BETAOPTTOGETHER
        assert(false);
    #endif

    assert(ground_truth.size() == prediction.size());
    int nSeqs = ground_truth.size();
    int nConds = ground_truth[0].size();
    double squaredErr = 0.0;

    vector<double> norm;

    for(int i = 0;i<ground_truth.size();i++)
    {
      double temp_norm = 0;
      for(int j = 0;j<nConds;j++)
      {
        temp_norm += pow(prediction[i][j], 3);
      }
      norm.push_back(temp_norm);
    }

    for(int i = 0;i<ground_truth.size();i++){
      double beta = 1.0;
      if(NULL != par){ beta = par->getBetaForSeq(i); }

      for(int j = 0;j<nConds;j++){
          double single_sqr_error = (beta*pow(prediction[i][j], 3)/norm[i] - ground_truth[i][j]);
          single_sqr_error = weights->getElement(i,j)*pow(single_sqr_error,2);
          squaredErr += single_sqr_error;
      }
    }
    //total_weight = 1;
    double rmse = sqrt( squaredErr / total_weight );
    return rmse;
}



void Weighted_ObjFunc_Mixin::set_weights(Matrix *in_weights){
	//cerr << "Weighted Objective called#" << endl;
    if(NULL != weights){delete weights;}
    weights = in_weights;
    //cerr << weights->nRows() << "\t" << weights->nCols() << endl;

    //Caculate the total weight.
    total_weight = 0.0;
    for(int i = 0;i<weights->nRows();i++){
        for(int j = 0;j<weights->nCols();j++){
            total_weight+=weights->getElement(i,j);
            //cerr << weights->getElement(i,j) << "\t";
        }
        //cerr << endl;
    }
    cerr << total_weight << endl;
}