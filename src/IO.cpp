#include "IO.h"

int readEdgelistGraph( const string& filename, const map<string, int>& factorIdxMap, IntMatrix& destination, bool directed){
	string factor1, factor2;

	ifstream matrix_infile(filename.c_str());
	if( !matrix_infile.is_open() ){
		return RET_ERROR;
	}

	while( matrix_infile >> factor1 >> factor2 ){
		assert( factorIdxMap.count( factor1 ) && factorIdxMap.count( factor2 ) );
		int idx1 = factorIdxMap.at(factor1);
		int idx2 = factorIdxMap.at(factor2);

		destination.setElement( idx1, idx2, true);
		if(!directed){
			destination.setElement( idx2, idx1, true);
		}
	}

	return 0;
}

int readFactorThresholdFile( const string& filename, vector< double >& destination, int nFactors){

	ifstream factor_thr_input( filename.c_str() );

	if(!factor_thr_input.is_open() ){
		cerr << "Cannot open the factor threshold file " << filename << endl;
		return RET_ERROR;
	}

	destination.clear();

	for( int index = 0; index < nFactors; index++ )
        {
            double temp;
            factor_thr_input >> temp;
            destination.push_back( temp );
        }
	//Ensure there is only whitespace remaining in the file!
        factor_thr_input.close();

	return 0;
}

int readFactorRoleFile(const string& filename, const map<string, int>& factorIdxMap,  vector< bool>& actIndicators, vector<bool>& repIndicators){

	ifstream finfo( filename.c_str());
	if(!finfo.is_open()){
		cerr << "Cannot open the factor information file " << filename << endl;
		return RET_ERROR;
	}

	string name;
	int i = 0, actRole, repRole;
	while( finfo >> name >> actRole >> repRole )
	{
		if( factorIdxMap.at(name) != i){
			cerr << "An entry in the factor information file was out of order or otherwise invalid: " << filename << ":" << i+1 << endl;
			return RET_ERROR;
		}

		if( (actRole != 0 && actRole != 1) || (repRole != 0 && repRole !=1) ){
			cerr << "An invalid role setting was provided in the factor information file: " << filename << ":" << i+1 << endl;
			return RET_ERROR;
		}

		actIndicators[i] = (1 == actRole);//This does not rely on outside initialization of the actIndicators or repIndicators, other than instantiation.
		repIndicators[i] = (1 == repRole);
		i++;
	}

	finfo.close();

	return 0;
}

int readAxisWeights(const string& filename, vector< int >& axis_start, vector< int >& axis_end, vector< double >& axis_wts){
	//Seems to load a file of ranges and weights. All weights should total 100.
	//Ranges start at 0
	ifstream axis_wtInfo ( filename.c_str() );
        if( !axis_wtInfo )
        {
            cerr << "Cannot open the axis weight information file " << filename << endl;
            exit( 1 );
        }
        int temp1, temp2;
        double temp3;
        double temp3_sum = 0;
        while( axis_wtInfo >> temp1 >> temp2 >> temp3 ) //TODO: Change to readline to enforce end of lines
        {
            axis_start.push_back( temp1 );
            axis_end.push_back( temp2 );
            axis_wts.push_back( temp3 );
            temp3_sum += temp3;
        }
        assert( !( temp3_sum > 100 ) && !( temp3_sum < 100 ));

	return 0;
}

int writePredictions(const string& filename, ExprPredictor& predictor, const Matrix& exprData, vector< string >& expr_condNames, bool write_gt, bool fix_beta /*= false*/){
	// print the predictions
	//ofstream threshout("threshold.txt");
	ofstream fout( filename.c_str() );
	if ( !fout )
	{
		cerr << "Cannot open file " << filename << endl;
		exit( 1 );
	}
	ExprPar par = predictor.getPar();
	//SiteVec unusedSV = SiteVec();
	vector< vector< double > > predictions;

	predictor.predict_all(par, predictions);

	fout << "Rows\t" << expr_condNames << endl;

	/*if(objtype == "Learn" && predtype == "Classifier")
	{
		for(int j = 0; j < predictor.nConds(); j++)
		{
			double temp = 0.0;
			for ( int i = 0; i < predictor.nSeqs(); i++ )
			{
				temp = temp + (exprData.getRow(i)[j]*predictions[i][j])/float(predictor.nSeqs());
			}
			threshold[j] = temp;
		}
		threshout << threshold << endl;		
	}*/


	for ( int i = 0; i < predictor.nSeqs(); i++ )
    {
        vector< double > targetExprs = predictions[i];
        /*if(predtype == "Classifier")
        {
        	vector<double> class_predictions(predictor.nConds(), 0);
        	if(objtype == "Predict")
        	{
        		for(int j = 0; j < predictor.nConds(); j++)
        		{
        			class_predictions[j] = (targetExprs[j] > threshold[j]);
        		}
        		//threshout << "It works." << endl;
        		//threshout << threshold << endl;
        		//threshout << predictor.nConds() << endl;
        		//threshout << class_predictions << endl;
        	}
        }*/
        vector< double > observedExprs = exprData.getRow( i );

        // error
        // print the results
				// observations
				if( write_gt ){
        	fout << predictor.seqs[i].getName() << "_GT" << "\t" << observedExprs << "\t";
				}
        fout << predictor.seqs[i].getName();

        double beta = par.getBetaForSeq(i);

				double error_or_score;

				vector<vector<double> > multiple_predictions;
				vector<vector<double> > multiple_observations;
				multiple_predictions.push_back(targetExprs);
				multiple_observations.push_back(observedExprs);

				error_or_score = predictor.trainingObjective->eval(multiple_predictions, multiple_observations, &par);

        for ( int j = 0; j < predictor.nConds(); j++ ){
						fout << "\t" << ( beta * targetExprs[j] );
				}
        fout << endl;

        // print the agreement bewtween predictions and observations
        cout << predictor.seqs[i].getName() << "\t" << beta << "\t" << error_or_score << endl;
    }

    //ofstream threshout("threshold.txt");
    //if(predtype == "Classifier")
    //{
    //	threshout << "############# It's working. ##############" << endl;
    //	threshout << threshold << endl;
    //}

	return 0;
}

int writePredictions_classifier(const string& filename, ExprPredictor& predictor, const Matrix& exprData, vector< string >& expr_condNames, vector<double>& threshold, bool learn, bool write_gt, bool fix_beta /*= false*/)
{
	ofstream fout( filename.c_str() );
	if ( !fout )
	{
		cerr << "Cannot open file " << filename << endl;
		exit( 1 );
	}
	ExprPar par = predictor.getPar();
	//SiteVec unusedSV = SiteVec();
	vector< vector< double > > predictions;

	predictor.predict_all(par, predictions);

	fout << "Rows\t" << expr_condNames << "\tRows_Classifier\t" << expr_condNames << "\tRows_Actual\t" << expr_condNames << endl;
	if(learn == true)
	{

		//cerr << "learn activated" << endl;	
		ofstream threshout("threshold.txt");
		for(int j = 0; j < predictor.nConds(); j++)
		{
			double temp1 = 0.0;
			double temp0 = 0.0;
			int count1 = 0;
			int count0 = 0;
			for ( int i = 0; i < predictor.nSeqs(); i++ )
			{
				temp1 = temp1 + exprData.getRow(i)[j]*predictions[i][j];
				temp0 += (1-exprData.getRow(i)[j])*predictions[i][j];
				count1 = count1 + exprData.getRow(i)[j];
				count0 += (1 - exprData.getRow(i)[j]);
				//temp = temp + predictions[i][j]/float(predictor.nSeqs());
			}
			double temp = (temp1/float(count1) + temp0/float(count0))/2;
			threshold.push_back(temp);
		}	
		threshout << threshold << endl;
	}


	for ( int i = 0; i < predictor.nSeqs(); i++ )
    {
        vector< double > targetExprs = predictions[i];
        vector<double> class_predictions(predictor.nConds(), 0);
        for(int j = 0; j < predictor.nConds(); j++)
    	{
       		class_predictions[j] = (targetExprs[j] > threshold[j]);
    	}
		vector< double > observedExprs = exprData.getRow( i );

        // error
        // print the results
				// observations
		if( write_gt )
		{
        fout << predictor.seqs[i].getName() << "_GT" << "\t" << observedExprs << "\t";
		}
        fout << predictor.seqs[i].getName();

        double beta = par.getBetaForSeq(i);
		double error_or_score;
		vector<vector<double> > multiple_predictions;
		vector<vector<double> > multiple_observations;
		multiple_predictions.push_back(targetExprs);
		multiple_observations.push_back(observedExprs);

		error_or_score = predictor.trainingObjective->eval(multiple_predictions, multiple_observations, &par);
		
        for ( int j = 0; j < predictor.nConds(); j++ )
        {
			//fout << "\t" << ( beta * targetExprs[j] );
			fout << "\t" << class_predictions[j];
		}
		fout << "\t" << predictor.seqs[i].getName();
		for ( int j = 0; j < predictor.nConds(); j++ )
		{
			fout << "\t" << ( beta * targetExprs[j] );
			//fout << "\t" << targetExprs[j];
			//fout << "\t" << class_predictions[j];
		}	
		fout << endl;

        // print the agreement bewtween predictions and observations
        cout << predictor.seqs[i].getName() << "\t" << beta << "\t" << error_or_score << endl;
    }

    //ofstream threshout("threshold.txt");
    //if(predtype == "Classifier")
    //{
    //	threshout << "############# It's working. ##############" << endl;
    //	threshout << threshold << endl;
    //}

	return 0;	
}
