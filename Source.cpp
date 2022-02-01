#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <thread>
#include <mutex> // std::mutex, std::unique_lock, std::defer_lock
#include <set>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <random>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include "ThreadPool.h"

using namespace std;

typedef struct s
{
	vector<int> lhs;
	vector<int> rhs;
} implication;

typedef struct
{
	boost::dynamic_bitset<unsigned long> lhs;
	boost::dynamic_bitset<unsigned long> rhs;
} implicationBS;

#define TIMEPRINT(X) (((double)X) / ((double)1000000))

vector<vector<int>> objInp;	 //For storing which attributes are associated with which objects
vector<vector<int>> attrInp; //For storing which objects are associated with which attributes
vector<boost::dynamic_bitset<unsigned long>> objInpBS;
vector<int> frequencyOrderedAttributes;
double totalTime = 0;
double totalExecTime2 = 0; //Stores total time spent generating counter examples
double totalClosureTime = 0;
double intersectionTime = 0;
double thisIterMaxImplicationClosureTime = 0;
double thisIterMaxContextClosureTime = 0;
double updownTime = 0;
int numThreads = 1, maxThreads;
long long totCounterExamples = 0;
bool globalFlag; //For terminating other threads in case one thread found a counter-example
boost::dynamic_bitset<unsigned long> counterExampleBS;
bool isPositiveCounterExample = true;
int gCounter = 0; //For counting how many times the equivalence oracle has been used
int totTries = 0;
long long sumTotTries = 0;
long long totClosureComputations = 0;
long long totUpDownComputes = 0; //Stores how many random attribute sets needed to be tested before finding a counter-example. For debugging purposes.
bool basisUpdate = false;
long long countPositiveCounterExample = 0, countNegativeCounterExample = 0;
implicationBS updatedImplication;
int indexOfUpdatedImplication;
int implicationsSeen;
std::mutex mtx; // mutex for critical section
vector<boost::dynamic_bitset<unsigned long>> potentialCounterExamplesBS;
double epsilon, del;
bool epsilonStrong = false, frequentCounterExamples = false, bothCounterExamples = false;
int maxTries; //Updated by getLoopCount() based on the value of gCounter, epsilon and delta.
bool implicationSupport = false;
bool emptySetClosureComputed = false;
boost::dynamic_bitset<unsigned long> emptySetClosure;
long long emptySetClosureComputes = 0;
long long aEqualToCCount = 0;
int notFoundKRules = 1;
std::random_device rd;
std::discrete_distribution<int> discreteDistribution, discreteDistributionArea;
std::discrete_distribution<long long> discreteDistributionSquared;
std::discrete_distribution<long long> discreteDistributionDiscriminativity;
std::binomial_distribution<int> binomialDistribution;
std::default_random_engine re(rd());
thread_local std::minstd_rand new_rand(rd());

vector<std::discrete_distribution<int>> discreteDistributionAttributeSets;
vector<int> objectLabels, positiveObjects, negativeObjects;
vector<long double> attrSetWeight;
vector<implicationBS> ansBasisBS;
vector<pair<int, implicationBS>> updatedImplications;

double threadOverheadTime = 6;
double prevIterTime = 0;
int UpdateImplicationTries = 0;
int prevThreads = 1;
int singletonCounterexamples = 0;
int countClosedPremises = 0;
int k_value;
double minconf_value;
double percentAttrClosure;

//Can be used in case the input format is:
//Each line has the attribute numbers of attributes associated with the object represented by the line number.
int counterexampleType = 1;

// time
std::chrono::_V2::system_clock::time_point startTime, endTime;
double ioTime = 0;

vector<implicationBS> topKRulesBS;
vector<int> topK_times;
int timePointer = 0;

bool debug = true; // print iteration details
int print_count = 0;
int maxImplicationUpdates = 0;
int minconfRulesCount = 0;

void readFormalContext1(string fileName)
{
	ifstream inFile(fileName);
	string line;
	while (getline(inFile, line))
	{
		vector<int> cur;
		istringstream iss(line);
		int x;
		while (iss >> x)
		{
			if (x >= attrInp.size())
				attrInp.resize(x + 1);
			attrInp[x].push_back(objInp.size());
			cur.push_back(x);
		}
		if (cur.size() != 0)
			objInp.push_back(cur);
	}
	//cout << "Done reading context\n";
	//cout << objInp.size() << " " << attrInp.size() << "\n";
	inFile.close();
}

//Can be used if the input format is:
//Line 1 contains number of objects
//Line 2 contains number of attributes
//There is a binary matrix from line 3 which represents the formal context, much like how we studied in class.

void readFormalContext2(string fileName)
{
	ifstream inFile(fileName);
	int obj, attr;
	inFile >> obj >> attr;
	objInp.resize(obj);
	attrInp.resize(attr);
	for (int i = 0; i < obj; i++)
	{
		int x;
		for (int j = 0; j < attr; j++)
		{
			inFile >> x;
			if (x == 1)
			{
				objInp[i].push_back(j);
				attrInp[j].push_back(i);
			}
		}
	}
	//cout << "Done reading formal context\n";
	//cout << objInp.size() << " " << attrInp.size() << "\n";
	inFile.close();
}

void readLabels(string labelFile)
{
	ifstream labelInput(labelFile);
	int temp;
	int oID = 0;

	while (labelInput >> temp)
	{
		objectLabels.push_back(temp);

		if (temp == 0)
			negativeObjects.push_back(oID);
		else
			positiveObjects.push_back(oID);

		oID++;
	}

	labelInput.close();
}

long double nChooseK(long long n, long long k)
{
	long double res = 1;
	for (long long i = 1; i <= k; ++i)
		res = res * ((long double)(n - k + i)) / ((long double)i);
	return res;
}

void initializeRandSetGen()
{
	vector<long double> powersOfTwo(attrInp.size() + 2);
	powersOfTwo[0] = 1;

	for (int i = 1; i < powersOfTwo.size(); i++)
		powersOfTwo[i] = ((long double)2) * powersOfTwo[i - 1];

	attrSetWeight.resize(objInp.size());

	for (int i = 0; i < objInp.size(); i++)
	{
		attrSetWeight[i] = powersOfTwo[objInp[i].size()];
	}

	discreteDistribution = std::discrete_distribution<int>(attrSetWeight.begin(), attrSetWeight.end());

	if (counterexampleType == 2)
	{
		for (int i = 0; i < objInp.size(); i++)
		{
			attrSetWeight[i] *= ((long double)objInp[i].size()) * ((long double)0.5);
		}

		discreteDistributionArea = std::discrete_distribution<int>(attrSetWeight.begin(), attrSetWeight.end());

		discreteDistributionAttributeSets.resize(attrInp.size() + 2);

		for (int i = 0; i < discreteDistributionAttributeSets.size(); i++)
		{
			vector<long double> nChooseKWeights(i + 1);

			for (int j = 0; j <= i; j++)
			{
				nChooseKWeights[j] = nChooseK(i, j);
			}

			discreteDistributionAttributeSets[i] = std::discrete_distribution<int>(nChooseKWeights.begin(), nChooseKWeights.end());
		}
	}

	if (counterexampleType == 3)
	{
		long long numObj = objInp.size();
		vector<long double> weights(numObj * numObj);
		long long size = 0;

		for (long long i = 0; i < numObj; i++)
		{
			for (long long j = 0; j < numObj; j++)
			{
				long double power = (objInpBS[i] & objInpBS[j]).count();
				weights[size] = powersOfTwo[power];
				size++;
			}
		}

		discreteDistributionSquared = std::discrete_distribution<long long>(weights.begin(), weights.end());
	}

	if (counterexampleType == 4)
	{
		long long numObj = objInp.size();
		vector<long double> weights(numObj * numObj);
		long long size = 0;

		for (long long i = 0; i < numObj; i++)
		{
			for (long long j = 0; j < numObj; j++)
			{
				long double power = 0;

				if ((objectLabels[i] == 0) && (objectLabels[j] == 1))
				{
					power = powersOfTwo[(objInpBS[i] - objInpBS[j]).count()];
					power = (power - 1) * powersOfTwo[(objInpBS[i] & objInpBS[j]).count()];
				}

				weights[size] = power;
				size++;
			}
		}

		discreteDistributionDiscriminativity = std::discrete_distribution<long long>(weights.begin(), weights.end());
	}

	if (counterexampleType == 5)
	{
		int n = objInp.size();
		float p = 0.2;
		binomialDistribution = std::binomial_distribution<int>(n, p);
		// cout << "\nGenerated binomial dist with n = " << n << ", p = " << p << "\n";
	}
}

void getLoopCount()
{
	double loopCount = log(del / ((double)(gCounter * (gCounter + 1))));
	loopCount = loopCount / log(1 - epsilon);
	maxTries = (int)ceil(loopCount);
}

void printVector(vector<int> &A)
{
	for (auto x : A)
	{
		cout << x << " ";
	}
}

vector<int> attrBSToAttrVector(boost::dynamic_bitset<unsigned long> &attrBS)
{
	vector<int> ans;
	// //cout <<"BS = "<< attrBS <<"\n";

	for (int i = 0; i < attrBS.size(); i++)
	{
		if (attrBS[i])
			ans.push_back(i);
	}

	return ans;
}

boost::dynamic_bitset<unsigned long> attrVectorToAttrBS(vector<int> &attrVec)
{
	boost::dynamic_bitset<unsigned long> ans(attrInp.size());

	for (int i = 0; i < attrVec.size(); i++)
	{
		ans[attrVec[i]] = true;
	}

	// //cout<<"BS = "<< ans <<"\n";
	return ans;
}

boost::dynamic_bitset<unsigned long> contextClosureBS(boost::dynamic_bitset<unsigned long> &aset)
{
	totUpDownComputes++;
	boost::dynamic_bitset<unsigned long> aBS = aset, ansBS(attrInp.size());
	ansBS.set();
	ansBS[0] = false;

	int aid = -1;
	int osize = objInp.size() + 1;

	// find min object-size among all attributes present in input
	for (int i = 0; i < aset.size(); i++)
	{
		if (aset[i] && (attrInp[i].size() < osize))
		{
			osize = attrInp[i].size();
			aid = i;
		}
	}

	if (aid != -1)
	{
		for (int i = 0; i < attrInp[aid].size(); i++)
		{
			int cObj = attrInp[aid][i];

			if (aBS.is_subset_of(objInpBS[cObj]))
			{
				ansBS &= objInpBS[cObj];
			}

			// if(ansBS.count() == aBS.count())
			// 	return ansBS;
		}
	}

	else // if no attr is set in input
	{
		emptySetClosureComputes++;

		if (emptySetClosureComputed)
			return emptySetClosure;

		for (int i = 0; i < objInp.size(); i++)
		{
			int cObj = i;
			ansBS &= objInpBS[cObj];
		}

		emptySetClosure = ansBS;
		emptySetClosureComputed = true;
	}

	return ansBS;
}

int biasInclusion(double bias)
{
	if ((rand() / RAND_MAX) < bias)
	{
		return 1;
	}
	return 0;
}

boost::dynamic_bitset<unsigned long> randomContextClosureBS(boost::dynamic_bitset<unsigned long> &aset, double percentObj, int threadIndex)
{
	totUpDownComputes++;
	percentObj = ceil(RAND_MAX * percentObj);
	boost::dynamic_bitset<unsigned long> aBS = aset, ansBS(attrInp.size());
	ansBS.set();
	ansBS[0] = false;

	int aid = -1;
	int osize = objInp.size() + 1;

	// find min object-size among all attributes present in input
	for (int i = 0; i < aset.size(); i++)
	{
		if (aset[i] && (attrInp[i].size() < osize))
		{
			osize = attrInp[i].size();
			aid = i;
		}
	}

	if (aid != -1)
	{
		for (int i = 0; i < attrInp[aid].size(); i++)
		{
			int cObj = attrInp[aid][i];

			if (aBS.is_subset_of(objInpBS[cObj]))
			{
				if ((new_rand() < percentObj))
					ansBS &= objInpBS[cObj];
			}

			// if(ansBS.count() == aBS.count())
			// 	return ansBS;
		}
	}

	else // if no attr is set in input
	{
		emptySetClosureComputes++;

		if (emptySetClosureComputed)
			return emptySetClosure;

		for (int i = 0; i < objInp.size(); i++)
		{
			int cObj = i;
			ansBS &= objInpBS[cObj];
		}

		emptySetClosure = ansBS;
		emptySetClosureComputed = true;
	}

	return ansBS;
}

//Given L and X, find L(X).
boost::dynamic_bitset<unsigned long> closureBS(vector<implicationBS> &basis, boost::dynamic_bitset<unsigned long> X)
{
	totClosureComputations++;
	if (basis.size() == 0)
		return X;
	vector<bool> cons;
	for (int i = 0; i <= basis.size(); i++)
		cons.push_back(false);
	bool changed = true;

	while (changed)
	{
		changed = false;

		for (int i = 0; i < basis.size(); i++)
		{
			if (cons[i] == true)
				continue;

			if (basis[i].lhs.is_subset_of(X))
			{
				cons[i] = true;

				if (!basis[i].rhs.is_subset_of(X))
				{
					X |= basis[i].rhs;
					changed = true;
					break;
				}
			}
		}
	}

	return X;
}

bool isSetEqualToImpCLosure(vector<implicationBS> &basis, boost::dynamic_bitset<unsigned long> &X)
{
	for (int i = 0; i < basis.size(); i++)
	{
		if (basis[i].lhs.is_subset_of(X) && (!basis[i].rhs.is_subset_of(X)))
			return false;
	}

	return true;
}

boost::dynamic_bitset<unsigned long> getRandomSubsetBS(boost::dynamic_bitset<unsigned long> &st)
{
	int numElems = st.size(), processedElems = 0;
	boost::dynamic_bitset<unsigned long> ansSet(numElems);

	while (processedElems < numElems)
	{
		int bset = rand();

		for (int i = 0; i < 30; i++)
		{
			if ((bset & (1 << i)) && (st[processedElems]))
			{
				ansSet[processedElems] = true;
			}

			processedElems++;

			if (processedElems >= numElems)
				break;
		}
	}

	return ansSet;
}

boost::dynamic_bitset<unsigned long> getFrequentAttrSetBS()
{
	// cout <<"1\n";
	if (counterexampleType == 1)
		return getRandomSubsetBS(objInpBS[discreteDistribution(re)]);
	if (counterexampleType == 2)
	{
		// cout <<"2--\n";
		int objId = discreteDistributionArea(re);
		int objSize = objInp[objId].size();
		vector<int> object = objInp[objId];
		int setSize = discreteDistributionAttributeSets[objSize](re);
		vector<int> indices(objSize);

		for (int i = 0; i < objSize; i++)
			indices[i] = i;

		shuffle(indices.begin(), indices.end(), re);
		boost::dynamic_bitset<unsigned long> ans(attrInp.size());

		for (int i = 0; i < setSize; i++)
		{
			ans[object[indices[i]]] = true;
		}

		return ans;
	}

	if (counterexampleType == 3)
	{
		// cout <<"3--\n";
		long long intersectionId = discreteDistributionSquared(re);
		long long set1 = intersectionId / (long long)objInp.size();
		long long set2 = intersectionId % (long long)objInp.size();
		boost::dynamic_bitset<unsigned long> tempSet = objInpBS[set1] & objInpBS[set2];
		return getRandomSubsetBS(tempSet);
	}

	if (counterexampleType == 4)
	{
		long long intersectionId = discreteDistributionDiscriminativity(re);
		long long set1 = intersectionId / (long long)objInp.size();
		long long set2 = intersectionId % (long long)objInp.size();
		boost::dynamic_bitset<unsigned long> tempSet1 = objInpBS[set1] - objInpBS[set2],
											 tempSet2 = objInpBS[set1] & objInpBS[set2];
		boost::dynamic_bitset<unsigned long> setF = getRandomSubsetBS(tempSet1),
											 setFp = getRandomSubsetBS(tempSet2);
		return (setF | setFp);
	}

	if (counterexampleType == 5)
	{
		return getRandomSubsetBS(objInpBS[binomialDistribution(re)]);
	}
}

boost::dynamic_bitset<unsigned long> getRandomAttrSetBS()
{
	boost::dynamic_bitset<unsigned long> ans(attrInp.size());
	ans.set();
	ans[0] = false;
	return getRandomSubsetBS(ans);
}

void getCounterExample(vector<implicationBS> &basis, int s)
{
	double threadContextClosureTime = 0, threadImplicationClosureTime = 0;
	std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
	int threadTries = 0;
	boost::dynamic_bitset<unsigned long> X;

	for (int i = s; i < maxTries && globalFlag; i += numThreads)
	{ // Each thread handles an equal number of iterations.
		threadTries++;

		if (frequentCounterExamples)
			X = getFrequentAttrSetBS();
		else
			X = getRandomAttrSetBS();

		auto start = chrono::high_resolution_clock::now();
		boost::dynamic_bitset<unsigned long> cX = contextClosureBS(X);
		auto end = chrono::high_resolution_clock::now();
		threadContextClosureTime += (chrono::duration_cast<chrono::microseconds>(end - start)).count();

		start = chrono::high_resolution_clock::now();
		boost::dynamic_bitset<unsigned long> cL = closureBS(basis, X);
		end = chrono::high_resolution_clock::now();
		threadImplicationClosureTime += (chrono::duration_cast<chrono::microseconds>(end - start)).count();

		if (epsilonStrong)
		{
			if (!cL.is_subset_of(cX))
			{
				lck.lock();
				globalFlag = false;
				counterExampleBS = cX;
				isPositiveCounterExample = true;
				lck.unlock();
				break;
			}

			if (!cX.is_subset_of(cL))
			{
				lck.lock();
				globalFlag = false;
				counterExampleBS = cL;
				isPositiveCounterExample = false;
				lck.unlock();
				break;
			}
		}

		else
		{
			if (X.count() == cX.count())
			{
				if (!isSetEqualToImpCLosure(basis, X))
				{
					lck.lock();
					globalFlag = false;
					counterExampleBS = X;
					isPositiveCounterExample = true;
					lck.unlock();
					break;
				}
			}
			else
			{
				if (isSetEqualToImpCLosure(basis, X))
				{
					lck.lock();
					globalFlag = false;
					counterExampleBS = X;
					isPositiveCounterExample = false;
					lck.unlock();
					break;
				}
			}
		}
	}

	lck.lock();

	totTries += threadTries;

	if (threadContextClosureTime > thisIterMaxContextClosureTime)
		thisIterMaxContextClosureTime = threadContextClosureTime;

	if (threadImplicationClosureTime > thisIterMaxImplicationClosureTime)
		thisIterMaxImplicationClosureTime = threadImplicationClosureTime;

	lck.unlock();
}

void tryPotentialCounterExamples(vector<implicationBS> &basis)
{
	while (!potentialCounterExamplesBS.empty())
	{
		boost::dynamic_bitset<unsigned long> X = potentialCounterExamplesBS.back();
		//cout <<"Trying a Potential Counter Example: ";
		//printVector(X);
		potentialCounterExamplesBS.pop_back();
		boost::dynamic_bitset<unsigned long> cX = contextClosureBS(X);
		if (X.count() == cX.count())
			continue;
		boost::dynamic_bitset<unsigned long> cL = closureBS(basis, X);

		if (epsilonStrong)
		{
			if (cL.count() != cX.count())
			{
				//cout <<"It is a Counter Example!!\n";
				counterExampleBS = cL;
				globalFlag = false;
				return;
			}
		}

		else
		{
			if (cL.count() == X.count())
			{
				//cout <<"It is a Counter Example!!\n";
				counterExampleBS = cL;
				globalFlag = false;
				return;
			}
		}
	}
}

void tryToUpdateImplicationBasis(vector<implicationBS> &basis, int threadIndex)
{
	std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
	double threadContextClosureTime = 0;
	lck.lock();
	if (isPositiveCounterExample)
	{
		while (implicationsSeen < basis.size())
		{

			UpdateImplicationTries++;
			int currIndex = implicationsSeen;
			implicationsSeen++;
			boost::dynamic_bitset<unsigned long> A = basis[currIndex].lhs;
			boost::dynamic_bitset<unsigned long> B = basis[currIndex].rhs;
			boost::dynamic_bitset<unsigned long> newB = B & counterExampleBS;
			lck.unlock();
			if (A.is_subset_of(counterExampleBS) && !B.is_subset_of(counterExampleBS))
			{
				lck.lock();
				updatedImplications.push_back({currIndex, implicationBS({A, B & counterExampleBS})});
				vector<int> vectorA = attrBSToAttrVector(A);
				vector<int> vectorRHS = attrBSToAttrVector(newB);
				continue;
			}
			lck.lock();
		}
	}
	else
	{
		while ((implicationsSeen < basis.size()) && (!basisUpdate))
		{
			UpdateImplicationTries++;
			boost::dynamic_bitset<unsigned long> A = basis[implicationsSeen].lhs;
			boost::dynamic_bitset<unsigned long> B = basis[implicationsSeen].rhs;
			int curIndex = implicationsSeen;
			implicationsSeen++;
			boost::dynamic_bitset<unsigned long> C = A & counterExampleBS;
			lck.unlock();
			aEqualToCCount++;

			if (A != C)
			{
				aEqualToCCount--;
				auto durBegin = chrono::high_resolution_clock::now();
				// boost::dynamic_bitset<unsigned long> cC = contextClosureBS(C);
				boost::dynamic_bitset<unsigned long> cC = randomContextClosureBS(C, percentAttrClosure, threadIndex);
				auto durEnd = chrono::high_resolution_clock::now();
				threadContextClosureTime += (chrono::duration_cast<chrono::microseconds>(durEnd - durBegin)).count();

				if (C == cC)
				{
					lck.lock();
					continue;
				}

				lck.lock();

				if (!basisUpdate)
				{
					basisUpdate = true;
					indexOfUpdatedImplication = curIndex;
					updatedImplication.lhs = C;
					updatedImplication.rhs = B;
					vector<int> newLHS = attrBSToAttrVector(C), newRHS = attrBSToAttrVector(B);
				}

				else if (basisUpdate && (curIndex < indexOfUpdatedImplication))
				{
					indexOfUpdatedImplication = curIndex;
					updatedImplication.lhs = C;
					updatedImplication.rhs = B;
					vector<int> newLHS = attrBSToAttrVector(C), newRHS = attrBSToAttrVector(B);
				}

				continue;
			}

			lck.lock();
		}
	}

	if (threadContextClosureTime > thisIterMaxContextClosureTime)
		thisIterMaxContextClosureTime = threadContextClosureTime;

	lck.unlock();
}

vector<implication> BSBasisToVectorBasis(vector<implicationBS> ansBS)
{
	vector<implication> ans;

	for (int i = 0; i < ansBS.size(); i++)
	{
		ans.push_back(implication{attrBSToAttrVector(ansBS[i].lhs), attrBSToAttrVector(ansBS[i].rhs)});
	}

	return ans;
}

int findPremiseSupportOfParticularImplication(vector<implicationBS> &basisBS, int i)
{
	int support_premsis = 0;
	for (int j = 0; j < objInpBS.size(); j++)
	{

		if (basisBS[i].lhs.is_subset_of(objInpBS[j]))
		{
			support_premsis++;
		}
	}
	return support_premsis;
}

int findImplicationSupportOfParticularImplication(vector<implicationBS> &basisBS, int i)
{
	int support_premsis = 0;
	int support_implication = 0;
	for (int j = 0; j < objInpBS.size(); j++)
	{
		if (basisBS[i].lhs.is_subset_of(objInpBS[j]) && basisBS[i].rhs.is_subset_of(objInpBS[j]))
		{
			support_implication++;
		}
	}
	return support_implication;
}

double FindConfidenceOfParticularImplication(vector<implicationBS> &basisBS, int i)
{

	int support_premsis = 0;
	int support_implication = 0;
	for (int j = 0; j < objInpBS.size(); j++)
	{

		if (basisBS[i].lhs.is_subset_of(objInpBS[j]))
		{
			support_premsis++;
		}

		if (basisBS[i].lhs.is_subset_of(objInpBS[j]) && basisBS[i].rhs.is_subset_of(objInpBS[j]))
		{
			support_implication++;
		}
	}

	return ((double)support_implication / support_premsis);
}

double calculatePrecision(vector<implicationBS> &basisBS)
{

	// int count = 0;

	// 	for (int i = 0; i < topKRulesBS.size(); i++)
	// 	{

	// 		for (int j = 0; j < basisBS.size(); j++)
	// 		{
	// 			cout << "topklhs: "<<topKRulesBS[i].lhs << ' ' << "topkrhs: "<<topKRulesBS[i].rhs << endl;
	// 			cout<<"basislhs: "<<basisBS[j].lhs<<' '<<"basisrhs: "<<basisBS[j].rhs<<endl;

	// 			if (topKRulesBS[i].lhs == basisBS[j].lhs && topKRulesBS[i].rhs == basisBS[j].rhs)
	// 			{
	// 				// cout << topKRulesBS[i].lhs << ' ' << topKRulesBS[i].rhs << endl;
	// 				// cout<<basisBS[j].lhs<<' 'basisBS[j].rhs<<endl;
	// 				count++;
	// 				break;
	// 			}
	// 		}
	// 	}
	// 	return ((double)count) / basisBS.size();

	long long result = 0;

	for (int i = 0; i < basisBS.size(); i++)
	{
		boost::dynamic_bitset<unsigned long> lhsCl =
			closureBS(topKRulesBS, basisBS[i].lhs);

		if ((basisBS[i].rhs).is_subset_of(lhsCl))
			result++;
	}

	return result / ((double)basisBS.size());
}

double calculatePrecisionFilter(vector<implicationBS> &baBS, double minconf)
{
	long long result = 0;
	vector<implicationBS> basisBS;
	for (int i = 0; i < baBS.size(); i++)
	{
		if (FindConfidenceOfParticularImplication(baBS, i) >= minconf_value)
		{
			basisBS.push_back(baBS[i]);
		}
	}

	for (int i = 0; i < basisBS.size(); i++)
	{

		boost::dynamic_bitset<unsigned long> lhsCl =
			closureBS(topKRulesBS, basisBS[i].lhs);

		if ((basisBS[i].rhs).is_subset_of(lhsCl))
			result++;
	}

	return result / ((double)basisBS.size());
}

double calculateRecall(vector<implicationBS> &basisBS)
{

	long long result = 0;

	for (int i = 0; i < topKRulesBS.size(); i++)
	{

		boost::dynamic_bitset<unsigned long> lhsCl =
			closureBS(basisBS, topKRulesBS[i].lhs);

		if ((topKRulesBS[i].rhs).is_subset_of(lhsCl))
			result++;
	}

	return result / ((double)topKRulesBS.size());
}

std::vector<std::pair<boost::dynamic_bitset<unsigned long>, float>> PremWiseRecall;
double calculateRecallFilter(vector<implicationBS> &baBS, double minconf)
{
	PremWiseRecall.clear();

	long long result = 0;
	std::vector<implicationBS> basisBS;
	for (int i = 0; i < baBS.size(); i++)
	{
		if (FindConfidenceOfParticularImplication(baBS, i) >= minconf_value)
		{
			basisBS.push_back(baBS[i]);
		}
	}

	//finding the percent of rules being followed premise wise
	int siz = topKRulesBS.size();
	std::vector<bool> visited(siz, 0);
	int countOnes = 0;
	int uniqPrem = 0;

	while (countOnes != siz)
	{
		uniqPrem++;
		boost::dynamic_bitset<unsigned long> currPremise;
		bool premFound = 0;
		int count = 0;
		int countF = 0;
		for (int i = 0; i < siz; i++)
		{
			if (!premFound)
			{
				if (visited[i] == 0)
				{
					currPremise = topKRulesBS[i].lhs;
					premFound = 1;
					i--;
				}
			}
			else
			{

				if (topKRulesBS[i].lhs == currPremise)
				{
					count++;
					boost::dynamic_bitset<unsigned long> lhsCl = closureBS(basisBS, topKRulesBS[i].lhs);

					if ((topKRulesBS[i].rhs).is_subset_of(lhsCl))
					{
						result++;
						countF++;
					}
					visited[i] = 1;
					countOnes++;
				}
			}
		}

		// auto premiseVector = attrBSToAttrVector(currPremise);
		// cout<<"Premise: ";printVector(premiseVector); cout << "\n";
		// cout << "#rules:" << count<<' '<<countF<<endl;
		PremWiseRecall.push_back(make_pair(currPremise, (float)countF * 100 / count));
	}

	return result / ((double)topKRulesBS.size());
}

void verboseImplicationOutput(vector<implicationBS> &ansBS, double timestamp)
{
	cout << "\n";
	countClosedPremises = 0;

	for (int i = 0; i < ansBS.size(); i++)
	{
		vector<int> impl_lhs = attrBSToAttrVector(ansBS[i].lhs), impl_rhs = attrBSToAttrVector(ansBS[i].rhs);
		printVector(impl_lhs);
		cout << " ==> ";
		printVector(impl_rhs);
		double con = FindConfidenceOfParticularImplication(ansBS, i);
		int supp_impl = findImplicationSupportOfParticularImplication(ansBS, i),
			supp_prem = findPremiseSupportOfParticularImplication(ansBS, i);
		boost::dynamic_bitset<unsigned long> cP = contextClosureBS(ansBS[i].lhs);
		bool isPremiseClosed = (ansBS[i].lhs == cP);
		countClosedPremises += isPremiseClosed;
		cout << " #SUP_IMPL: " << supp_impl << " #Supp_Prem: " << supp_prem << " #CONF: " << con << " #CLOSED: " << isPremiseClosed;
		if (con >= minconf_value)
		{
			cout << " #Y";
		}
		cout << "\n";
	}
	auto precision = calculatePrecisionFilter(ansBS, minconf_value);
	auto recall = calculateRecallFilter(ansBS, minconf_value);

	cout << "\nPremise wise recall:\n";
	for (int i = 0; i < PremWiseRecall.size(); i++)
	{
		vector<int> vec = attrBSToAttrVector(PremWiseRecall[i].first);
		printVector(vec);
		cout << " percent: " << PremWiseRecall[i].second << endl;
	}
	cout << "\nBasisSize " << ansBS.size() << "  Timestamp " << TIMEPRINT(timestamp) << " "
		 << "Precision " << precision << " Recall " << recall << " Closed Count: " << countClosedPremises << " Max Updates: " << maxImplicationUpdates << "\n";
	cout << "\n";
}

void setNumThreads()
{
	double temp = (prevThreads * prevIterTime) / threadOverheadTime;
	temp -= (prevThreads * prevThreads);

	if (temp < 0)
	{
		numThreads = 1;
		return;
	}

	temp = sqrt(temp);
	numThreads = max((int)temp, 1);
	numThreads = min((int)numThreads, maxThreads);
	// cout << maxThreads <<","<< numThreads << endl ;
}

vector<implication> generateImplicationBasis(ThreadPool &threadPool)
{
	vector<implicationBS> ansBS;
	double prevIterTime1 = 0, prevIterTime2 = 0;
	double prevThreads1 = 1, prevThreads2 = 1;
	double iterationIOTime = 0;

	int count_rules = 0;
	while (true)
	{
		auto start = chrono::high_resolution_clock::now();
		gCounter++;
		totTries = 0;
		iterationIOTime = 0;
		// cout << "Going to get counter example. (Iteration Number: " << gCounter << " )" << endl;
		getLoopCount();
		// cout << "Max number of tries for this iteration: " << maxTries << "\n";
		globalFlag = true;
		counterExampleBS.clear();
		thisIterMaxContextClosureTime = 0;
		thisIterMaxImplicationClosureTime = 0;

		// if (!potentialCounterExamplesBS.empty())
		// {
		// 	tryPotentialCounterExamples(ansBS);

		// 	if(!globalFlag)
		// 		singletonCounterexamples++;

		// 	gCounter = 0;
		// }

		if (globalFlag)
		{
			prevThreads = prevThreads1;
			prevIterTime = prevIterTime1;
			setNumThreads();
			vector<std::future<void>> taskVector;

			for (int i = 1; i < numThreads; i++)
			{
				taskVector.emplace_back(threadPool.enqueue(getCounterExample, ref(ansBS), i));
			}
			//
			//This is important. If we don't write the next statement,
			//the main thread will simply keep waiting without doing anything.
			//This initially caused quite a bit of confusion, as a program without multi-threading was running faster
			//due to the main thread sitting idle.
			//
			getCounterExample(ansBS, 0);

			for (int i = 0; i < taskVector.size(); i++)
			{
				taskVector[i].get();
			}
		}

		updownTime += thisIterMaxContextClosureTime;
		totalClosureTime += thisIterMaxImplicationClosureTime;

		if (globalFlag && bothCounterExamples)
		{
			bothCounterExamples = false;
			frequentCounterExamples = false;
			gCounter = max(0, gCounter - 1);
			continue;
		}

		sumTotTries += totTries;
		if (globalFlag)
			break;

		boost::dynamic_bitset<unsigned long> X = counterExampleBS;
		//printVector(X);

		totCounterExamples++;
		// cout << "Counterexample found after " << totTries << " tries\n";
		vector<int> counterExampleFound = attrBSToAttrVector(X);
		if (isPositiveCounterExample)
		{
			countPositiveCounterExample++;
			//	cout << "Got Positive CS  "; printVector(counterExampleFound); cout << "\n";
		}
		else
		{
			countNegativeCounterExample++;
			//	cout << "Got Negative CS  "; printVector(counterExampleFound); cout << "\n";
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
		prevThreads1 = numThreads;
		prevIterTime1 = duration.count();
		// cout << duration.count() << ",";
		totalTime += duration.count();
		bool found = false;
		start = chrono::high_resolution_clock::now();
		basisUpdate = false;
		implicationsSeen = 0;
		thisIterMaxContextClosureTime = 0;

		//The algorithm implemented as-is.
		prevThreads = prevThreads2;
		prevIterTime = prevIterTime2;
		setNumThreads();

		vector<std::future<void>> taskVector;
		updatedImplications.clear();
		UpdateImplicationTries = 0;

		for (int i = 1; i < numThreads; i++)
			taskVector.emplace_back(threadPool.enqueue(tryToUpdateImplicationBasis, ref(ansBS), i));

		tryToUpdateImplicationBasis(ansBS, 0);

		for (int i = 0; i < taskVector.size(); i++)
		{
			taskVector[i].get();
		}

		updownTime += thisIterMaxContextClosureTime;
		// cout << UpdateImplicationTries << " iterations in tryToUpdateImplicationBasis\n";

		if (gCounter % 5000 == 0)
		{
			print_count = 0;
			debug = true;
		}

		if (debug)
		{
			auto debug_start = std::chrono::high_resolution_clock::now();
			print_count++;
			vector<int> vectorX = attrBSToAttrVector(X);

			cout << "\nIteration " << gCounter << "\n";
			cout << "CS: ";
			printVector(vectorX);
			cout << "Pos: " << isPositiveCounterExample << " Tries: " << totTries << "\n";

			if (isPositiveCounterExample)
			{
				cout << "Update " << updatedImplications.size() << " Implications\n";
				for (auto &updateImp : updatedImplications)
				{
					vector<int> initial_lhs = attrBSToAttrVector(ansBS[updateImp.first].lhs),
								initial_rhs = attrBSToAttrVector(ansBS[updateImp.first].rhs),
								new_lhs = attrBSToAttrVector(updateImp.second.lhs),
								new_rhs = attrBSToAttrVector(updateImp.second.rhs);
					cout << "\nIndex " << updateImp.first << "\n";
					cout << "Old ";
					printVector(initial_lhs);
					cout << " =>  ";
					printVector(initial_rhs);
					cout << "\n";

					if (updateImp.second.lhs == updateImp.second.rhs)
					{
						cout << "Delete\n";
						continue;
					}

					cout << "New ";
					printVector(new_lhs);
					cout << " =>  ";
					printVector(new_rhs);
					cout << "\n";
				}
			}
			else
			{
				if (!basisUpdate)
				{
					boost::dynamic_bitset<unsigned long> allattribute(attrInp.size());
					allattribute.set();
					allattribute[0] = false;
					vector<int> vectorM = attrBSToAttrVector(allattribute);

					cout << "\nIndex " << ansBS.size() << "\n";
					cout << "Add ";
					printVector(vectorX);
					cout << " =>  ";
					printVector(vectorM);
					cout << "\n";
				}
				else
				{
					vector<int> initialLHS = attrBSToAttrVector(ansBS[indexOfUpdatedImplication].lhs),
								initialRHS = attrBSToAttrVector(ansBS[indexOfUpdatedImplication].rhs),
								newLHS = attrBSToAttrVector(updatedImplication.lhs),
								newRHS = attrBSToAttrVector(updatedImplication.rhs);

					cout << "\nIndex " << indexOfUpdatedImplication << "\n";
					cout << "Old ";
					printVector(initialLHS);
					cout << " => ";
					printVector(initialRHS);
					cout << "\n";
					cout << "New ";
					printVector(newLHS);
					cout << " => ";
					printVector(newRHS);
					cout << "\n";
				}
			}

			verboseImplicationOutput(ansBS, (chrono::duration_cast<chrono::microseconds>(debug_start- startTime)).count() - ioTime - iterationIOTime);

			if (print_count == 20)
			{
				debug = false;
			}

			auto debug_end = std::chrono::high_resolution_clock::now();
			auto debug_duration = chrono::duration_cast<chrono::microseconds>(debug_end - debug_start).count();
			iterationIOTime += debug_duration;
		}

		if (isPositiveCounterExample)
		{
			sort(updatedImplications.begin(), updatedImplications.end(), 
				[](pair<int, implicationBS> &a, pair<int, implicationBS> &b)
				 { return a.first > b.first; });

			for (auto &updateImp : updatedImplications)
			{
				if (updateImp.second.lhs == updateImp.second.rhs)
				{
					if (FindConfidenceOfParticularImplication(ansBS, updateImp.first) >= minconf_value)
						count_rules--;
					ansBS[updateImp.first] = ansBS.back();
					ansBS.pop_back();
					continue;
				}

				bool isgrt = false;
				if (FindConfidenceOfParticularImplication(ansBS, updateImp.first) >= minconf_value)
				{
					isgrt = true;
				}
				ansBS[updateImp.first] = updateImp.second;

				int ToBeAdded = 0;
				if (FindConfidenceOfParticularImplication(ansBS, updateImp.first) >= minconf_value)
				{
					ToBeAdded = 1;
				}
				if (!isgrt)
				{
					count_rules += ToBeAdded;
				}
				else
				{
					if (ToBeAdded == 0)
					{
						if (isgrt)
						{
							// cout<<"WARNING1!!!\n";
							// cout<<"Old implication: ";printVector(initial_lhs);cout<<" ==> ";printVector(initial_rhs);cout<<endl;
							// cout<<"New implication: ";printVector(new_lhs);cout<<" ==> ";printVector(new_rhs);
							count_rules -= 1;
						}
					}
				}
			}
			maxImplicationUpdates = max(maxImplicationUpdates, (int)updatedImplications.size());
		}
		else
		{
			if (!basisUpdate)
			{
				// ansBS.push_back(implicationBS{X, contextClosureBS(X)});
				boost::dynamic_bitset<unsigned long> allattribute(attrInp.size());
				allattribute.set();
				allattribute[0] = false;
				ansBS.push_back(implicationBS{X, allattribute});
				if (FindConfidenceOfParticularImplication(ansBS, ansBS.size() - 1) >= minconf_value)
				{
					count_rules++;
				}
			}
			else
			{

				bool isgrt = false;
				double confBefore = FindConfidenceOfParticularImplication(ansBS, indexOfUpdatedImplication);
				if (confBefore >= minconf_value)
				{
					isgrt = true;
				}
				int supp_impl_before = findImplicationSupportOfParticularImplication(ansBS, indexOfUpdatedImplication),
					supp_prem_before = findPremiseSupportOfParticularImplication(ansBS, indexOfUpdatedImplication);
				ansBS[indexOfUpdatedImplication] = updatedImplication;

				int ToBeAdded = 0;
				double confAfter = FindConfidenceOfParticularImplication(ansBS, indexOfUpdatedImplication);
				if (confAfter >= minconf_value)
				{
					ToBeAdded = 1;
				}
				if (!isgrt)
				{
					count_rules += ToBeAdded;
				}
				else
				{
					if (ToBeAdded == 0)
					{
						if (isgrt)
						{
							// cout<<"WARNING2!!!\n";
							// cout<<"Old implication: ";printVector(initialLHS);cout<<" ==> ";printVector(initialRHS);cout<< " #SUP_IMPL: " << supp_impl_before << " #SUP_PREM: " << supp_prem_before << " #CONF: "<<confBefore;cout<<endl;
							// cout<<"New implication: ";printVector(newLHS);cout<<" ==> ";printVector(newRHS);cout << " #SUP_IMPL: " << findImplicationSupportOfParticularImplication(ansBS, indexOfUpdatedImplication) << " #SUP_PREM: " << findPremiseSupportOfParticularImplication(ansBS, indexOfUpdatedImplication) <<"#CONF: "<<confAfter;
							// cout<<endl;
							count_rules -= 1;
						}
					}
				}
			}
		}

		if (!topK_times.empty() && topK_times[0] < 0)
		{
			//K stop

			if (count_rules >= k_value)
			{
				auto time_difference = (chrono::duration_cast<chrono::microseconds>((chrono::high_resolution_clock::now() - startTime))).count() - ioTime - iterationIOTime;
				notFoundKRules = 0;
				auto ioStart = chrono::high_resolution_clock::now();
				verboseImplicationOutput(ansBS, time_difference);
				auto ioEnd = chrono::high_resolution_clock::now();
				iterationIOTime += (chrono::duration_cast<chrono::microseconds>(ioEnd - ioStart)).count();
				break;
			}
		}
		if (!topK_times.empty() && topK_times[0] >= 0)
		{
			// early stop
			if (timePointer >= topK_times.size())
			{
				break;
			}
			else
			{
				countClosedPremises = 0;
				auto time_difference = (chrono::duration_cast<chrono::microseconds>((chrono::high_resolution_clock::now() - startTime))).count() - ioTime - iterationIOTime;
				if (time_difference >= topK_times[timePointer] * 1000000)
				{
					auto ioStart = chrono::high_resolution_clock::now();
					verboseImplicationOutput(ansBS, time_difference);
					auto ioEnd = chrono::high_resolution_clock::now();
					iterationIOTime += (chrono::duration_cast<chrono::microseconds>(ioEnd - ioStart)).count();
					timePointer++;
				}
			}
		}

		end = std::chrono::high_resolution_clock::now();
		totalExecTime2 += (chrono::duration_cast<chrono::microseconds>(end - start)).count() - ioTime;
		duration = chrono::duration_cast<chrono::microseconds>(end - start);
		prevThreads2 = numThreads;
		prevIterTime2 = duration.count() - iterationIOTime;
		ioTime += iterationIOTime;

		// cout << duration.count() << "\n";
	}

	if (notFoundKRules)
	{
		auto time_difference = (chrono::duration_cast<chrono::microseconds>((chrono::high_resolution_clock::now() - startTime))).count() - ioTime;
		auto ioStart = chrono::high_resolution_clock::now();
		verboseImplicationOutput(ansBS, time_difference);
		auto ioEnd = chrono::high_resolution_clock::now();
		ioTime += (chrono::duration_cast<chrono::microseconds>(ioEnd - ioStart)).count();
	}

	ansBasisBS = ansBS;
	minconfRulesCount = count_rules;
	return BSBasisToVectorBasis(ansBS);
}

vector<double> confidenceOfImplicationBasis;

vector<int> supp_imp;
vector<int> supp_prem;

void FindConfidenceOfImplications()
{

	for (int i = 0; i < ansBasisBS.size(); i++)
	{
		int support_premsis = 0;
		int support_implication = 0;
		for (int j = 0; j < objInpBS.size(); j++)
		{

			if (ansBasisBS[i].lhs.is_subset_of(objInpBS[j]))
			{
				support_premsis++;
			}

			if (ansBasisBS[i].lhs.is_subset_of(objInpBS[j]) && ansBasisBS[i].rhs.is_subset_of(objInpBS[j]))
			{
				support_implication++;
			}
		}
		supp_imp.push_back(support_implication);
		supp_prem.push_back(support_premsis);
		confidenceOfImplicationBasis.push_back((double)support_implication / support_premsis);
	}
}

void printUsageAndExit()
{
	cout << "Usage: ./algo <contextFileFullPath> <epsilon> <delta> <strong/weak> <uniform/frequent/both> <numThreads> <support/none>\n";
	exit(0);
}

void fillPotentialCounterExamples()
{
	// Two attribute sets
	// for(int i = 0; i < attrInp.size(); i++)
	// {
	// 	for(int j = (i + 1); j < attrInp.size(); j++)
	// 	{
	// 		potentialCounterExamples.push_back({i, j});
	// 	}
	// }

	// Singleton
	for (int i = 1; i < attrInp.size(); i++)
	{
		vector<int> cVec = {i};
		potentialCounterExamplesBS.push_back(attrVectorToAttrBS(cVec));
	}
}

void initializeObjInpBS()
{
	objInpBS.resize(objInp.size());

	for (int i = 0; i < objInp.size(); i++)
	{
		objInpBS[i] = attrVectorToAttrBS(objInp[i]);
	}
}

bool isLectGreater(boost::dynamic_bitset<unsigned long> &closedSet, int lectInd)
{
	for (int i = 0; i <= lectInd; i++)
		if (closedSet[frequencyOrderedAttributes[i]])
			return true;

	return false;
}

boost::dynamic_bitset<unsigned long> nextContextClosure(boost::dynamic_bitset<unsigned long> A, boost::dynamic_bitset<unsigned long> finalClosedSet)
{
	int nAttr = attrInp.size() - 1;

	for (int i = nAttr; i > 0; i--)
	{
		if (A[frequencyOrderedAttributes[i]])
			A[frequencyOrderedAttributes[i]] = false;
		else
		{
			boost::dynamic_bitset<unsigned long> B, temp = A;
			temp[frequencyOrderedAttributes[i]] = true;
			B = contextClosureBS(temp);

			bool flag = true;

			for (int j = 1; j < i; j++)
			{
				if (B[frequencyOrderedAttributes[j]] & (!A[frequencyOrderedAttributes[j]]))
				{
					flag = false;
					break;
				}
			}

			if (flag)
				return B;
		}
	}

	return finalClosedSet;
}

int allContextClosures()
{
	int totalClosedSets = 1;
	boost::dynamic_bitset<unsigned long> currentClosedSet, finalClosedSet(attrInp.size()), emptySet(attrInp.size());
	currentClosedSet = contextClosureBS(emptySet);
	finalClosedSet.set();
	finalClosedSet[0] = false;
	int nattr = attrInp.size();
	int lectInd = max(1, ((3 * nattr) / 4)), lectLessClosures;
	bool lectDone = false;
	auto timeStart = chrono::high_resolution_clock::now();
	auto timePrev = chrono::high_resolution_clock::now();

	while (currentClosedSet != finalClosedSet)
	{
		currentClosedSet = nextContextClosure(currentClosedSet, finalClosedSet);
		totalClosedSets++;
		auto timeNow = chrono::high_resolution_clock::now();
		double duration = (chrono::duration_cast<chrono::microseconds>(timeNow - timePrev)).count();

		if (duration > 60000000)
		{
			// cout <<"Total Context closures till now: "<< totalClosedSets << endl;
			timePrev = timeNow;
		}

		if ((!lectDone) && isLectGreater(currentClosedSet, lectInd))
		{
			lectLessClosures = totalClosedSets;
			lectDone = true;
		}

		duration = (chrono::duration_cast<chrono::microseconds>(timeNow - timeStart)).count();

		if (lectDone && (duration > 6000000))
		{
			// cout <<"Lectically less Context Closures:"<< lectLessClosures << endl;
			return lectLessClosures;
		}
	}

	// cout <<"Lectically less Context Closures:"<< lectLessClosures << endl;
	return lectLessClosures;
}

boost::dynamic_bitset<unsigned long> nextImplicationClosure(boost::dynamic_bitset<unsigned long> A, boost::dynamic_bitset<unsigned long> finalClosedSet)
{
	int nAttr = attrInp.size() - 1;

	for (int i = nAttr; i > 0; i--)
	{
		if (A[frequencyOrderedAttributes[i]])
			A[frequencyOrderedAttributes[i]] = false;
		else
		{
			boost::dynamic_bitset<unsigned long> B, temp = A;
			temp[frequencyOrderedAttributes[i]] = true;
			B = closureBS(ansBasisBS, temp);

			bool flag = true;

			for (int j = 1; j < i; j++)
			{
				if (B[frequencyOrderedAttributes[j]] & (!A[frequencyOrderedAttributes[j]]))
				{
					flag = false;
					break;
				}
			}

			if (flag)
				return B;
		}
	}

	return finalClosedSet;
}

int allImplicationClosures()
{
	int totalClosedSets = 1;
	boost::dynamic_bitset<unsigned long> currentClosedSet, finalClosedSet(attrInp.size()), emptySet(attrInp.size());
	currentClosedSet = closureBS(ansBasisBS, emptySet);
	finalClosedSet.set();
	finalClosedSet[0] = false;

	int nattr = attrInp.size();
	int lectInd = max(1, ((3 * nattr) / 4)), lectLessClosures;
	bool lectDone = false;
	auto timeStart = chrono::high_resolution_clock::now();
	auto timePrev = chrono::high_resolution_clock::now();

	while (currentClosedSet != finalClosedSet)
	{
		currentClosedSet = nextImplicationClosure(currentClosedSet, finalClosedSet);
		totalClosedSets++;

		auto timeNow = chrono::high_resolution_clock::now();
		double duration = (chrono::duration_cast<chrono::microseconds>(timeNow - timePrev)).count();

		if (duration > 60000000)
		{
			// cout <<"Total Implication closures till now: "<< totalClosedSets << endl;
			timePrev = timeNow;
		}

		if ((!lectDone) && isLectGreater(currentClosedSet, lectInd))
		{
			lectLessClosures = totalClosedSets;
			lectDone = true;
		}

		duration = (chrono::duration_cast<chrono::microseconds>(timeNow - timeStart)).count();

		if (lectDone && (duration > 6000000))
		{
			// cout <<"Lectically less Implication Closures:"<< lectLessClosures << endl;
			return lectLessClosures;
		}
	}

	// cout <<"Lectically less Implication Closures:"<< lectLessClosures << endl;
	return lectLessClosures;
}

void getSupportOfImplicationsFrequent()
{
	vector<long long> supports;

	for (int i = 0; i < ansBasisBS.size(); i++)
	{
		long long support = 0;

		for (int j = 0; j < objInpBS.size(); j++)
		{
			if (ansBasisBS[i].lhs.is_subset_of(objInpBS[j]))
				support++;
		}

		supports.push_back(support);
	}

	sort(supports.rbegin(), supports.rend());
	double meanSupport = accumulate(supports.begin(), supports.end(), (double)0);
	meanSupport /= supports.size();
	double p10, p50, p90, p95;
	p10 = supports[0.1 * supports.size()];
	p50 = supports[0.5 * supports.size()];
	p90 = supports[0.9 * supports.size()];
	p95 = supports[0.95 * supports.size()];
	cout << 100 * meanSupport / objInpBS.size() << ";";
	cout << 100 * p10 / objInpBS.size() << ";";
	cout << 100 * p50 / objInpBS.size() << ";";
	cout << 100 * p90 / objInpBS.size() << ";";
	cout << 100 * p95 / objInpBS.size() << ";";
	return;
}

void getSupportOfImplicationsArea()
{
	vector<long long> supports;

	for (int i = 0; i < ansBasisBS.size(); i++)
	{
		long long support = 0;

		for (int j = 0; j < objInpBS.size(); j++)
		{
			if (ansBasisBS[i].lhs.is_subset_of(objInpBS[j]))
				support++;
		}

		supports.push_back(support * ansBasisBS[i].lhs.count());
	}

	sort(supports.rbegin(), supports.rend());
	double meanSupport = accumulate(supports.begin(), supports.end(), (double)0);
	meanSupport /= supports.size();
	double p10, p50, p90, p95;
	p10 = supports[0.1 * supports.size()];
	p50 = supports[0.5 * supports.size()];
	p90 = supports[0.9 * supports.size()];
	p95 = supports[0.95 * supports.size()];
	cout << meanSupport << ";";
	cout << p10 << ";";
	cout << p50 << ";";
	cout << p90 << ";";
	cout << p95 << ";";
	return;
}

void getSupportOfImplicationsSquared()
{
	vector<long long> supports;

	for (int i = 0; i < ansBasisBS.size(); i++)
	{
		long long support = 0;

		for (int j = 0; j < objInpBS.size(); j++)
		{
			if (ansBasisBS[i].lhs.is_subset_of(objInpBS[j]))
				support++;
		}

		supports.push_back(support * support);
	}

	sort(supports.rbegin(), supports.rend());
	double meanSupport = accumulate(supports.begin(), supports.end(), (double)0);
	meanSupport /= supports.size();
	double p10, p50, p90, p95;
	p10 = supports[0.1 * supports.size()];
	p50 = supports[0.5 * supports.size()];
	p90 = supports[0.9 * supports.size()];
	p95 = supports[0.95 * supports.size()];
	double numObjSq = objInpBS.size();
	numObjSq *= numObjSq;
	cout << 100 * meanSupport / numObjSq << ";";
	cout << 100 * p10 / numObjSq << ";";
	cout << 100 * p50 / numObjSq << ";";
	cout << 100 * p90 / numObjSq << ";";
	cout << 100 * p95 / numObjSq << ";";
	return;
}

void initFrequencyOrderedAttributes()
{
	vector<int> freqAttr(attrInp.size(), 0);

	for (int i = 0; i < objInp.size(); i++)
	{
		for (int j = 0; j < objInp[i].size(); j++)
			freqAttr[objInp[i][j]]++;
	}

	vector<pair<int, int>> freqPairs;

	for (int i = 1; i < attrInp.size(); i++)
	{
		freqPairs.push_back({freqAttr[i], i});
	}

	sort(freqPairs.begin(), freqPairs.end());
	frequencyOrderedAttributes.push_back(0);

	for (int i = 0; i < freqPairs.size(); i++)
		frequencyOrderedAttributes.push_back(freqPairs[i].second);
}

int NoOFExactRules = 0;
int NoOfRulesConfHighThanPoint9 = 0;

void CountExactRules()
{

	for (int i = 0; i < confidenceOfImplicationBasis.size(); i++)
	{
		if (confidenceOfImplicationBasis[i] == 1)
		{
			NoOFExactRules++;
			NoOfRulesConfHighThanPoint9++;
		}
		else if (confidenceOfImplicationBasis[i] > 0.9)
		{
			NoOfRulesConfHighThanPoint9++;
		}
	}
}

void getTopKRules(string filename)
{
	ifstream File(filename);

	string line;
	vector<implication> curImpVector;
	topKRulesBS.clear();

	while (getline(File, line))
	{
		stringstream ss(line);
		string word;
		implication curImp; // lhs, rhs

		while (ss >> word)
		{
			if (word[0] == '=')
				break;

			curImp.lhs.push_back(stoi(word));
		}

		while (ss >> word)
		{
			if (word[0] == '#')
				break;

			curImp.rhs.push_back(stoi(word));
		}
		topKRulesBS.push_back(
			implicationBS(
				{attrVectorToAttrBS(curImp.lhs), attrVectorToAttrBS(curImp.rhs)}));
	}
	File.close();
}

void get_kvalue_minconf(string filename)
{
	int startcoll = 0, count_us = 0;
	string kval = "";
	for (int i = 0; i < filename.length(); i++)
	{
		if (startcoll && filename[i] != '_')
		{
			kval += filename[i];
			if (filename[i + 1] == '_')
			{
				break;
			}
		}
		else if (count_us == 1 && filename[i] == '_')
		{
			startcoll = 1;
		}
		else if (filename[i] == '_')
		{
			count_us++;
			// startcoll=1;
		}
	}

	int len_file = filename.length();
	string minconf = "";
	for (int i = len_file - 7; i < len_file; i++)
	{
		minconf += filename[i];
	}
	if (minconf[0] == '_')
	{
		minconf = minconf.substr(1, 2);
	}
	k_value = stoi(kval);
	minconf_value = (double)stoi(minconf) / 100;
}

using namespace std;

int main(int argc, char **argv)
{
	string filename = argv[9];
	get_kvalue_minconf(filename);

	startTime = chrono::high_resolution_clock::now();
	srand(time(NULL));

	if (argc < 10)
	{
		printUsageAndExit();
	}

	readFormalContext1(argv[1]);
	initializeObjInpBS();
	initFrequencyOrderedAttributes();

	for (int i = 10; i < argc; i++)
	{
		topK_times.push_back(stoi(argv[i]));
	}

	epsilon = atof(argv[2]);
	del = atof(argv[3]);
	percentAttrClosure = atof(argv[4]);
	if (percentAttrClosure > 1.0)
		percentAttrClosure /= 100;

	getTopKRules(filename);

	if (string(argv[5]) == string("strong"))
		epsilonStrong = true;

	if (string(argv[6]) != string("uniform"))
	{
		frequentCounterExamples = true;
		string temp = argv[6];

		if (temp == string("area"))
			counterexampleType = 2;
		if (temp == string("squared"))
			counterexampleType = 3;
		if (temp == string("discriminativity"))
		{
			counterexampleType = 4;
			readLabels(argv[9]);
		}
		if (temp == "binomial")
			counterexampleType = 5;
	}

	if (string(argv[6]) == string("both"))
		bothCounterExamples = true;

	maxThreads = atoi(argv[7]);
	numThreads = 1;
	if (string(argv[8]) == string("support"))
		implicationSupport = true;

	ThreadPool threadPool(maxThreads - 1);
	fillPotentialCounterExamples();
	initializeRandSetGen();
	vector<implication> ans = generateImplicationBasis(threadPool);
	endTime = chrono::high_resolution_clock::now();
	double TotalExecTime = 0;
	TotalExecTime += (chrono::duration_cast<chrono::microseconds>(endTime - startTime)).count();

	if (implicationSupport)
	{
		getSupportOfImplicationsFrequent();
		getSupportOfImplicationsArea();
		getSupportOfImplicationsSquared();
	}

	// if (topK_times.empty())
	// {
	// 	countClosedPremises = 0;
	// 	for (int i = 0; i < ansBasisBS.size(); i++)
	// 	{
	// 		vector<int> impl_lhs = attrBSToAttrVector(ansBasisBS[i].lhs);
	// 		boost::dynamic_bitset<unsigned long> cP = contextClosureBS(ansBasisBS[i].lhs);
	// 		bool isPremiseClosed = (ansBasisBS[i].lhs == cP);
	// 		countClosedPremises += isPremiseClosed;
	// 		printVector(impl_lhs);
	// 		cout << "==> ";
	// 		vector<int> impl_rhs = attrBSToAttrVector(ansBasisBS[i].rhs);
	// 		printVector(impl_rhs);
	// 		cout << " #SUP_IMPL: " << findImplicationSupportOfParticularImplication(ansBasisBS, i) << " #Supp_Prem: " << findPremiseSupportOfParticularImplication(ansBasisBS, i) << "#CONF: " << FindConfidenceOfParticularImplication(ansBasisBS, i) << " #Closed: " << isPremiseClosed;
	// 		cout << "\n";
	// 	}
	// }

	cout << "TimeTaken: " << TIMEPRINT(TotalExecTime - ioTime) << " Basis Size: "<< ans.size() << " Minconf Rules: " << minconfRulesCount << " Closed Count: " << countClosedPremises << "\n";
	// ExecutionTime, #iteration, #implications, #TotalCounterEx, #positiveCounterEx, #negativeCounterEx, #exactRules, #highConfidence, qualityFactor, QFtime
	for (int i = 1; i < 10; i++)
		cout << argv[i] << ",";
	cout << TIMEPRINT(TotalExecTime - ioTime) << ",";
	cout << gCounter << ",";
	cout << ans.size() << ",";
	cout << totCounterExamples << ",";
	cout << countPositiveCounterExample << ",";
	cout << countNegativeCounterExample << ",";
	cout << NoOFExactRules << ",";
	cout << NoOfRulesConfHighThanPoint9 << ",";
	cout << maxImplicationUpdates << ",";
	cout << "\n\n";

	return 0;
}
