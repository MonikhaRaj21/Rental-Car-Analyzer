import pandas as pd
import numpy as np
import bisect
import math

from numpy.random import normal, gamma, beta, binomial, dirichlet, multinomial


def readDF(fileloc):

    df = pd.read_csv(fileloc, header=None)
    dfCols = df.iloc[:2, ].T
    df = df.iloc[2:, ]
    levels = [list(set(dfCols.iloc[:, 0])), list(set(dfCols.iloc[:, 1]))]
    nFeat = dfCols.shape[0]

    rowLabels = [[-1 for x in range(nFeat)] for x in range(2)]
    for k in range(2):
        for i, level in enumerate(levels[k]):
            indices = dfCols[dfCols[k] == level].index.tolist()
            for ind in indices:
                rowLabels[k][ind] = i

    midx = pd.MultiIndex(levels=levels, codes=rowLabels)

    df = pd.DataFrame(df.values, columns=midx)
    return df


def create_aug_prod_df(df, CategoricalAttributes):

    categoricalAttr = list(CategoricalAttributes.keys())
    setAttributes = [list(CategoricalAttributes[x]) for x in categoricalAttr]

    numericalAttr = list(set(df.columns) - set(categoricalAttr))
    numInd = sorted([list(df.columns).index(x) for x in numericalAttr])
    numerical = list(np.array(df.columns)[numInd])

    catInd = sorted([list(df.columns).index(x) for x in categoricalAttr])
    categorical = [item for sublist in setAttributes for item in sublist]  # flatten the categorical List
    
  
    topCategorical = [[x + len(numerical) for i in range(len(CategoricalAttributes[categoricalAttr[x]]))]
                      for x in range(len(categoricalAttr))]
    topCategorical = list(range(len(numerical))) + [item for sublist in topCategorical for item in sublist]

    indices = pd.Series(numerical + categorical)
    topIndices = pd.Series(numerical + categoricalAttr)
    #print(df)
    def calculateFeatures(row):

        catList = [list((pd.Series(setAttributes[i]) == row.iat[x]).astype(int)) for i, x in enumerate(catInd)]
        catList = [item for sublist in catList for item in sublist]  # flatten the categorical List
        val = pd.Series(np.array(list(row[numInd]) + catList), index=indices)

        return val

    augmentedDf = df.apply(calculateFeatures, axis=1)
    #print(augmentedDf)
    midx = pd.MultiIndex(levels=[list(topIndices), list(indices)],
                         codes=[topCategorical, list(range(len(topCategorical)))])
    augmentedDf = pd.DataFrame(augmentedDf.values, columns=midx)

    return augmentedDf

"""
    simulate the utility scores across all attributes across all generations
"""
def simulateUtilityScore(N, VehicleShare, NumericalAttributes, CategoricalAttributes):

    mcost_k_M = 100 * beta(a=4.5, b=2, size=N[0])
    mcost_t_M = 0.1 * beta(a=3.5, b=2, size=N[0])
    mcost_utility_M = ((-2e-3 * gamma(mcost_k_M, mcost_t_M, [len(NumericalAttributes['monthly_cost']), N[0]]))
                       * np.array(NumericalAttributes['monthly_cost'])[:, np.newaxis])
    mcost_utility_M -= mcost_utility_M.mean(axis=0)

    mcost_k_X = 100 * beta(a=4., b=2, size=N[1])
    mcost_t_X = 0.1 * beta(a=4.5, b=2, size=N[1])
    mcost_utility_X = ((-1.6e-3 * gamma(mcost_k_X, mcost_t_X, [len(NumericalAttributes['monthly_cost']), N[1]]))
                       * np.array(NumericalAttributes['monthly_cost'])[:, np.newaxis])
    mcost_utility_X -= mcost_utility_X.mean(axis=0)

    mcost_k_B = 100 * beta(a=3.5, b=2, size=N[2])
    mcost_t_B = 0.1 * beta(a=5, b=2, size=N[2])
    mcost_utility_B = ((-1.8e-3 * gamma(mcost_k_B, mcost_t_B, [len(NumericalAttributes['monthly_cost']), N[2]]))
                       * np.array(NumericalAttributes['monthly_cost'])[:, np.newaxis])
    mcost_utility_B -= mcost_utility_B.mean(axis=0)

    mcost_utility = np.hstack([mcost_utility_M, mcost_utility_X, mcost_utility_B]).T

    upcost_k_M = 100 * beta(a=4.5, b=2, size=N[0])
    upcost_t_M = 0.1 * beta(a=3.5, b=2, size=N[0])
    upcost_utility_M = ((-2e-5 * gamma(upcost_k_M, upcost_t_M, [len(NumericalAttributes['upfront_cost']), N[0]]))
                       * np.array(NumericalAttributes['upfront_cost'])[:, np.newaxis])
    upcost_utility_M -= upcost_utility_M.mean(axis=0)

    upcost_k_X = 100 * beta(a=4., b=2, size=N[1])
    upcost_t_X = 0.1 * beta(a=4.5, b=2, size=N[1])
    upcost_utility_X = ((-1e-5 * gamma(upcost_k_X, upcost_t_X, [len(NumericalAttributes['upfront_cost']), N[1]]))
                       * np.array(NumericalAttributes['upfront_cost'])[:, np.newaxis])
    upcost_utility_X -= upcost_utility_X.mean(axis=0)

    upcost_k_B = 100 * beta(a=3.5, b=2, size=N[2])
    upcost_t_B = 0.1 * beta(a=5, b=2, size=N[2])
    upcost_utility_B = ((-1.5e-5 * gamma(upcost_k_B, upcost_t_B, [len(NumericalAttributes['upfront_cost']), N[2]]))
                       * np.array(NumericalAttributes['upfront_cost'])[:, np.newaxis])
    upcost_utility_B -= upcost_utility_B.mean(axis=0)

    upcost_utility = np.hstack([upcost_utility_M, upcost_utility_X, upcost_utility_B]).T

    term_k_M = 100 * beta(a=4.5, b=2, size=N[0])
    term_t_M = 0.1 * beta(a=3.5, b=2, size=N[0])
    term_utility_M = ((-1e-2 * gamma(term_k_M, term_t_M, [len(NumericalAttributes['term']), N[0]]))
                       * np.array(NumericalAttributes['term'])[:, np.newaxis])
    term_utility_M -= term_utility_M.mean(axis=0)

    term_k_X = 100 * beta(a=4., b=2, size=N[1])
    term_t_X = 0.1 * beta(a=4.5, b=2, size=N[1])
    term_utility_X = ((-1.2e-2 * gamma(term_k_X, term_t_X, [len(NumericalAttributes['term']), N[1]]))
                       * np.array(NumericalAttributes['term'])[:, np.newaxis])
    term_utility_X -= term_utility_X.mean(axis=0)

    term_k_B = 100 * beta(a=3.5, b=2, size=N[2])
    term_t_B = 0.1 * beta(a=5, b=2, size=N[2])
    term_utility_B = ((-1.2e-2 * gamma(term_k_B, term_t_B, [len(NumericalAttributes['term']), N[2]]))
                       * np.array(NumericalAttributes['term'])[:, np.newaxis])
    term_utility_B -= term_utility_B.mean(axis=0)

    term_utility = np.hstack([term_utility_M, term_utility_X, term_utility_B]).T


    worth_k_M = 1000 * beta(a=4.5, b=2, size=N[0])
    worth_t_M = 0.01 * beta(a=3.5, b=2, size=N[0])
    worth_utility_M = ((1.5e-5 * gamma(worth_k_M, worth_t_M, [len(NumericalAttributes['vehicle_worth']), N[0]]))
                     * np.array(NumericalAttributes['vehicle_worth'])[:, np.newaxis])
    worth_utility_M -= worth_utility_M.mean(axis=0)

    worth_k_X = 1000 * beta(a=4, b=2, size=N[1])
    worth_t_X = 0.01 * beta(a=4.5, b=2, size=N[1])
    worth_utility_X = ((1.5e-5 * gamma(worth_k_X, worth_t_X, [len(NumericalAttributes['vehicle_worth']), N[1]]))
                       * np.array(NumericalAttributes['vehicle_worth'])[:, np.newaxis])
    worth_utility_X -= worth_utility_X.mean(axis=0)

    worth_k_B = 1000 * beta(a=3.5, b=2, size=N[2])
    worth_t_B = 0.01 * beta(a=4, b=2, size=N[2])
    worth_utility_B = ((1.5e-5 * gamma(worth_k_B, worth_t_B, [len(NumericalAttributes['vehicle_worth']), N[2]]))
                       * np.array(NumericalAttributes['vehicle_worth'])[:, np.newaxis])
    worth_utility_B -= worth_utility_B.mean(axis=0)

    worth_utility = np.hstack([worth_utility_M, worth_utility_X, worth_utility_B]).T

    range_k_M = 1000 * beta(a=4.5, b=2, size=N[0])
    range_t_M = 0.007 * beta(a=4, b=2, size=N[0])
    range_utility_M = ((2e-3 * gamma(range_k_M, range_t_M, [len(NumericalAttributes['range']), N[0]]))
                       * np.array(NumericalAttributes['range'])[:, np.newaxis])
    range_utility_M -= range_utility_M.mean(axis=0)

    range_k_X = 1000 * beta(a=4, b=2, size=N[1])
    range_t_X = 0.008 * beta(a=4, b=2, size=N[1])
    range_utility_X = ((2e-3 * gamma(range_k_X, range_t_X, [len(NumericalAttributes['range']), N[1]]))
                       * np.array(NumericalAttributes['range'])[:, np.newaxis])
    range_utility_X -= range_utility_X.mean(axis=0)

    range_k_B = 1000 * beta(a=3.5, b=2, size=N[2])
    range_t_B = 0.008 * beta(a=4, b=2, size=N[2])
    range_utility_B = ((2e-3 * gamma(range_k_B, range_t_B, [len(NumericalAttributes['range']), N[2]]))
                       * np.array(NumericalAttributes['range'])[:, np.newaxis])
    range_utility_B -= range_utility_B.mean(axis=0)

    range_utility = np.hstack([range_utility_M, range_utility_X, range_utility_B]).T

    charge_k_M = 1000 * beta(a=4.5, b=2, size=N[0])
    charge_t_M = 0.007 * beta(a=4, b=2, size=N[0])
    charge_utility_M = ((2e-3 * gamma(charge_k_M, charge_t_M, [len(NumericalAttributes['charge']), N[0]]))
                       * np.array(NumericalAttributes['charge'])[:, np.newaxis])
    charge_utility_M -= charge_utility_M.mean(axis=0)

    charge_k_X = 1000 * beta(a=4, b=2, size=N[1])
    charge_t_X = 0.008 * beta(a=4, b=2, size=N[1])
    charge_utility_X = ((2e-3 * gamma(charge_k_X, charge_t_X, [len(NumericalAttributes['charge']), N[1]]))
                       * np.array(NumericalAttributes['charge'])[:, np.newaxis])
    charge_utility_X -= charge_utility_X.mean(axis=0)

    charge_k_B = 1000 * beta(a=3.5, b=2, size=N[2])
    charge_t_B = 0.008 * beta(a=4, b=2, size=N[2])
    charge_utility_B = ((2e-3 * gamma(charge_k_B, charge_t_B, [len(NumericalAttributes['charge']), N[2]]))
                       * np.array(NumericalAttributes['charge'])[:, np.newaxis])
    charge_utility_B -= charge_utility_B.mean(axis=0)

    charge_utility = np.hstack([charge_utility_M, charge_utility_X, charge_utility_B]).T

    energy_sig_M = 0.4 * beta(a=10, b=2, size=N[0])
    energy_mu_M = normal(loc=0.9, scale=0.1, size=N[0])
    energy_inter = (1 * normal(energy_mu_M, energy_sig_M, [1, N[0]]))
    energy_utility_M = np.vstack([-1 * energy_inter, energy_inter]).T

    energy_sig_X = 0.4 * beta(a=10, b=2, size=N[1])
    energy_mu_X = normal(loc=0.9, scale=0.1, size=N[1])
    energy_inter = (1 * normal(energy_mu_X, energy_sig_X, [1, N[1]]))
    energy_utility_X = np.vstack([-1 * energy_inter, energy_inter]).T

    energy_sig_B = 0.4 * beta(a=10, b=2, size=N[2])
    energy_mu_B = normal(loc=1.1, scale=0.1, size=N[2])
    energy_inter = (1.2 * normal(energy_mu_B, energy_sig_B, [1, N[2]]))
    energy_utility_B = np.vstack([-1 * energy_inter, energy_inter]).T

    energy_utility = np.vstack([energy_utility_M, energy_utility_X, energy_utility_B])

    type_mix_M = binomial(n=1, p=0.4, size=N[0])
    type_sig_M = 2 * beta(a=10, b=2, size=N[0])
    type_mu_M = normal(loc=(3 * (type_mix_M - 0.5)), scale=0.1, size=N[0])
    type_inter = (0.5 * normal(type_mu_M, type_sig_M, [1, N[0]]))
    type_utility_M = np.vstack([type_inter, -1 * type_inter]).T

    type_mix_X = binomial(n=1, p=0.2, size=N[1])
    type_sig_X = 2 * beta(a=10, b=2, size=N[1])
    type_mu_X = normal(loc=(3 * (type_mix_X - 0.5)), scale=0.1, size=N[1])
    type_inter = (0.5 * normal(type_mu_X, type_sig_X, [1, N[1]]))
    type_utility_X = np.vstack([type_inter, -1 * type_inter]).T

    type_mix_B = binomial(n=1, p=0.4, size=N[2])
    type_sig_B = 2 * beta(a=10, b=2, size=N[2])
    type_mu_B = normal(loc=(3 * (type_mix_B - 0.5)), scale=0.1, size=N[2])
    type_inter = (0.5 * normal(type_mu_B, type_sig_B, [1, N[2]]))
    type_utility_B = np.vstack([type_inter, -1 * type_inter]).T

    type_utility = np.vstack([type_utility_M, type_utility_X, type_utility_B])
    type_utility = type_utility.clip(-5, 5)

    brand_scale = 10 * beta(a=10, b=2, size=sum(N))
    brand_alloc = dirichlet(10*np.array([0.06, 0.07, 0.07, 0.01, 0.09, 0.44, 0.16, 0.1]), size=sum(N))
    brand_utility = brand_alloc * brand_scale[:, np.newaxis]
    brand_utility -= brand_utility.mean(axis=1)[:, np.newaxis]

    utility = np.hstack([brand_utility, mcost_utility, upcost_utility, term_utility, worth_utility, range_utility,
                         charge_utility, energy_utility, type_utility])

    # simulate the current market share
    simCust = multinomial(1, list(VehicleShare.values()), sum(N))
    simProduct = [list(VehicleShare.keys())[x] for x in np.argmax(simCust, axis=1)]

    aug = pd.DataFrame(np.array([list(range(1, sum(N) + 1, 1)), ['Millenial'] * N[0] + ['Gen X'] * N[1]
                                 + ['Baby Boomer'] * N[2], simProduct]).T, columns = ['id', 'segment', 'current brand'])

    k = 3  # we start at three as first three columns reserved for id, segment and current brand
    topLevel = [0, 1, 2]
    topLevelLab = ['id', 'segment', 'current brand']
    bottomLevelLab = ['id', 'segment', 'current brand']
    dictAttributes = {**CategoricalAttributes, **NumericalAttributes}
    for col in ['id', 'brand', 'model', 'monthly_cost', 'upfront_cost', 'term',
                'vehicle_worth', 'range', 'charge', 'energy', 'vehicle_type']:
        if col not in ['id', 'model']:
            topLevel = topLevel + [k] * len(list(dictAttributes[col]))
            topLevelLab = topLevelLab + [col]
            bottomLevelLab = bottomLevelLab + [lvl for lvl in list(dictAttributes[col])]
            k += 1

    midx = pd.MultiIndex(levels=[list(topLevelLab), list(bottomLevelLab)],
                         codes=[topLevel, list(range(len(bottomLevelLab)))])

    utilityDf = pd.DataFrame(pd.concat([aug, pd.DataFrame(utility)], axis=1).values, columns=midx)

    return utilityDf


"""
    Calculte the nRespondent x nProduct matrix of utilities
    : utilityDf: Respondent's recorded utility across various attributes + current brand, segment
    : augmentProductsDf: DataDf with one-hot encoding for categorical variables
    : calculate: boolean whether we should calculate utility or read from file
    : delta:  it shows product index for which utility must be calculated(for any change in data)
    : combine: a previous nRespondent x nProduct matrix of utilities over which we will overwrite the delta column
    :return: i-th respondent's utility on the j-th product
    """
def add_utility(utilityDf, augmentProductsDf, NumericalAttributes, calculate, delta=None, combine=None):
    

    utilityDf = utilityDf.copy()
    numerical = list(NumericalAttributes.keys())

    if calculate:
        productID = augmentProductsDf[('id', 'id')]

        listNumVec = []
        listGridVec = []
        listUtilityMat = []
        for numAttr in numerical:
            listNumVec = listNumVec + [augmentProductsDf[(numAttr, numAttr)].astype(float)]
            listGridVec = listGridVec + [utilityDf[numAttr].columns.astype(int)]
            listUtilityMat = listUtilityMat + [utilityDf[numAttr].values.astype(float)]

        nProduct = len(productID)
        n = utilityDf.shape[0]

        if delta is None:
            delta = range(nProduct)
            combineMat = np.full((n, nProduct), -9.)
        else:
            combineMat = combine.copy()  # in cases where delta has values, we still need to modify relevant columns

        # for each product, find which grid interval each attribute lies in
        matrixIndMapping = np.full([nProduct, len(numerical)], -9)  # matrix which contains
        for i, numAttr in enumerate(numerical):
            for j in range(nProduct):
                matrixIndMapping[j, i] = max(bisect.bisect(listGridVec[i], listNumVec[i][j]) - 1, 0)

        attrTensor = np.full((len(numerical), n, nProduct), -9.)
        for i in range(n):
            for j in delta:

                indices = matrixIndMapping[j, :]
                for k in range(len(numerical)):
                    if math.isnan(listNumVec[k][j]):
                        attrTensor[k, i, j] = 0.
                    else:
                        X0 = listGridVec[k][indices[k]]
                        X1 = listGridVec[k][indices[k] + 1]
                        Y0 = listUtilityMat[k][i, indices[k]]
                        Y1 = listUtilityMat[k][i, indices[k] + 1]
                        attrTensor[k, i, j] = ((Y1 - Y0) / (X1 - X0) * (listNumVec[k][j] - X0)) + Y0

                combineMat[i, j] = sum(attrTensor[:, i, j])
    else:
        combineMat = np.genfromtxt("./Precomputed Data/utilityNumeric.csv", delimiter=',')

    return combineMat

"""
    predict respondent's product based on utility scores with respect to each product
"""
def pred_prod(augmentProductsDf, utilityDf, productsDf, utilityNumeric,
                  delta=None, personProductUtilityMatUse=None, baselineDf=None):

    utilityDf = utilityDf.copy()
    segment = pd.DataFrame(utilityDf.loc[:, 'segment'].values)

    if baselineDf is not None:
        baselineBrand = baselineDf.loc[:, 'Baseline Brand']
        baselineProduct = baselineDf.loc[:, 'Baseline Product']

    ids = utilityDf[("id", "id")].astype(int)

    productID = augmentProductsDf.loc[:, ('id', 'id')]
    brand = productsDf.loc[:, 'brand']
    nProduct = len(productID)
    n = utilityDf.shape[0]
    if delta is None:
        delta = range(nProduct)
        personProductUtilityMat = np.full((n, nProduct), -9.)
    else:
        personProductUtilityMat = personProductUtilityMatUse.copy()

    featuresCat = augmentProductsDf[['brand', 'energy', 'vehicle_type']].values.astype(int)
    brandAttribute = augmentProductsDf.loc[:, 'brand'].values.astype(int)
    utilityCat = utilityDf[['brand', 'energy', 'vehicle_type']].values.astype(float)

    for j in delta:
        if (max(brandAttribute[j, :]) == 1):
            for i in range(n):
                sumOther = utilityNumeric[i, j]
                sumProduct = np.dot(featuresCat[j, :], utilityCat[i, :])
                personProductUtilityMat[i, j] = sumProduct + sumOther
        else:
            for i in range(n):
                personProductUtilityMat[i, j] = 0.

    predictChoice = pd.DataFrame(productID.loc[np.argmax(personProductUtilityMat, axis=1)].values)
    predictBrand = pd.DataFrame(brand.iloc[np.argmax(personProductUtilityMat, axis=1)].values)
    if baselineDf is None:
        baselineBrand = predictBrand.copy()
        baselineProduct = predictChoice.copy()
    
    predictProductDf = pd.concat([ids, segment, baselineBrand, predictBrand, baselineProduct, predictChoice,
                                 pd.DataFrame(personProductUtilityMat)], axis=1)
    #print(predictProductDf)
    lvl_list = ['id', 'segment', 'Brand', 'Product'] + sorted(set(list(brand.astype(str))),
                                                                         key=list(brand.astype(str)).index)
    lvl_list2 = [lvl_list.index(str(x)) for x in productsDf['brand']]

    midx = pd.MultiIndex(levels=[lvl_list, ['id', 'segment', 'Baseline', 'Live'] + list(productID)],
                         codes=[[0, 1, 2, 2, 3, 3] + lvl_list2,
                                 [0, 1, 2, 3, 2, 3] + [4 + x for x in range(len(list(productID)))]])
    predictProductDf = pd.DataFrame(predictProductDf.values, columns=midx)
    
    return predictProductDf, personProductUtilityMat


"""
    Calculate the utility Score
    : N : Market Segmentation
    : WriteUtility : indicates whether to compute utility Score
"""
def precompute(N, writeUtility, folderStr, destFolder):

    # car rentals data
    DataDf = pd.read_csv(folderStr + "RentalCars.csv")
    
    ### Initialzation phase for generating utility score
    VehicleShare = {'Audi': 0.028, 'Chevrolet': 0.070, 'Jaguar': 0.010, 'Kia': 0.019, 'Nissan': 0.049,
                    'Tesla': 0.722, 'Toyota': 0.084, 'VW': 0.018}

    setMonthly = pd.Series([120, 200, 250, 400, 500, 750, 1000, 1250])
    setUpfront = pd.Series([2000, 4000, 6000, 8000, 10000])
    setTerm = pd.Series([24, 36, 48])
    setWorth = pd.Series([20000, 30000, 40000, 80000, 100000])
    setRange = pd.Series([20, 40, 80, 180, 300])
    setCharge = pd.Series([3, 10, 50, 100, 160])
    
    NumericalAttributes = {'monthly_cost': setMonthly, 'upfront_cost': setUpfront, 'term': setTerm,
                         'vehicle_worth': setWorth, 'range': setRange, 'charge': setCharge}

    
    setBrands = pd.Series(['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'])
    setEnergy = pd.Series(['Electric Vehicle', 'Plug-in Hybrid'])
    setType = pd.Series(['Sedan', 'SUV'])
    
    CategoricalAttributes = {'brand': setBrands, 'energy': setEnergy, 'vehicle_type': setType}

    ##one hot encoded data for categorical Attributes alone.
    augmentProductsDf = create_aug_prod_df(DataDf, CategoricalAttributes)

    if writeUtility:
        utilityDf = simulateUtilityScore(N, VehicleShare, NumericalAttributes, CategoricalAttributes)
        utilityDf.to_csv(folderStr + 'UtilityScoresRC.csv', index=False)
        
    else:
        utilityDf = readDF(folderStr + "UtilityScoresRC.csv")

    
    ## nRespondent x nProduct matrix of utilities
    utilityNumeric = add_utility(utilityDf, augmentProductsDf, NumericalAttributes, True, delta=None, combine=None)
    
    
    pd.DataFrame(utilityNumeric).to_csv(destFolder + 'utilityNumericValue.csv', index=False, header=False)

    ## predict respondent's product based on utility scores with respect to each product
    predictProduct, personProdMatch = pred_prod(augmentProductsDf, utilityDf, DataDf, utilityNumeric,
                                                   None, None, None)

    
    predictProduct.to_csv(destFolder + 'predictProduct.csv', index=False)

    baselinePred = predictProduct[[("id", "id"), ("Brand", "Baseline"), ("Product", "Baseline")]]

    baselinePred.columns = ['id', 'Baseline Brand', 'Baseline Product']

    baselinePred.to_csv(destFolder + 'baselinePrediction.csv', index=False)

    pd.DataFrame(personProdMatch).to_csv(destFolder + 'personProdMatch.csv', index=False, header=False)


if __name__ == '__main__':

    folderStr = './Data/'
    destFolder = './Precomputed Data/'
    
    # Market Segmentation
    N = [200, 200, 160]    
    writeUtility = True
    
    precompute(N, writeUtility, folderStr, destFolder)
    
    print("data has been simulated..")