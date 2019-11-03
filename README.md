# Graduation Rate Predictions
Fred Berendse

![img/Graduation.jpg](img/Graduation.jpg)

## Motivation
Institutions of higher education stand as beacons of progressive thought and social change. Yet, our society still struggles to realize equity in the courtroom, in the workplace, and in the classroom. Education is touted as the great equalizer that opens doors for the oppressed. If this is true, then equity in education must be a top priority for creating positive social change. It is necessary to examine how our institutions of education, from pre-K to post-secondary, are facilitating (or preventing) the path toward this ideal. 

The motivating goal of this capstone project is to determine whether one can accurately predict an institution’s graduation rate for minority/low SES students based on institutional features. Also, which of these features is most influential on graduation rate?

## The IPEDS Dataset
The National Center for Education Statistics, part of the US Department of Education, has a publicly available central repository for postsecondary education data: the [Integrated Postsecondary Education Data System (IPEDS)](https://nces.ed.gov/ipeds/). Data contained in the data system are aggregate data reported by over 7000 US postsecondary institutions reported annually in three reporting periods. 

At the time of this capstone project, the data in Table 1 were publicly available from the IPEDS database. 

![ipeds-tables.png](img/ipeds-tables.png)*Tables in the IPEDS dataset*

The data from these tables can be downloaded in CSV for all collection years listed. Beginning with the 2006-07 collection year, annual collections were also compiled into an MS Access database. Unfortunately, data are not available in other database formats (e.g. Oracle, Postgres, and MongoDB). For this reason, I created the [IPEDS python library](https://github.com/Fred-B-Berendse/ipeds_library) to read in and merge tables from this database. 

One challenge presented by these tables is that several of them are many-to-one tables _i.e._ multiple rows refer to a single institution. Merging these into a single one-to-one table for modeling results in a very wide table. Therefore, feature selection will become a very important aspect of this project.

![a many-to-one dataset](img/gr2017-head-raw.png)

*An example of a many-to-one table in the IPEDS dataset. This is a snippet of the first few columns of the GR2017 table.*

The following tables from the 2016-2017 reporting year were used in this analysis. A description of the columns extracted from each table can be found [here](columns.md). 

| Table | Rows | Full Rows | Description |
|--------|------------|-----------|-------------|
| HD2017 | 7153 | 7153 | Institutional name, location, and descriptive categories | 
| ADM2017 | 2075 | 920 | Application, admission and enrollment data including SAT/ACT percentiles |
| GR2017 | 54714 | 49981 | Number of students who completed a bachelor's degree within 100% (2013 cohort), 125% (2012 cohort), and 150% (2011 cohort) of normal time disaggregated by gender and race/ethnicity |
| GR2017_PELL_SSL | 9116 | 5557 | Number of students who completed a bachelor's degree within 150% (2011 cohort) of normal time, disaggregated by Pell Grant recipients, Subsidized Stafford Loan recipients, and non-recipients |
| SFA2017 | 6394 | 0 | Number of students paying in-state/out-of-state tuition and receive grant/scholarship aid, disaggregated by aid type and income level |

*The column "Full Rows" indicates the number of rows that do not contain any NaN values.*

## Data Cleaning

The raw data were cleaned using methods from the [IPEDS pipeline library](https://github.com/Fred-B-Berendse/ipeds_library). This library allows one to build a collection of tables with its `IpedsCollection` class. The collection contains methods to keep only a subset of columns, purge unwanted imputation values, map numerical categories to strings, filter rows by category values, flatten many-to-one tables, and finally merge those tables to a single table.

![IPEDS pipeline flowchart](img/ipeds-pipeline.png)
*The IPEDS cleaning pipeline*

The first attempt to clean the dataset raised three key issues addressed below.

### The Admissions Table
 The admissions table (ADM2017) only has 939 full rows after the keep columns/purge imputations steps. A snapshot of ten random rows from the raw table (shown below) reveals two reasons why: 1) some institutions did not report part-time enrollment totals, and 2) many institutions did not report both SAT and ACT quartiles.

![adm-random-rows.png](img/adm-random-rows.png)
*Ten random rows from the ADM2017 table. Many institutions did not report SAT/ACT quartiles. Some did not report part-time enrollment totals.*

Blank part-time enrollment totals can easily be fixed by imputing values by subtracting full-time enrollment from the total enrollment count.

Below are the number of institutions that reported SAT/ACT quartile data:

| Reported | Count |
|----------|-------:|
| Neither SAT nor ACT | 753 |
| SAT only | 109 |
| ACT only | 83 |
| Both SAT and ACT | 1130 |
|||
| Total | 2075 |
||| 

 The second problem, however, is trickier to solve since many post-secondary institutions don't require SAT/ACT tests. To resolve this, each benchmark was normalized to a mean of 0 and standard deviation of 1. The mean of the two standardized test benchmark was calculated as a new feature in the dataset.

### The Graduation Table
The second key issue is that the graduation rate table (GR2017) has only 753 full rows after it is flattened to a one-to-one table. 

 Each student within the bachelor's degree cohort is assigned a cohort status (*e.g* completed in 4 years, completed in 5 years, completed in 6 years, transfered out, not completed - still enrolled, or not completed - no longer enrolled). A small institution that did not have any students transfer out would have no row for that cohort status in the table. When the table is flattened to a one-to-one table, that status is entered as NaN for that institution. To fix this issue, values of 0 are imputed for NaN values after the table is flattened. 

If an instiution has no members of a particular race/ethnicity within a cohort, calculating a percentage will result in a division by zero. To fix this issue, Laplace smoothing is applied when calculating the percentage in such a way that the graduation percentage for all races will replace a potential division by zero.

### The Graduation Pell/SSL Table
The Graduation Pell/Subsidized Stafford Loan (SSL) table contains categories (Pell Grant recipient, SSL without Pell recipient, non-recipient) with a zero count. This would return NaN values as a graduation rate for that category. To fix this issue, an overall graduation rate was calculated from sums over all categories. Laplace smoothing with a pseudocount of 0.01 times the sum over all three categories was used. 

### The Student Financial Aid Table
Finally, the student financial aid column (SFA2017) has __no__ full rows in any of the steps of the pipeline process. 

This was addressed by doing three things:

* Dropping columns that contain counts of students paying in-district, in-state, and out-of-state tuition fees.
* Dropping rows that only contain information about percentages of undergraduates awarded financial aid
* Imputing a value of 0 for rows containing no value for the number of students living on campus. This was done after verifying that the counts of students living off campus sum to equal the number of students.


### Final pipeline
Once these three key issues were resolved, the pipeline produces the following number of rows (full rows) for each stable during key steps in the process.

| Table | Raw | Keep Columns / Purge Imputations | Filtering Categories | Flattening Tables | Imputing Values |
|-------|-----|----------|-----------|---------------|------------|
| HD2017            | 7153 (7153)   | 7153 (7153)   | 7153 (7153)   | 7153 (7153) | 7153 (7153) |
| ADM2017           | 2075 (920)    | 2075 (1130)    | 2075 (1130)    | 2075 (1130)  | 2075 (1322) |
| GR2017            | 54714 (49981) | 54714 (49981) | 4203 (4203) | 2139 (2064) | 2139 (2123) |
| GR2017_PELL_SSL   | 9116 (5557)   | 9116 (5557)   | 2139 (2139)   | 2139 (2139) | 2139 (2139) |
| SFA2017           | 6394 (0)      | 6394 (1184)   | 6394 (1184)   | 6394 (1184) | 4029 (4029) |
|||||||
*Finalized pipeline. The number in parentheses is the number of full (non-NaN) rows.*

The final step in the pipeline is to merge all of the tables by an institution's unitid then drop rows containing any NaN values. The merged table contains 682 institutions available for modeling. 

## Exploratory Data Analysis

### Graduation Rates 

There are 307244 students in the 2011 cohort across all institutions in the dataset. Of those, 209811 students received a bachelor's degree within 150% of normal time (*i.e.* 6 years). The bar graph below shows the racial breakdown of these students. A bar graph of racial distribution from the U.S. Census (2018 estimate) is shown for comparison.

![img/completions-bar.png](img/completion-bar.png)

*Number of bachelor's degree completions across all institutions in the dataset. U.S. Census (2018 estimate): https://www.census.gov/quickfacts/fact/table/US/IPE120218*

Because there is such a small number of American Indian/Native Alaskan and Native Hawaiian/Pacific Islander students in the dataset, it was decided to exclude these races from further analysis.

Distributions of graduation rates among all institutions in the dataset show that students receiving need-based financial aid tend to have lower graduation rates than their white peers.

![img/completion-race.png](img/completion-race.png)

A similar disparity exists between students receiving federal need-based aid compared to their more affluent peers.

![img/completion-pell-ssl.png](img/completion-pell-ssl.png)

Correlation plots of graduation rates between different races reveals a surprising amount of collinearity: institutions that have high white graduation rates also tend to have higher minority graduation rates.

![img/correlation-races.png](img/correlation-races.png)

Likewise, institutions that graduate a high percentage of non-recipients also graduate a higher percentage of financial aid recipients.

![img/correlation-ps.png](img/correlation-ps.png)

### Principal Component Analysis
A principal component analysis was performed on the table of institutions to determine if there is some combination of features (called a principal component) which can separate high graduation rates from low graduation rates. The first principal component accounts for 30.9% of the total variance of the data in feature space. Subsequent axes account for 10.2% or less per axis.

![img/pca-scree.png](img/pca-scree.png)
*Principal Component Analysis (PCA) variance plot*

Below is a series of scatterplot maps of each institution along the two most important principal components. Each institution is labeled with its graduation rate. One can see from these maps that high graduation rates tend to be separated from low rates on the map. These results demonstrate that supervised learning models are likely able to predict graduation rates to some degree.

|  |  |
|:-------------------------:|:-------------------------:|
| ![img/pca-white.png](img/pca-white.png) |  ![img/pca-black.png](img/pca-black.png) |
| ![img/pca-hispanic.png](img/pca-hispanic.png) |  ![img/pca-asian.png](img/pca-asian.png) |
 ![img/pca-2plus.png](img/pca-2plus.png) |
 ||| 
 *Principal Component Analysis (PCA) maps of graduation rates by race*


|  |  |
|:-------------------------:|:-------------------------:|
| ![img/pca-pell.png](img/pca-pell.png) |  ![img/pca-ssl.png](img/pca-ssl.png) |
| ![img/pca-nonrecipient.png](img/pca-nonrecipient.png) ||
|||
*Principal Component Analysis (PCA) maps of graduation rates by Pell/SSL status*


When graduation rates of different races or grant/loan status are used as labels, one generally sees the same trend. It may be possible that the Laplace smoothing is causing this. To test this, graduation rates without smoothing were also mapped onto component space. For most races and grant/loan status, this did not change the general trend. However, two groups, 2+ races and Asians, showed a much more homogeneous mixutre of high and low graduation rates when smoothing was removed. 

| With Laplace Smoothing | Without Laplace Smoothing |
|:-------------------------:|:-------------------------:|
| ![img/pca-white.png](img/pca-white.png) |  ![img/pca-white-nosm.png](img/pca-white-nosm.png) |
| ![img/pca-2plus.png](img/pca-2plus.png) |  ![img/pca-2plus-nosm.png](img/pca-2plus-nosm.png)|
| ![img/pca-asian.png](img/pca-asian.png) |  ![img/pca-asian-nosm.png](img/pca-asian-nosm.png)|
|||
*Principal Component Analysis (PCA) maps with and without Laplace smoothing*

It makes sense that these two minority categories would be affected by smoothing since they have smaller counts than other race categories. 

The principal component analysis class in `sklearn` utilizes singular value decomposition (SVD) to determine the principal component axes. SVD decomposes a matrix of institutions and their features into a multiplication of three separate matrices. The last of these matrices contains values that indicate how each feature loads onto each principal component. By looking at the values in the first row of this vector, we can get a general idea of which features are most important along the first principal component. Below is a table of the five features that load most onto the first principal component axis:

* average math 25th percentile
* average math 75th percentile
* average english 25th percentile
* average english 75th percentile
* percent of undergraduates awarded a Pell Grant

## Modeling

### Ordinary Linear Regression

The first two models fitted are linear regression models. One of the strengths of a regression model is that its coefficients are interpretable if the following conditions are met:

* __Linearity__: violated when nonlinear trends exist in residuals
* __Independence__: violated if a point depends on the value of another point in the data set
* __Homoscedasticity__: violated when the variance of residuals isn't constant 
* __Normality__: violated when the residuals are not normally distributed
* __Multicollinearity__: violated when one feature is highly correlated with others

Two of these assumptions can be checked before running the model: independence of data points and multicollinearity. We will assume that independence holds. It seems to be reasonable given that no student attends more than one institution and each institution can set its own policy decisions even if it is part of a multi-campus system. 

Multicollinearity can be checked by calculating a variance inflation factor for each of the features in the model. The feature with the largest variance inflation factor was eliminated and the process repeated until no remaining features had a variance feature factor above five. The surviving features after this recursive elimination are in the table below.

| Feature | Description | Variance Inflation Factor |
|---------|-------------|---------------------------|
| latitude | latitude of institution | 1.09 |
| longitud | longitude of institution | 1.14 |
| enrlft_pct | percent of enrolled students attending full time | 1.20 |
| enrlt_pct | percent of accepted students enrolled | 1.29 |
| admssn_pct | percent of applicants accepted | 1.42 |
| en25 | average of normalized ACT English/SAT Verbal 25th percentile | 2.81 |
| uagrntp | percent of students awarded financial aid (any source) | 1.84 |
| upgrntp | percent of students awarded Pell Grants | 3.17 |
| grntwf2_pct | percent of students living with family off campus 2016-17 | 1.44 |
| grntof2_pct | percent of students living off campus (not with family) 2016-17  | 1.32 |

Several of the percentage features are very different than a normal distribution. Percentages bunched near zero (enrlt_pct and grntwf2_pct), these were transformed by the function `log10(percent + 1)`. Percentages bunched near 100 (enrlft_pct, uagrntp, and grntof2_pct) were transformed by the function `log10(101-percent)`. 

![img/linreg-features-histograms.png](img/linreg-features-histograms.png)
*Histograms of features used in linear regression modeling.*

A linear regression was performed on each race and Pell Grant/SSL status. All features and targets were standardized to a mean of zero and standard deviation of 1 before fitting. The data were split into a 25% test- 75% train. 

Below are the residual sum of squares and the root mean squared error for each of the regressions.

| Target | Train R<sup>2</sup> | Test R<sup>2</sup> | Test RMSE |
|--------|------|----------|---------|
| 2 or More Races | 0.56 | 0.44 | 15.4 |
| Asian | 0.49 | 0.41 | 17.2 |
| Black | 0.59 | 0.52 | 15.4 |
| Hispanic | 0.52 | 0.38 | 16.6 |
| White | 0.70 | 0.60 | 10.9 |
| Pell Grant | 0.67 | 0.55 | 12.5 |
| Stafford Loan (SSL) | 0.57 | 0.46 | 14.1 |
| Non-Recipient | 0.54 | 0.39 | 15.4 |
|||||

The residuals of each regression all have very normal-like distributions, indicating that the normality condition of a linear regression has at least been loosely met.

![img/linreg-residuals-histograms.png](img/linreg-residuals-histograms.png)

Scale-location (or spread-location) plots are [an accepted method to test for homoscedacity](https://boostedml.com/2019/03/linear-regression-plots-scale-location-plot.html). These graphs show the square root of the absolute value of standardized residuals vs the predicted value. If a scale-location graph shows a horizontal trend with equal spread along the entire graph, then both the homoscedacity condition of linear regression is met.

The scale-location graphs shown loosely meet these criteria, although there is some extra spread for low fitted values as well as a hint that the residual decreases for higher values.

![img/linreg-spread-location.png](img/linreg-spread-location.png)

The coefficients of the regression using normalized features and targets reveals that the most influential features are the same as those identified by PCA analysis: the SAT Verbal/ACT English 25th percentile benchmark and the percentage of students who received a Pell Grant. A third influential feature came out of this analysis: latitude. Apparently, universities located farther north tend to have higher graduation rates for all races and financial aid statuses. 

If the coefficients of this model are reliable, then the model also indicates that living off campus with family members has a larger negative effect on black students than students in other categories. The model also shows that while the percentage of students who received a Pell Grant is negatively correlated with graduation rates for all categories, this correlation is stronger for white students than those of other races.  

![img/linreg-norm-coeff-heatmap.png](img/linreg-norm-coeff-heatmap.png)
*Linear regression coefficients for normalized targets and features.*

![img/linreg-coeff-heatmap.png](img/linreg-coeff-heatmap.png)
*Linear regression coefficients scaled to features.*

### Lasso Linear Regression

In an attempt to limit the number of features, normalized targets and features were fitted with sklearn's `LassoLarsCV` model, which utilizes cross validation to optimize the shrinkage hyperparameter $\alpha$. For this model, five-fold validation was used with a maximum number of iterations of 500. Because the targets were pre-normalized, the option to fit the intercept was turned off. 

When the model was run on the training set, only 10 iterations were necessary to find that the best value of $\alpha$ was around 0.002, which indicates that the model is doing very little to punish large coefficient values. Below are the best-fit values of $\alpha$, as well as R<sup>2</sup> and RMSE for each target. Residuals and coefficients for each model are very similar to the ordinary regression model.

| Target | Train R<sup>2</sup> | Test R<sup>2</sup> | Test RMSE | Best $\alpha$ |
|--------|------|----------|---------|-----|
| 2 or More Races | 0.56 | 0.44 | 26.2 | 0.001 |
| Asian | 0.49 | 0.41 | 27.8 | 0.002 |
| Black | 0.59 | 0.52 | 28.1 | 0.002 |
| Hispanic | 0.52 | 0.38 | 27.4 | 0.002 |
| White | 0.70 | 0.60 | 22.5 | 0.001 |
| Pell Grant | 0.67 | 0.55 | 24.6 | 0.001 |
| Stafford Loan (SSL) | 0.57 | 0.46 | 23.5 | 0.001 |
| Non-Recipient | 0.54 | 0.39 | 26.2 | 0.002 |
||||||

![img/lasso-residuals-histograms.png](img/lasso-residuals-histograms.png)*Residuals of Lasso regularized regression to normalized targets and features.*

![img/lasso-spread-location.png](img/lasso-spread-location.png)*Spread-location graph of residuals: Lasso regularized regression to normalized targets and features.*

![img/lasso-norm-coeff-heatmap.png](img/lasso-norm-coeff-heatmap.png)*Coefficients of Lasso regularized regression to normalized targets and features.*

![img/lasso-coeff-heatmap.png](img/lasso-coeff-heatmap.png)*Coefficients of Lasso regularized regression when scaled to feature dimensions*

### Random Forest Regression

A random forest regression was the next model utilized to predict graduation rates. A baseline model was first fitted to get a general idea of the performance of random forest model compared to the regression models. Once a baseline performance was established, sklearn's `RandomizedSearchCV` module was employed to do cross validation to search for the best combination of hyperparameters. The table below shows the range of values made available for the search. The number of iterations was set so that approximately 25% of the entire gridspace was searched. 

| Hyperparameter | Baseline Model | Search Range | Best Model |
|-----------|----------|------------|--------------|
| number of trees | 100 | 100 - 1000 | 600 |
| criterion for split | mean-squared error (MSE) | MSE, mean-absolute error (MAE) | MAE |
| min samples for split | 2 | 2, 5, 10, 20 | 5 |
| min samples per leaf | 1 | 1, 2, 5 | 2 | 
| max features per split | sqrt(n_features) | n_features, sqrt(n_features) | sqrt(n_features) |
||||

Both the baseline and best random forest models yield values of R<sup>2</sup> comparable to regression models. The random search for hyperparameters using cross validation did not significantly improve the model's performance. Interestingly, however, the root mean squared errors are lower for random forest than for the linear regression models. 

| Target | R<sup>2</sup> | RMSE |
|--------|----------------|------------|
| Two or More Races | 0.44 (0.42) | 15.30 (15.72) |
| Asian | 0.38 (0.38) | 17.26 (17.31) |
| Black | 0.56 (0.55) | 14.92 (14.86) |
| Hispanic | 0.36 (0.32) | 16.65 (17.23) |
| White | 0.63 (0.62) | 10.52 (10.70) |
| Pell Grant | 0.57 (0.55) | 12.33 (12.23) |
| SSL | 0.48 (0.47) | 13.84 (14.21) |
| Non-Recipient | 0.38 (0.37) | 15.34 (15.70) |
||||
*Performance metrics of the best random forest model selected by CV. Metrics of the baseline model are in parentheses.*

The best random forest model shows that the two most important features are the average normalized SAT/ACT English 25th percentile benchmark and the percentage of students who received a Pell Grant - consistent with the PCA analysis and regression models. The third most important feature identified by this model is the percent of applicants admitted to the institution, not the latitude of the institution as was identified by the regression models. 
These rankings were consistent across all target demographics.

|  |  |
|:-------------------------:|:-------------------------:|
| ![img/rf-feat-imp-2plus.png](img/rf-feat-imp-2plus.png) |  ![img/rf-feat-imp-asian.png](img/rf-feat-imp-asian.png) |
| ![img/rf-feat-imp-black.png](img/rf-feat-imp-black.png) |  ![img/rf-feat-imp-hisp.png](img/rf-feat-imp-hisp.png) |
| ![img/rf-feat-imp-white.png](img/rf-feat-imp-white.png) |  |
|||
*Feature importance plots for all racial/ethnic demographics.*

|  |  |
|:-------------------------:|:-------------------------:|
| ![img/rf-feat-imp-pell.png](img/rf-feat-imp-pell.png) |  ![img/rf-feat-imp-ssl.png](img/rf-feat-imp-ssl.png) |
| ![img/rf-feat-imp-nonrec.png](img/rf-feat-imp-nonrec.png) |  |
|||
*Feature importance plots for Pell Grant, SSL, and non-recipient demographics.*

Feature importances do not determine the direction of influence a given feature has upon graduation rates. The influence of each of the top four features is shown. All demographics are impacted very similarly by English 25th percentile benchmark and percentage of students receiving a Pell Grant. There is some variation in the impact of admission percentage between demographics, but this has less than a 5% impact on graduation rate. 

![img/rf-part-dep-races.png](img/rf-part-dep-races.png)
*Partial dependence plots of the top four features for all racial/ethnic groups.*

![img/rf-part-dep-pgs.png](img/rf-part-dep-pgs.png)
*Partial dependence plots of the top four features for Pell Grant, SSL, and non-recipient groups.*

## Conclusions and Further Work
Institutional features of 682 institutions in the IPEDS database were modeled with ordinary least-squares linear regression, a Lasso regularized linear regression model, and a random forest model. All of the models have an R<sup>2</sup> value of 0.4-0.6 on a test holdout data set.

All three models agree that the two most influential predictors of graduation rates are SAT/ACT benchmark scores and the percentage of students who receive a Pell Grant. The former has a positive correlation with graduation rate while the latter has a negative correlation. All three models also show that the graduation rate of one race/ethnicity at a given institution is highly correlated with graduation rates of other races/ethnicities. A similar high correlation exists between Pell Grant/Subsidized Student Loan status groups. 

There is an abundance of data about institutions in the IPEDS database that were not examined in this analysis. It would be interesting to see if adding additional data such as the institutions locale (urban vs rural) or data about the types of degrees granted at a given institution can improve model predictions.

