# ChatGPT Q&A - Statistical Analysis for GWAS and Population Genetics 
## Shweta Pipaliya - February 22, 2023
 
### Q1. I want to perform GRM analysis using PLINK input format files. How should I run this using GCTA?

> - Performing a genome-wide relatedness matrix (GRM) analysis using GCTA requires the following steps:
> 
> - Convert your PLINK input files into GCTA format using the plink2gcta command in GCTA. For example:
> 
> ```
> gcta64 --plink2gcta --bfile mydata --make-grm --out mydata
> ```
> - This command will create a GCTA-format file mydata.grm.bin containing the GRM.
> 
> - Use the grm command in GCTA to perform the GRM analysis. For example:
> 
> ```
> gcta64 --grm mydata --grm-cutoff 0.05 --out mydata_grm
> ```
> - This command will calculate the genetic relationship matrix for your data and output the results to the file mydata_grm.
> 
> - In the above example, --grm-cutoff 0.05 sets a cutoff for the minor allele frequency (MAF) at 5% for SNPs included in the GRM calculation. You can adjust this value as needed based on the characteristics of your dataset.
> 
> - Additionally, there are other parameters you may want to use in the grm command, such as --thread-num to specify the number of threads to use for the calculation.

### Q2. How can I use the genetic relationship matrix (grm) to perform a GWAS using mixed linear model in GCTA?

> - You can use the genetic relationship matrix (GRM) obtained from GCTA to perform genome-wide association studies (GWAS) using a mixed linear model (MLM) approach in GCTA. Here are the steps to follow:
> 
> - Prepare your phenotype and genotype data in GCTA format using the --pheno and --bfile options, respectively. For example:
> 
> ```
> gcta64 --bfile mydata --pheno mypheno.txt --make-grm --out mydata
> ```
> 
> - This command will create a GCTA-format file mydata.grm.bin containing the GRM and use the phenotype data in mypheno.txt for analysis.
> 
> - Use the mlma command in GCTA to perform the MLM analysis. For example:
> 
> ```
> gcta64 --mlma --grm mydata --pheno mypheno.txt --out myresult
> ```
> 
> - This command will perform the MLM analysis using the GRM obtained from step 1 and output the results to the file myresult.
> 
> - In the above example, you can also add the --thread-num option to specify the number of threads to use for the calculation.
> 
> - The output file myresult will contain the following columns: SNP name, chromosome, position, effect allele frequency, beta coefficient, standard error, p-value, and significance level after Bonferroni correction.

### Q3. Do I need to perform a principal component analysis and include the resulting PCs as covariates for an mlm GWAS?

> - Performing a principal component analysis (PCA) and including the resulting principal components (PCs) as covariates in a mixed linear model (MLM) GWAS can help to control for population stratification and reduce the risk of false positive associations. However, the need for PCA depends on the characteristics of your dataset.
> 
> - If your dataset is well-matched for population structure, and there is no significant population stratification, you may not need to perform PCA and include PCs as covariates. In such cases, using the GRM obtained from GCTA as a covariate in the MLM analysis may be sufficient to control for population structure.
> 
> - On the other hand, if your dataset has significant population stratification, including the top PCs as covariates in the MLM analysis can help to adjust for this effect and improve the accuracy of the results.
> 
> - In summary, whether or not to perform PCA and include PCs as covariates depends on the characteristics of your dataset, and it is a good practice to assess the population structure and perform appropriate adjustments to control for it.

### Q4. Should I provide a covariate file to run an mlm GWAS using GCTA?

> - Providing a covariate file is optional when running a mixed linear model (MLM) genome-wide association study (GWAS) using GCTA.
> 
> - Including covariates in the MLM model can improve the accuracy of the results by adjusting for any confounding effects in the data. Covariates can include demographic variables, such as age and sex, or other relevant factors that may influence the phenotype of interest.
> 
> - If you have covariate data that you believe may impact your GWAS results, you can provide a covariate file to GCTA using the --covar option, followed by the filename. For example:
> 
> ```
> gcta64 --mlma --grm mydata --pheno mypheno.txt --covar mycovar.txt --out myresult
> ```
> 
> - In this example, mycovar.txt contains the covariate data that will be included in the MLM model alongside the genetic relationship matrix (GRM) and phenotype data.
> 
> - If you do not have any covariate data or do not wish to include covariates in your MLM analysis, you can simply omit the --covar option from your command line.
> 
> - It is worth noting that the decision to include or exclude covariates should be based on a careful consideration of the characteristics of your dataset and the nature of the phenotype you are studying.

### Q5. Can you perform a power calculation for a GWAS with sample size of 2200 and MAF threshold of less than 1% with a p-value significance threshold of 5x10^-8?
> - Yes, you can perform a power calculation for a GWAS with a sample size of 2200, MAF threshold of less than 1%, and a p-value significance threshold of 5x10^-8.
> 
> - Here's an example of how you can perform the power calculation using G*Power software:
> 
> 	- Open G*Power software and select "t tests" from the test family menu.
> 
> 	- Choose "Logistic regression: Wald test" as the test type.
> 
> 	- Enter the following parameters:
> 
> 		- Effect size (odds ratio): assume a hypothetical odds ratio of 1.2
> 		- Significance level: 5x10^-8
> 		- Power: 80%
> 		- Sample size: 2200
> 		- Number of predictors (excluding the constant): assume 1 predictor (the SNP)
> 	- Click the "Calculate" button to obtain the results.
> - Based on the above parameters, G*Power calculates that a sample size of 2200 would have 80% power to detect an odds ratio of 1.2 at a significance level of 5x10^-8, assuming a SNP with MAF of less than 1%. However, please note that this calculation is based on several assumptions and may not be an accurate reflection of the actual power of your study.

### Q6. Calculate the odds ratio for a GWAS with sample size of 1942, minor allele frequency threshold of > 1%, and a genome wide significance threshold of p > 5x10^-8. The number of controls are 1300 while cases are 642.
> - To calculate the odds ratio for a GWAS, we first need to obtain the number of cases and controls who carry the minor allele and who do not. We can then use these counts to calculate the odds ratio.
> 
> - Let's start by calculating the number of cases and controls who carry the minor allele. We know that the minor allele frequency threshold is > 1%, so any variant with a frequency less than this threshold will be excluded from the analysis.
> 
> - Assuming that there are no missing genotypes, we can use the following formula to calculate the number of cases and controls who carry the minor allele:
> 
> - Number of cases with minor allele = number of cases * minor allele frequency
> Number of controls with minor allele = number of controls * minor allele frequency
> 
> - Plugging in the numbers given in the question, we get:
> 
> - Number of cases with minor allele = 642 * 0.01 = 6.42
> - Number of controls with minor allele = 1300 * 0.01 = 13
> 
> - Next, we can calculate the number of cases and controls who do not carry the minor allele:
> 
> - Number of cases without minor allele = number of cases - number of cases with minor allele
> - Number of controls without minor allele = number of controls - number of controls with minor allele
> 
> - Plugging in the numbers given in the question, we get:
> 
> - Number of cases without minor allele = 642 - 6.42 = 635.58
> - Number of controls without minor allele = 1300 - 13 = 1287
> 
> - Now we can use these counts to calculate the odds ratio. The odds ratio (OR) is the ratio of the odds of the outcome (i.e., disease) in the group exposed to the risk factor (i.e., minor allele) to the odds of the outcome in the group not exposed to the risk factor. The formula for calculating OR is:
> 
> 	- OR = (a/b) / (c/d)
> 
> 		- where a is the number of cases with the minor allele, b is the number of cases without the minor allele, c is the number of controls with the minor allele, and d is the number of controls without the minor allele.
> 
> - Plugging in the numbers from above, we get:
> 
> 	- OR = (6.42/635.58) / (13/1287) = 1.218
> 
> - Therefore, the odds ratio for this GWAS is 1.218.

### Q7. What is the difference between mlm-LOCO and mlm in GCTA?

> - MLM-LOCO and MLM are two methods used in GCTA (Genome-wide Complex Trait Analysis) to perform genome-wide association studies (GWAS) and estimate the genetic effects of variants on complex traits. The main difference between the two methods lies in the way they control for population structure and relatedness between individuals.
> 
> - MLM (Mixed Linear Model) is a method that uses a linear mixed model to account for population structure and relatedness in GWAS. It estimates the genetic effects of variants on a trait while accounting for the correlation structure of individuals in the study due to their relatedness and shared ancestry. The MLM method uses a genetic relationship matrix (GRM) to model the covariance structure of the genetic effects and the residuals, and estimates the variance components of the model using restricted maximum likelihood (REML).
> 
> - MLM-LOCO (MLM with Leave-One-Chromosome-Out) is a variant of the MLM method that controls for population structure and relatedness by fitting a linear mixed model for each chromosome separately while leaving out the chromosome of interest. This method is particularly useful when the genetic effects of a variant on a trait are confounded by population structure or relatedness on a specific chromosome. By fitting the model on the remaining chromosomes, MLM-LOCO can remove the confounding effects of population structure and relatedness from the chromosome of interest, allowing for more accurate estimation of the genetic effects of variants.
> 
> - In summary, while both MLM and MLM-LOCO use linear mixed models to account for population structure and relatedness, MLM-LOCO fits the model on each chromosome separately while leaving out the chromosome of interest. This can lead to more a
> ccurate estimation of the genetic effects of variants in GWAS, particularly when population structure or relatedness is confounding the genetic effects on a specific chromosome.

### Q8. What are the parameters to run a GWAS using MLM-LOCO?

> - `To run a genome-wide association study (GWAS) using MLM-LOCO in GCTA (Genome-wide Complex Trait Analysis), you need to specify several parameters in the GCTA command line. Here are the main parameters required to run MLM-LOCO:
> 
> - The input data file: This is a file containing the genotypes and phenotypes of the individuals in the study. The genotypes are typically represented as a binary PLINK file (.bed, .bim, .fam), while the phenotypes are represented as a text file (.phe).
> 
> - The GRM file: This is a file containing the genetic relationship matrix (GRM) of the individuals in the study. The GRM is used to model the covariance structure of the genetic effects and the residuals in the linear mixed model. You can create the GRM file using the GCTA tool "gcta --make-grm".
> 
> - The trait file: This is a file containing the phenotype data of the individuals in the study. The trait file should be in the format of a text file (.phe), and should include the name of the trait and the phenotype values for each individual.
> 
> - The MLM-LOCO command: To run MLM-LOCO, you need to specify the GCTA command line option "--mlm" followed by the option "--loco" to specify that MLM-LOCO should be used. You can also specify other options, such as the number of threads to use for parallel computing ("-thread-num") and the significance threshold for genome-wide association ("-p").
> 
> - Here is an example command line to run MLM-LOCO in GCTA:

> ```
> gcta --mlm --bfile mydata --grm mygrm --pheno mytrait.phe --loco --out myresults
> ```
> - In this example, "mydata" is the name of the input binary PLINK file, "mygrm" is the name of the genetic relationship matrix file, "mytrait.phe" is the name of the phenotype file, and "myresults" is the name of the output file. The "--loco" option specifies that MLM-LOCO should be used. You can also add other options to customize the analysis as needed.

### Q9. What tools do you recommend to perform power calculation and odds ratio calculations for a GWAS?
> There are several tools and software packages available for performing power calculations and odds ratio calculations for a genome-wide association study (GWAS). Here are some recommendations:
> 
> 1. GPower: GPower is a free and user-friendly software package that can be used to perform power calculations for various types of statistical tests, including GWAS. GPower allows you to specify various parameters of your GWAS, such as sample size, effect size, alpha level, and statistical power, and provides estimates of the required sample size or power for your study. You can download GPower from the following website: https://gpower.software.informer.com/
> 2. QUANTO: QUANTO is another software package that can be used for power calculations in genetic association studies, including GWAS. QUANTO allows you to specify various parameters of your study, such as sample size, disease prevalence, effect size, and allele frequency, and provides estimates of the statistical power or required sample size. QUANTO is a commercial software package that can be purchased from the following website: http://biostats.usc.edu/software
> 3. R packages: There are several R packages that can be used to perform power calculations and odds ratio calculations for GWAS, including "snpPower", "powerGWASinteraction", and "GWASTools". These packages provide functions for calculating power or sample size for various types of association tests, as well as functions for estimating odds ratios and other effect sizes. R is a free and open-source statistical software that can be downloaded from the following website: https://www.r-project.org/
> 4. Online calculators: There are also several online calculators and web-based tools that can be used for power calculations and odds ratio calculations for GWAS. Examples include "GENESIS GWAS Power Calculator" (https://genesis.bwh.harvard.edu/gwaspowercalc/) and "POWER Calculator for Genetic Studies" (http://zzz.bwh.harvard.edu/power/).
> 
> These are just a few examples of the tools and software packages available for power calculations and odds ratio calculations in GWAS. The choice of tool will depend on your specific needs, the available resources, and your level of expertise.


`