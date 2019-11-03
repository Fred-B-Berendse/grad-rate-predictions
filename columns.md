# Data Description

The following data columns were utilized in this analysis:

## HD2017 

This contains institution identifying information. Below is a table of colums pulled from this table.


| Column Name | Data Type | Description | 
|-------------|-----------|-------------| 
| UNITID | integer | unique institution ID |
| INSTNM | text | Institution Name |
| CITY | text | city |
| STABBR | text, categorical | state abbreviation |
| ICLEVEL | integer, categorical | level category (4-yr, 2-yr, <2-yr) | 
| CONTROL | integer, categorical | institutional control (public, private not-for-profit, private for-profit) |
| HLOFFER | integer, categorical | highest level offered  |
| HBCU | integer, categorical | code to indicate an HBCU institution |
| TRIBAL | integer, categorical | code to indicate a tribal institution |
| LOCALE | integer, categorical | "large city" to "rural" |
| INSTSIZE | integer, categorical | category based on total enrollment |
| LONGITUD | float | longitude of the institution |
| LATITUDE | float | latitude of the institution |
| LANDGRNT | integer, categorical | category indicates whether the institution is land grant institution |
||||

## ADM2017

This table contains dmissions-related information such as admissions considerations, applicants, admitted applicants, applicants who enrolled, and SAT/ACT test data (if applicable.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| UNITID | integer | unique institution ID |
| APPLCN | integer | total number of applicants |
| ADMSSN | integer | total number of admissions |
| ENRLT | integer | total enrollment |
| ENRLFT | integer | full-time enrollment |
| ENRLPT | integer | part-time enrollment |
| SATVR25 | integer | SAT verbal 25th percentile |
| SATVR75 | integer | SAT verbal 75th percentile |
| SATMT25 | integer | SAT math 25th percentile |
| SATMT75 | integer | SAT math 75th percentile |
| ACTEN25 | integer | ACT english 25th percentile |
| ACTEN75 | integer | ACT english 75th percentile |
| ACTMT25 | integer | ACT math 25th percentile |
| ACTMT75 | integer | ACT math 75th percentile|
||||

## GR2017

 This table contains graduation rates for the cohort of full-time, first-degree/certificate-seeking stidemts within 100% and 150% normal time. Cohorts are broken down by race/ethnicity and gender.
    
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| UNITID | integer | unique institution ID |
| CHRTSTAT | integer, categorical | code indicating completion status by program type |
| COHORT | integer, cateogrical | cohort of program type |
| GRAIANT | integer | American Indian/Alaska Native total |
| GRASIAT | integer | Asian total |
| GRBKAAT | integer | Black or African American total |
| GRHISPT | integer | Hispanic or Latino total |
| GRNHPIT | integer | Native Hawaiian/Pacific Islander total |
| GRWHITT | integer | White total |
| GR2MORT | integer | Two or more races total |
||||

## GR2017_PELL_SSL

This table contains gaduation rates for three subcohorts: Pell Grant recipients, Subsidized Stafford Loan (without Pell Grant), and non PG/SSL recipients. 

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| UNITID | integer | unique institution ID |
| PSGRTYPE | integer, categorical | subgroup: 4-yr degree+cert, 4-yr degree, 4-yr cert, <=2-yr degree+cert |
| PGADJCT | integer | Pell Grant recipients - count adjusted for exclusions |
| PGCMBAC | integer | Pell Grant bachelor's degree recipients - completed within 150% time | 
| SSADJCT | integer | Subsidized Stafford Loan (w/o Pell Grant) recipients - count adjusted for exclusions |
| SSCMBAC | integer | Subsidized Stafford Loan (w/o Pell Grant) bachelor's degree recipients - completed within 150% time | 
| NRADJCT | integer | Non-aid-recipients - count adjusted for exclusions |
| NRCMBAC | integer | Non-aid-recipients bachelor's degree recipients - completed within 150% time |

## SFA2017 

This table contains student financial aid data,including scholarships/grants and loands. Awards are disaggregated into federal, state/district, and institutional. Award numbers and amounts are also disaggregated by recipient family's income level. 

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| UNITID | integer | unique institution ID |
| SCFY1N | integer | number of students in full-year cohort |
| SCFY11N | integer | number of full-year cohort paying in-district tuition rates |
| SCFY12N | integer | number of full-year cohort paying in-state tuition |
| SCFY13N | integer | number of full-year cohort paying out-of-state tuition |
| UAGRNTP | integer | Percent of undergraduate students awarded grant/scholarship aid (any type) |
| UPGRNTP | integer |  Percent of undergraduate students awarded Pell Grants |
| GRNTN2 | integer |  Number of students in 2016-17 cohort |
| GRNTON2 | integer |  Number of students in cohort living on-campus |
| GRNTWF2 | integer |  Number of students in cohort living off-campus with family | 
| GRNTOF2 | integer |  Number of students in cohort living off-campus |
| NPGRN2 | integer | Average net price (tuition - grant/scholarship aid) for 2016-17 cohort |
| GRNT4A2 | integer | Average amount of grant and scholarship aid awarded to cohort, all income levels |

    
