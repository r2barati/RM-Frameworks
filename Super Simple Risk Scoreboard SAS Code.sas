data RiskScoreboard;
  input Risk_Category $ Risk_Description $ Likelihood $ Impact $ Severity $ Risk_Owner $ Mitigation_Strategy $ Current_Status $;
  datalines;
Category 1   Risk 1 description     High       Medium High     John Doe   Implement controls  In progress
Category 1   Risk 2 description     Low        High   Medium   Jane Smith Develop contingency Not started
Category 2   Risk 3 description     Medium     Low    Low      Mark Lee   Enhance monitoring  Completed
Category 2   Risk 4 description     High       High   High     Sarah Wong Transfer risk       In progress
;
run;

proc print data=RiskScoreboard;
run;
