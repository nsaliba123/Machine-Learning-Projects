# Machine-Learning-Projects
The three projects include the following models: trend following, hiearchial risk parity, and hege fund replciation.  The indicators.py file is used for trend following and the two excel files are for the hedge fund replication.

Hedge Fund Replication: Uses KNN to replciated hedge fund returns with ETFs

Xgboost trend following: Import indicators.py. Uses Xgboost to build a trading model based on technical indactors.  The positonal size is determined by the probability of buy or sell signal. The technical indicators can be found in indicators.py file   

Risk Paritity: Uses Hierchial Risk Parity to create a risk parity portfolio.  What's uniuqe aobut this partiular model is fixed income is broken into 3 parts, short duration, floating rate, and long duration bonds. The dendogram is shown at the bottom allowing you to obersve which assets provided maximum diversificaiton. Further, you can see the superior returns vs a tradional inverse variance portfolio


The ultimate goal is to combine the strategies to build a robust multi strategy portfolio.
