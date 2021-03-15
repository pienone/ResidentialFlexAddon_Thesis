# Residential Flexibility Addon

This repository collects a first version of the BALMOREL model add-on developed for including residential demand side flexibilty

# BALMOREL community source and home page:

https://github.com/balmorelcommunity/Balmorel

http://www.balmorel.com/index.php/contact

# Implementation
The add-on has been implemented as a stand alone optimization algorithm in Julia and Pyhton languages. The scope of the first is to determine the optimal load scheduling of flexible technologies (EVs and HPs). The flexible demand profiles determined by Julia linear programming model, scaled for scenario stocks of vehicles and heating plants, and aggregated included in Balmorel add-on inputs. These time series constitute a exogenous parameter named DERESEV HP.


Authors:
Pietro Nonis
Francesco Gaballo

Thesis project: 
Investigation of residential electricity consumption profiles with a focus on theenergy system flexibility potential

From Denmark Technical University
Sustainable Energy
Energy System Analysis


