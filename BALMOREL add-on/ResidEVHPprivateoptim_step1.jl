using Plots, CSV, JuMP, Gurobi, Printf, DataFrames, Tables,


### This file reads prices form Balmorel iteraions, driving patterns and heating demands and genrates optimal load scheduling
### the five models are implemented in separate block in order to run and debug stand alone. Results are saved in csv format and
###  read by Pyhton for aggregation as in as in section 7.2.3 - step 1, following project report.


## Change iteration number
iter="5iter"         ### choose #it
Iteration="5"        ### choose #it
scen="Expected"      ### choose between Expected, RapidScenario and SlowScenario
adopt="0.25"         ### choose
path="GitHub/Julia/"

# 4 Weeks optimization
Regions=collect(1:2)
Years=collect(1:4)
CategoriesAA=collect(1:24)
CategoriesSumHouse=collect(1:24)
CategoriesAW=collect(1:12)
Periods=collect(1:672)
yrs=["2020","2025","2030","2040"]
## Prices
DK1=CSV.read(path*"prices/pricesDK1"*scen*adopt*iter*".csv", delim=",")
DK2=CSV.read(path*"prices/pricesDK2"*scen*adopt*iter*".csv", delim=",")
DK1=dropmissing(DK1)
DK2=dropmissing(DK2)
Prices=[DK1,DK2]
λ=zeros(size(Years)[1],size(Regions)[1],size(Periods)[1])
for r in Regions
    for y in Years
        for t in Periods
            λ[y,r,t]=Prices[r][t,y]
        end
    end
end

#plot(DK1[:,2])
#plot(λ[1,1,:])

## HEATING FLEX1  A-A (Air-to-Air)  Flex=1h########
AAnoFlex=CSV.read(path*"DemAA.csv", delim=",")
AAnoFlex=dropmissing(AAnoFlex)
demandAA=vcat(AAnoFlex[1008:1175,:],AAnoFlex[2016:2183,:],AAnoFlex[4032:4199,:],AAnoFlex[6720:6887,:])
PmaxAA=Pmax=ones(size(Periods)[1],size(CategoriesAA)[1],size(Years)[1])*1
COP_AA=2.8
ηgas=0.98
Ps=demandAA[:,2:25]
λgas=100000

OPT=zeros(size(Years)[1],size(Regions)[1])
for y in Years
    for r in Regions
       finalAA=zeros(size(Periods)[1],size(CategoriesAA)[1])
       POWER=zeros(size(Periods)[1],size(CategoriesAA)[1])
       for c in CategoriesAA
          model_flexF1 = Model(solver = GurobiSolver())
          @variable(model_flexF1, PfAA[t=1:672]>=0)   #load consumption at time t for category i
          @variable(model_flexF1, Pgas[t=1:672]>=0)   #load consumption at time t for category i
          @objective(model_flexF1, Min, sum(λ[y,r,t]*PfAA[t]+λgas*Pgas[t] for t in Periods))
          @constraint(model_flexF1, [t in collect(1:668)], PfAW[t]+PfAW[t+1]>=Ps[t,c]+Ps[t+1,c])
          @constraint(model_flexF1, [t in Periods], PfAW[t]<= PmaxAW[t,c,y]) #Pmax
          @constraint(model_flexF1, sum(PfAW[t] for t in collect(1:672))==sum(Ps[t,c] for t in collect(1:672)))
          solve(model_flexF1)
          POWER[:,c]=getvalue.(PfAA[:])
       end
       if y == 3 && r==1
          display(plot(POWER[:,1]))
       end
       FINAL=vcat(DataFrame(repeat(POWER[1:168,:],10)),DataFrame(repeat(POWER[169:336,:],10)),DataFrame(repeat(POWER[337:504,:],10)),DataFrame(repeat(POWER[505:672,:],22)))
       name=string(path*"Iteration"*Iteration*"/FlexEH/FlexEHAA_DK", r ,"_y", yrs[y] ,".csv")
       CSV.write(name, FINAL, writeheader=false)
    end
end


## HEATING FLEX2  A-A (Air-to-Air) SummerHouse  Flex=1h########
SumHousenoFlex=CSV.read(path*"DemSumHouse.csv", delim=",")
SumHousenoFlex=dropmissing(SumHousenoFlex)
demandSH=vcat(SumHousenoFlex[1008:1175,:],SumHousenoFlex[2016:2183,:],SumHousenoFlex[4032:4199,:],SumHousenoFlex[6720:6887,:])
PmaxSumHouse=Pmax=ones(size(Periods)[1],size(CategoriesSumHouse)[1],size(Years)[1])*1
COP_AA=2.8
ηgas=0.98
Ps=demandSH[:,2:25]
λgas=100000
for y in Years
    for r in Regions
       finalSumHouse=zeros(size(Periods)[1],size(CategoriesSumHouse)[1])
       POWER=zeros(size(Periods)[1],size(CategoriesSumHouse)[1])
       for c in CategoriesSumHouse
          model_flexF2 = Model(solver = GurobiSolver())
          @variable(model_flexF2, PfSumHouse[t=1:672]>=0)   #load consumption at time t for category i
          @variable(model_flexF2, Pgas[t=1:672]>=0)   #load consumption at time t for category i
          @objective(model_flexF2, Min, sum(λ[y,r,t]*PfSumHouse[t]+λgas*Pgas[t] for t in Periods))
          @constraint(model_flexF2, [t in collect(1:668)], PfSumHouse[t]+PfSumHouse[t+1]>=Ps[t,c]+Ps[t+1,c])
          @constraint(model_flexF2, [t in Periods], PfSumHouse[t]<= PmaxSumHouse[t,c,y]) #Pmax
          @constraint(model_flexF2, sum(PfSumHouse[t] for t in collect(1:672))==sum(Ps[t,c] for t in collect(1:672)))
          solve(model_flexF2)
          POWER[:,c]=getvalue.(PfSumHouse[:])
       end
       if y == 1 && r==1
          display(plot(POWER[:,1]))
       end
       FINAL=vcat(DataFrame(repeat(POWER[1:168,:],10)),DataFrame(repeat(POWER[169:336,:],10)),DataFrame(repeat(POWER[337:504,:],10)),DataFrame(repeat(POWER[505:672,:],22)))
       name=string(path*"Iteration"*Iteration*"/FlexEH/FlexEHSumHouse_DK", r ,"_y", yrs[y] ,".csv")
       CSV.write(name,  FINAL, writeheader=false)
      end
end


#####
## HEATING FLEX 3 Air-to-Water with storage flex= 3h############
AWnoFlex=CSV.read(path*"DemAW.csv", delim=",")
AWnoFlex=dropmissing(AWnoFlex)
demandAA=vcat(AWnoFlex[1008:1175,:],AWnoFlex[2016:2183,:],AWnoFlex[4032:4199,:],AWnoFlex[6720:6887,:])
PmaxAW=ones(size(Periods)[1],size(CategoriesAW)[1],size(Years)[1])*3
COP_AW=2.8
Ps=demandAA[:,2:13]
final=zeros(size(Periods)[1],size(CategoriesAW)[1])
for y in Years
     for r in Regions
       POWER=zeros(size(Periods)[1],size(CategoriesSumHouse)[1])
       MISSING=zeros(size(Periods)[1],size(CategoriesSumHouse)[1])
       for c in CategoriesAW
       model_flexF3 = Model(solver = GurobiSolver())
       @variable(model_flexF3, PfAW[t=1:672]>=0)   #load consumption at time t for category i
       @objective(model_flexF3, Min, sum(λ[y,r,t]*PfAW[t] for t in Periods))
       @constraint(model_flexF3, [t in collect(1:668)], PfAW[t]+PfAW[t+1]+PfAW[t+2]+PfAW[t+3]>=Ps[t,c]+Ps[t+1,c]+Ps[t+2,c]+Ps[t+3,c])
       @constraint(model_flexF3, [t in Periods], PfAW[t]<= PmaxAW[t,c,y]) #Pmax
       @constraint(model_flexF3, sum(PfAW[t] for t in collect(1:672))==sum(Ps[t,c] for t in collect(1:672)))
       solve(model_flexF3)
       POWER[:,c]=getvalue.(PfAW[:])
       end
       if y == 1 && r==1
          display(plot(POWER[:,2]))
          display(plot!(Ps[:,1]))
          display(plot(POWER[40:88,4],label="Smart",legend=:topleft, right_margin = 15Plots.mm, ylabel = "Power [kW]",ylims=[0,3.6]))
          display(plot!(twinx(),λ[1,1,40:88],label="Price",color=:green, ylabel = "[€/MWh]", right_margin = 15Plots.mm,ylims=[0,140]))
          display(plot!(Ps[40:88,4],label="Passive",legend=:topleft, right_margin = 15Plots.mm, bottom_margin=5Plots.mm))
          xlabel!("Hours")
          plot!(size=(800,280))
          savefig("heatpump.png")
          plot!(label=["Smart","Passive","Price"])
       end
       FINAL=vcat(DataFrame(repeat(POWER[1:168,:],10)),DataFrame(repeat(POWER[169:336,:],10)),DataFrame(repeat(POWER[337:504,:],10)),DataFrame(repeat(POWER[505:672,:],22)))
       name=string(path*"Iteration"*Iteration*"/FlexEH/FlexEHAW_DK", r ,"_y", yrs[y] ,".csv")
       CSV.write(name,  DataFrame(FINAL), writeheader=false)
end
end








## ELECTRIC VEHICLES total
using Plots, CSV, JuMP, Gurobi, Printf, DataFrames, Tables

Regions=collect(1:2)
Years=collect(1:4)
Periods=collect(1:672)
CategoriesAA=collect(1:24)
CategoriesSumHouse=collect(1:24)
CategoriesAW=collect(1:12)
Days=collect(1:2917)*3
WEEKS=[6,12,24,40]*168


## Change iteration number
#iter="2"
path="C:/Users/pietr/Documents/GitHub/Tesi/Julia/"## Prices
DK1=CSV.read(path*"prices/pricesweeklyDK1"*scen*adopt*iter*".csv", delim=",")
DK2=CSV.read(path*"prices/pricesweeklyDK2"*scen*adopt*iter*".csv", delim=",")
DK1=dropmissing(DK1)
DK2=dropmissing(DK2)
Prices=[DK1,DK2]
λ=zeros(size(Years)[1],size(Regions)[1],size(Periods)[1])
for r in Regions
    for y in Years
        for t in Periods
            λ[y,r,t]=Prices[r][t,y]
        end
    end
end


#############################
########### BEV #############
#############################

BEVtypes=collect(1:20)
Availability=CSV.read(path*"Driving/Driving_patterns_avail.csv")
KM_demand=CSV.read(path*"Driving/Driving_patterns.csv")
Ex=CSV.read(path*"Driving/Driving_patterns_exit.csv")
ηcar=6     #Km/KWh
ChMax=3.7
fday=[2,2,1,2,6]*24
ηch=0.95
EM=0.9
BsizeBEV=60
M=50000   #BigM

for y in Years
    AvBEVWD=Availability[1:size(Periods)[1],3:22]#+fday[y],3:22]
    KMBEVWD=KM_demand[1:size(Periods)[1],3:22]#+fday[y],3:22]
    Exit=Ex[1:size(Periods)[1],3:22]#+fday[y],3:22]
    for r in Regions
        SOCC=zeros(size(Periods)[1],size(BEVtypes)[1])
        POWER=zeros(size(Periods)[1],size(BEVtypes)[1])
        for e in BEVtypes
            model_EV = Model(solver = GurobiSolver())
            @variable(model_EV, ChEV[t in Periods]>=0)   #load consumption at time t for type e
            @variable(model_EV, 0<=SOC[t in Periods]<=BsizeBEV)
            @objective(model_EV, Min, sum(λ[y,r,t]*ChEV[t] for t in Periods))
            @constraint(model_EV, SOC[1]==SOC[672]+ChEV[672]) ##t in Periods
            @constraint(model_EV, [t =2:size(Periods)[1]], SOC[t]==SOC[t-1]+ChEV[t]-KMBEVWD[t,e]/ηcar)
            @constraint(model_EV, [t in Periods], ChEV[t]<= AvBEVWD[t,e]*ChMax)
            @constraint(model_EV, [t in Periods], SOC[t]>=Exit[t,e]*BsizeBEV*EM)   #when going out full battery needed
            solve(model_EV)
            POWER[:,e]=getvalue.(ChEV[:])
            SOCC[:,e]=getvalue.(SOC[:])/60
        end
        if y == 1 && r==1
          display(plot(POWER[169:672]))
          display(plot!(λ[y,r,169:672]))
       end
       FINAL=vcat(DataFrame(repeat(POWER[1:168,:],10)),DataFrame(repeat(POWER[169:336,:],10)),DataFrame(repeat(POWER[337:504,:],10)),DataFrame(repeat(POWER[505:672,:],22)))
#       FINAL=vcat(DataFrame(POWER[1:168,:]),DataFrame(POWER[169:336,:]),DataFrame(POWER[337:504,:]),DataFrame(POWER[505:672,:]))
       name=string(path*"Iteration"*Iteration*"/FlexEV/FlexEV_DK", r ,"_y", yrs[y] ,".csv")
       CSV.write(name,  DataFrame(FINAL), writeheader=false)
    end
end


Ps=CSV.read(path*"FLEXNOBEV.csv")
display(plot(POWER[96:144,4],label="Smart",legend=:topleft, right_margin = 15Plots.mm, ylabel = "Power [kW]",ylims=[0,4.9]))
display(plot!(Ps[97:145,4],label="Passive",legend=:topleft, right_margin = 15Plots.mm, bottom_margin=5Plots.mm))
display(plot!(twinx(),λ[1,1,96:144],label="Price",color=:green,legend=:topleft, ylabel = "[€/MWh]", right_margin = 15Plots.mm,ylims=[20,55]))
vline!([18,37],color=:black,label="Arrive")
vline!([10,30],color=:red,label="Departure",legend=:topright)
xlabel!("Hours")
plot!(size=(800,280))
savefig("EV.png")








#############################
########## PHEV #############
#############################
PHEVtypes=collect(1:10)
Availability=CSV.read(path*"Driving/Driving_patterns_avail_PHEV.csv")
KM_demand=CSV.read(path*"Driving/Driving_patterns_PHEV.csv")
Ex=CSV.read(path*"Driving/Driving_patterns_exit_PHEV.csv")
ηcar=5     #Km/KWh
ChMax=3.7
fday=[2,2,1,2,6]*24
ηch=0.95
BsizePHEV=10
M=50000   #BigM

for y in Years
    AvPHEV=Availability[1:size(Periods)[1],3:12]#+fday[y],3:12]
    KMPHEV=KM_demand[1:size(Periods)[1],3:12]#+fday[y],3:12]
    ExitPHEV=Ex[1:size(Periods)[1],3:12]#+fday[y],3:12]
    for r in Regions
        SOCC=zeros(size(Periods)[1],size(PHEVtypes)[1])
        POWERPHEV=zeros(size(Periods)[1],size(PHEVtypes)[1])
        for e in PHEVtypes
            model_PHEV = Model(solver = GurobiSolver())
            @variable(model_PHEV, ChEV[t in Periods]>=0)   #load consumption at time t for type e
            @variable(model_PHEV, 0<=SOC[t in Periods]<=BsizePHEV)
            @variable(model_PHEV, Oil[t in Periods,e in PHEVtypes]>=0)
            @objective(model_PHEV, Min, sum(λ[y,r,t]*ChEV[t]+100000*Oil[t,e] for t in Periods))
            @constraint(model_PHEV, SOC[1]==SOC[672]+ChEV[672]) ##t in Periods
            @constraint(model_PHEV, [t =2:size(Periods)[1]], SOC[t]==SOC[t-1]+Oil[t,e]+ChEV[t]-KMPHEV[t,e]/ηcar)
            @constraint(model_PHEV, [t in Periods], ChEV[t]<= AvPHEV[t,e]*ChMax)
            @constraint(model_PHEV, [t in Periods], SOC[t]>=ExitPHEV[t,e]*BsizePHEV)   #when going out full battery needed
            solve(model_PHEV)
            POWERPHEV[:,e]=getvalue.(ChEV[:])
            SOCC[:,e]=getvalue.(SOC[:])/10
        end
        if r==1
          display(plot(POWERPHEV[:,1:3]))
        end
        FINAL=vcat(DataFrame(repeat(POWERPHEV[1:168,:],10)),DataFrame(repeat(POWERPHEV[169:336,:],10)),DataFrame(repeat(POWERPHEV[337:504,:],10)),DataFrame(repeat(POWERPHEV[505:672,:],22)))
#        FINAL=vcat(DataFrame(POWERPHEV[1:168,:]),DataFrame(POWERPHEV[169:336,:]),DataFrame(POWERPHEV[337:504,:]),DataFrame(POWERPHEV[505:672,:]))
        name=string(path*"Iteration"*Iteration*"/FlexEV/FlexPHEV_DK", r ,"_y", yrs[y] ,".csv")
        CSV.write(name,  DataFrame(FINAL), writeheader=false)
    end
end
