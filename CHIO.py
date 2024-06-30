#Coronavirus herd immunity optimizer (CHIO)
import numpy as np

def calculateFitness(f_obj_v):
    f_fitness = np.zeros_like(f_obj_v)
    ind = np.where(f_obj_v >= 0)[0]
    f_fitness[ind] = 1. / (f_obj_v[ind] + 1)
    ind = np.where(f_obj_v < 0)[0]
    f_fitness[ind] = 1 + np.abs(f_obj_v[ind])
    return f_fitness










 
def F1(x): 
    return np.sum(x**2) 
 
def F2(x): 
    return np.sum(np.abs(x)) + np.prod(np.abs(x)) 
 
def F3(x): 
    dim = x.shape[0] 
    o = 0 
    for i in range(dim): 
        o += np.sum(x[:i+1])**2 
    return o 
 
def F4(x): 
    return np.max(np.abs(x)) 
 
def F5(x): 
    dim = x.shape[0] 
    return np.sum(100*(x[1:dim] - x[:dim-1]**2)**2 + (x[:dim-1] - 1)**2) 
 
def F6(x): 
    return np.sum(np.abs(x + 0.5)**2) 
 
def F7(x): 
    dim = x.shape[0] 
    return np.sum(np.arange(1, dim+1) * x**4) + np.random.rand() 
 
def F8(x): 
    return np.sum(-x * np.sin(np.sqrt(np.abs(x)))) 
 
def F9(x): 
    dim = x.shape[0] 
    return np.sum(x**2 - 10*np.cos(2*np.pi*x)) + 10*dim 
 
# Define the remaining functions F10 to F23 similarly 
 
def Get_Functions_details(F): 
    switcher = { 
        'F1': (F1, -100, 100, 30), 
        'F2': (F2, -10, 10, 30), 
        'F3': (F3, -100, 100, 30), 
        'F4': (F4, -100, 100, 30), 
        'F5': (F5, -30, 30, 30), 
        'F6': (F6, -100, 100, 30), 
        'F7': (F7, -1.28, 1.28, 30), 
        'F8': (F8, -500, 500, 30), 
        'F9': (F9, -5.12, 5.12, 30), 
        # Add the remaining cases 
    } 
     
    fobj, lb, ub, dim = switcher.get(F, (None, None, None, None)) 
     
    return lb, ub, dim, fobj 
  

 
import numpy as np

def initialization(SearchAgents_no, dim, ub, lb):
    print("ub",ub)
    Boundary_no = ub  # number of boundaries
    Positions = np.zeros((SearchAgents_no, dim))
    
    # If the boundaries of all variables are equal and user enters a single number for both ub and lb
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    
    # If each variable has a different lb and ub
    elif Boundary_no > 1:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no, 1) * (ub_i - lb_i) + lb_i
    
    return Positions
def main(): 
    np.random.seed(42)  # Set random seed for reproducibility 
 
    PopSize = 30 
    MaxAge = 100 
    C0 = 1 
    Max_iter = 100000 
    SpreadingRate = 0.05 
    runs = 1 
    ObjVal = np.zeros(PopSize) 
    Age = np.zeros(PopSize) 
    BestResults = np.zeros(runs) 
    Function_name='F1' 
    for funNum in range(7, 8):  # fun#1 to fun#23 
        if funNum == 1: 
            Function_name = 'F1' 
        elif funNum == 2: 
            Function_name = 'F2' 
        # Add more conditions for other funNum values 
 
        lb, ub, dim, fobj = Get_Functions_details(Function_name) 
 
        for run in range(runs): 
            swarm = initialization(PopSize, dim, ub, lb) 
 
            for i in range(PopSize): 
                ObjVal[i] = fobj(swarm[i]) 
 
            Fitness = calculateFitness(ObjVal) 
            Status = np.zeros(PopSize) 
 
            for i in range(C0): 
                Status[np.random.randint(PopSize)] = 1 
 
            itr = 0 
            while itr < Max_iter: 
                for i in range(PopSize): 
                    NewSol = swarm[i].copy() 
                    CountCornoa = 0 
 
                    confirmed = np.where(Status == 1)[0] 
                    normal = np.where(Status == 0)[0] 
                    recovered = np.where((ObjVal < np.mean(ObjVal)) & (Status == 2))[0] 
 
                    for j in range(dim): 
                        r = np.random.rand() 
                        if (r < SpreadingRate/3) and (len(confirmed) > 0): 
                            zc = np.random.choice(confirmed) 
                            NewSol[j] = swarm[i, j] + (swarm[i, j] - swarm[zc, j]) * (np.random.rand() - 0.5) * 2 
                            NewSol[j] = np.clip(NewSol[j], lb, ub) 
                            CountCornoa += 1 
                        elif (r < SpreadingRate/2) and (len(normal) > 0): 
                            zn = np.random.choice(normal) 
                            NewSol[j] = swarm[i, j] + (swarm[i, j] - swarm[zn, j]) * (np.random.rand() - 0.5) * 2 
                            NewSol[j] = np.clip(NewSol[j], lb, ub) 
                        elif (r < SpreadingRate) and (len(recovered) > 0): 
                            Index3 = np.argmin(ObjVal[recovered]) 
                            NewSol[j] = swarm[i, j] + (swarm[i, j] - swarm[Index3, j]) * (np.random.rand() - 0.5) * 2 
                            NewSol[j] = np.clip(NewSol[j], lb, ub) 
 
                    ObjValSol = fobj(NewSol) 
                    FitnessSol = calculateFitness(ObjValSol) 
 
                    if ObjVal[i] > ObjValSol: 
                        swarm[i] = NewSol 
                        Fitness[i] = FitnessSol 
                        ObjVal[i] = ObjValSol 
                    else: 
                        if Status[i] == 1: 
                            Age[i] += 1 
 
                    if (Fitness[i] < np.mean(Fitness)) and (Status[i] == 0) and (CountCornoa > 0): 
                        Status[i] = 1 
                        Age[i] = 1 
 
                    if (Fitness[i] >= np.mean(Fitness)) and (Status[i] == 1): 
                        Status[i] = 2 
                        Age[i] = 0 
 
                    if Age[i] >= MaxAge: 
                        swarm[i] = initialization(1, dim, ub, lb).flatten() 
                        Status[i] = 0 
 
                if itr % 100 == 0: 
                    print(f'Fun#{funNum}, Run#{run}, Itr {itr}, Results {np.min(ObjVal)}') 
 
                itr += 1 
 
            BestResults[run] = np.min(ObjVal) 
 
        print('Done ') 
main()