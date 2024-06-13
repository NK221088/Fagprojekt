E_test = {0: (0.5714285969734192, 7), 1: (0.7142857313156128, 7), 2: (0.1666666716337204, 6), 3: (0.5, 6), 4: (0.5, 6), 5: (1.0, 3)}
modelList = ["ANN"]
E_gen_hat = {model: 0 for model in modelList}
N = 35
for model in modelList:
        for i in range(6):
            E_gen_hat[model] += E_test[i][0] * (E_test[i][1] / N)
            
print(E_gen_hat)