import math

T = 145.77
p = 81100
M = 1.5
gamma = 1.4
R= 287

c = math.sqrt(gamma*R*T)
u = M*c
rho = p/(R*T)

print("c = ", c)
print("u = ", u)
print("rho = ", rho)

# We know that p = rho*R*T (ideal gas law)
# We want p = 1, rho = 1 and T = 1
print('===================')
T = 1/287
p = 1
M = 2.5
gamma = 1.4
R= 287

c = math.sqrt(gamma*R*T)
u = M*c
rho = p/(R*T)

print("c = ", c)
print("u = ", u)
print("rho = ", rho)

# How to change the ratio p / rho? --> different value for R. Or better, temperature. 