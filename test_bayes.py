#============================================
# program: test_bayes.py
# purpose: testing Bayes fitting following D. Hogg's primer
#============================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import scipy.special
import utilities as util

#============================================
# reads table of measured data. Expects
# three columns with (x, y, sig)
# Returns arrays x,y,sig.
def readdata(name):
    f   = open(name)
    lst = []
    for line in f:
        lst.append([float(d) for d in line.split()])
    x   = np.array([d[0] for d in lst])
    y   = np.array([d[1] for d in lst])
    sig = np.array([d[2] for d in lst])
    return x,y,sig

# ???????????????????????????????????????????
    
def standardFit(x, y, sy, string):        
    n = len(x)
    Y = y.reshape(n, 1)
    A = np.ones([n,2])
    C = np.zeros([n,n])
    if string == 'outlier':
        x = x[-16:]
        Y = y[-16:]
        Y = y.reshape(n, 1)
        sy = sy[-16:] 
        A = np.ones([n,2])
        C = np.zeros([n,n])
    for i in range(n):
        A[i,1] = x[i]
        C[i,i] = (sy[i])**2
    A_T = np.matrix.transpose(A)
    C_inverse = np.linalg.inv(C)
    X = np.dot((np.linalg.inv(np.matmul(A_T, np.matmul(C_inverse, A)))), 
               (np.matmul(A_T, np.matmul(C_inverse, Y))))
    b = X[0]
    m = X[1]
    
    a = np.linalg.inv(np.matmul(A_T, np.matmul(C_inverse, A)))
    u_b = np.sqrt(a[0,0])
    u_m = np.sqrt(a[1,1])
    
    term2 = np.matrix.transpose(Y - np.matmul(A, X))
    term1 = Y - np.matmul(A, X)
    chi2 = np.dot(term2, np.dot(C_inverse, term1))
    q = scipy.special.gammaincc((0.5 * float(n - 2)), (0.5 * chi2))
    
    return b, m, u_b, u_m, chi2, q

def L(x, y, sy, theta): # the probability density function 
    n = len(x)
    result = 0.0
    for i in range(n):
        result = result + (((y[i] - (theta[1] * x[i]) - theta[0])**2) / (2 * (sy[i]**2)))
    return -result

def L2(x, y, sy, theta): # the probability density function 
    b, m, P_b, Y_b, V_b = theta
    result = ((((1 - P_b)/(np.sqrt(2 * np.pi * (sy**2)))) * 
        np.exp(-(y - (m * x) - b)**2/(2*(sy**2)))) + 
        (((P_b)/(np.sqrt(2 * np.pi * (V_b + sy**2)))) * 
        np.exp(-(y - Y_b)**2/(2*(V_b + (sy**2))))))
    r = np.sum(np.log(result))
    return r

def prior(theta):
    prior = 0.0
    b, m, P_b, Y_b, V_b = theta
    if V_b < 0 or P_b > 1 or P_b < 0: 
        prior = -1e60
    return prior
        
def methast(fTAR, T, theta0, delta, string):
    m = len(theta0)
    theta = np.zeros([T, m]) 
    theta[0,:] = theta0 # starting point
    x,y,sy = readdata('hogg.txt')
    
    for i in range(T-1):
        theta_prime = theta[i,:] + (delta * ((2 * np.random.rand(m)) - 1))
        if string != 'dim5':
            alpha = fTAR(x, y, sy, theta_prime) - fTAR(x, y, sy, theta[i,:])
        if string == 'dim5':
            alpha = (fTAR(x, y, sy, theta_prime) + prior(theta_prime)) - (fTAR(x, y, sy, theta[i,:]) + prior(theta[i,:]))
        if (alpha >= 0.0): # definitely accept
            theta[i+1,:] = theta_prime 
        else:
            u = np.random.rand(1) # accept with probability
            if (np.log(u) < alpha):
                theta[i+1,:] = theta_prime 
            else:
                theta[i+1,:] = theta[i,:]
    return theta  

# ???????????????????????????????????????????

#============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("Pbad",type=float,
                        help="probability for bad data points:\n"
                             "   0<=Pbad<1. First run with Pbad=0\n"
                             "   (no pruning of bad data points)")
    parser.add_argument("T",type=int,
                        help="length of Markov chain (depends on problem)")

    args       = parser.parse_args()
    Pb0        = args.Pbad
    T          = args.T

    x,y,sy       = readdata('hogg.txt')  

# ???????????????????????????????????????????

    # GENERALIZED LEAST-SQUARES FITTING
    
    b, m, u_b, u_m, chi2, q= standardFit(x, y, sy, 'nooutlier')
    
    values = 'The y-intercept, b: ' + str(b)
    values += '\n'
    values += 'The slope, m: ' + str(m)
    values += '\n'
    values += 'The uncertainty in b: ' + str(u_b)
    values += '\n'
    values += 'The uncertainty in m: ' + str(u_m)
    values += '\n'
    values += 'Chi-Squared, χ2: ' + str(chi2)
    values += '\n'
    values += 'q: ' + str(q)
    print(values)
    
    y_fit = (m * x) + b
    plt.plot(x, y, 'o', x, y_fit, '-')
    plt.errorbar(x, y, sy, fmt = '|')
    plt.legend(('Raw Data', 'Standard Best-Fit', 'Known Gaussian Uncertainties in y'),loc = 0)
    plt.title('The Standard Practice: Generalized Least-Squares Fitting')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    
# --------------------------------------
    
    # REMOVING OUTLIERS BY HAND
    
    x = x[-16:]
    y = y[-16:]
    sy = sy[-16:]
    b, m, u_b, u_m, chi2, q= standardFit(x, y, sy, 'outlier')
    
    values = 'The y-intercept, b: ' + str(b)
    values += '\n'
    values += 'The slope, m: ' + str(m)
    values += '\n'
    values += 'The uncertainty in b: ' + str(u_b)
    values += '\n'
    values += 'The uncertainty in m: ' + str(u_m)
    values += '\n'
    values += 'Chi-Squared, χ2: ' + str(chi2)
    values += '\n'
    values += 'q: ' + str(q)
    print(values)
    
    y_fit = (m * x) + b
    plt.plot(x, y, 'o', x, y_fit, '-')
    plt.errorbar(x, y, sy, fmt = '|')
    plt.legend(('Raw Data', 'Standard Best-Fit', 'Known Gaussian Uncertainties in y'),loc = 0)
    plt.title('The Standard Practice + Removing Outliers by Hand')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    
# --------------------------------------
    
    # THE DISTRIBUTION OF b AND m
    
    x,y,sy = readdata('hogg.txt')    
    theta0 = np.array([200.0, 1.0])
    delta = np.array([1.0, 0.1])
    theta = methast(L, T, theta0, delta, 'hey :p')
    b_MH = theta[10000:,0]
    m_MH = theta[10000:,1]
    bs = np.std(b_MH)
    ms = np.std(m_MH)
    values = 'b mean from Metropolis-Hastings: ' + str(np.mean(b_MH))
    values += '\n'
    values += 'm standard deviation Metropolis-Hastings: ' + str(bs)
    values += '\n'
    values += 'm mean from Metropolis-Hastings: ' + str(np.mean(m_MH))
    values += '\n'
    values += 'm standard deviation: ' + str(ms)
    print(values)
    
    # QUALITY ASSURANCE PLOT FOR b
    plt.xlabel('b')
    plt.plot(b_MH, range(len(b_MH)), linestyle='-', color='black', linewidth=1.0)
    plt.title('Quality Assurance Plot for b')
    plt.xlabel('b')
    plt.ylabel('Iteration Number, t')
    plt.grid(True)
    plt.show()

    # Finding the most frequent b
    your_data1, bins1, patches1 = plt.hist(b_MH,bins=100)
    plt.title('Histogram of b')
    plt.xlabel('b')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    for j in range(0, len(your_data1)):
        elem = your_data1[j]
        if elem == your_data1.max():
            break
    peak_b = bins1[j]
    print('peak M-H b value: ' + str(np.mean(peak_b)))
    
    # QUALITY ASSURANCE PLOT FOR m
    plt.xlabel('m')
    plt.plot(m_MH, range(len(m_MH)), linestyle='-', color='black', linewidth=1.0)
    plt.title('Quality Assurance Plot for m')
    plt.xlabel('m')
    plt.ylabel('Iteration Number, t')
    plt.grid(True)
    plt.show()

    # Finding the most frequent m
    your_data2, bins2, patches2 = plt.hist(m_MH,bins=100)
    plt.title('Histogram of m')
    plt.xlabel('m')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    for j in range(0, len(your_data2)):
        elem = your_data2[j]
        if elem == your_data2.max():
            break
    peak_m = bins2[j]
    print('peak M-H m value: ' + str(np.mean(peak_m)))
    
    # OVERLAYING THE PLOT IN 6a 
    b, m, u_b, u_m, chi2, q = standardFit(x, y, sy, 'nooutlier')
    y_fit = (m * x) + b
    y_fit2 = (peak_m * x) + peak_b
    plt.plot(x, y, 'o', x, y_fit, '-', x, y_fit2, '-')
    plt.errorbar(x, y, sy, fmt = '|')
    plt.legend(('Raw Data', 'Standard Best-Fit', 'Overlaid MH Standard Best-Fit', 'Known Gaussian Uncertainties in y'),loc = 0)
    plt.title('Overlaid Standard Practice Plots')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    
    plt.hexbin(b_MH, m_MH)
    plt.title('2D Histogram, b vs. m')
    plt.xlabel('b')
    plt.ylabel('m')
    plt.grid(True)
    plt.show()

# --------------------------------------
    
    # REMOVING OUTLIERS BY MODELING
        
    avg_y = np.average(y)
    theta0 = np.array([30.0, 2.0, 0.2, avg_y, np.average(y**2)])
    delta = np.array([1.0, 0.1, 0.1, 10, 300])
    theta = methast(L2, T, theta0, delta, 'dim5')
    b_MH = theta[10000:,0]
    m_MH = theta[10000:,1]
    Pb_MH = theta[10000:,2]
    Yb_MH = theta[10000:,3]
    Vb_MH = theta[10000:,4]
    values = 'b mean from Metropolis-Hastings: ' + str(np.mean(b_MH))
    values += '\n'
    values += 'm mean from Metropolis-Hastings: ' + str(np.mean(m_MH))
    print(values)
    
    # QUALITY ASSURANCE PLOT FOR b
    plt.plot(b_MH, range(len(b_MH)), linestyle='-', color='black', linewidth=1.0)
    plt.title('Quality Assurance Plot for b')
    plt.xlabel('b')
    plt.ylabel('Iteration Number, t')
    plt.grid(True)
    plt.show()
    
    #Finding the most frequent b
    your_data1, bins1, patches1 = plt.hist(b_MH,bins=100)
    plt.title('Histogram of b')
    plt.xlabel('b')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    for j in range(0, len(your_data1)):
        elem = your_data1[j]
        if elem == your_data1.max():
            break
    peak_b = bins1[j]
    print('peak M-H b value: ' + str(np.mean(peak_b)))
    
    # QUALITY ASSURANCE PLOT FOR m
    plt.xlabel('m')
    plt.plot(m_MH, range(len(m_MH)), linestyle='-', color='black', linewidth=1.0)
    plt.title('Quality Assurance Plot for m')
    plt.xlabel('m')
    plt.ylabel('Iteration Number, t')
    plt.grid(True)
    plt.show()
    
    # Finding the most frequent m
    your_data2, bins2, patches2 = plt.hist(m_MH,bins=100)
    plt.title('Histogram of m')
    plt.xlabel('m')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    for j in range(0, len(your_data2)):
        elem = your_data2[j]
        if elem == your_data2.max():
            break
    peak_m = bins2[j]
    print('peak M-H m value: ' + str(np.mean(peak_m)))

    # QUALITY ASSURANCE PLOT FOR P_b
    plt.xlabel('P_b')
    plt.plot(Pb_MH, range(len(Pb_MH)), linestyle='-', color='black', linewidth=1.0)
    plt.title('Quality Assurance Plot for P_b')
    plt.xlabel('P_b')
    plt.ylabel('Iteration Number, t')
    plt.grid(True)
    plt.show()
    
    # QUALITY ASSURANCE PLOT FOR Y_b
    plt.xlabel('Y_b')
    plt.plot(Yb_MH, range(len(Yb_MH)), linestyle='-', color='black', linewidth=1.0)
    plt.title('Quality Assurance Plot for Y_b')
    plt.xlabel('Y_b')
    plt.ylabel('Iteration Number, t')
    plt.grid(True)
    plt.show()
    
    # QUALITY ASSURANCE PLOT FOR V_b
    plt.xlabel('V_b')
    plt.plot(Vb_MH, range(len(Vb_MH)), linestyle='-', color='black', linewidth=1.0)
    plt.title('Quality Assurance Plot for V_b')
    plt.xlabel('V_b')
    plt.ylabel('Iteration Number, t')
    plt.grid(True)
    plt.show()
    
    # OVERLAYING THE PLOT IN 6a 
    b, m, u_b, u_m, chi2, q = standardFit(x, y, sy, 'nooutlier')
    y_fit = (m * x) + b
    y_fit2 = (peak_m * x) + peak_b
    plt.plot(x, y, 'o', x, y_fit, '-', x, y_fit2, '-')
    plt.errorbar(x, y, sy, fmt = '|')
    plt.legend(('Raw Data', 'Standard Best-Fit', 'Overlaid MH Standard Best-Fit', 'Known Gaussian Uncertainties in y'),loc = 0)
    plt.title('Overlaid Standard Practice Plots')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    
# ???????????????????????????????????????
    
#========================================

main()