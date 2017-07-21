
# coding: utf-8

# In[ ]:

from sympy import symbols, Rational, pi, Matrix,integrate, cse, IndexedBase, Idx,summation

tau,cs,tec,const,sig, nu0, dnu = symbols('tau cs tec const sig nu0 dnu')
nu = []
phi = []
sig = []
for i in range(4):
    nu.append(symbols("nu{:04d}".format(i)))
    phi.append(symbols("phi{:04d}".format(i)))
    sig.append(phi[i]*Rational(1,10))
    
L = Rational(0)
for i in range(len(nu)):
    nu_ = nu0 + Rational(i)*dnu
    #L+= (phi[i] - Rational(2)*pi*nu[i]*tau - const*tec/nu[i] - cs)**Rational(2)/Rational(2)/sig[i]**Rational(2)

    L+= (phi[i] - Rational(2)*pi*nu_*tau - const*tec/nu_ - cs)**Rational(2)/Rational(2)/sig[i]**Rational(2)

grad = Matrix([L.diff(tau), L.diff(tec), L.diff(cs)])

Hess = Matrix([[L.diff(tau).diff(tau), L.diff(tec).diff(tau), L.diff(cs).diff(tau)],
              [L.diff(tau).diff(tec), L.diff(tec).diff(tec), L.diff(cs).diff(tec)],
              [L.diff(tau).diff(cs), L.diff(tec).diff(cs), L.diff(cs).diff(cs)]])

grad


# In[ ]:

cse(Hess.inv().dot(grad),optimizations='basic')


# In[ ]:



