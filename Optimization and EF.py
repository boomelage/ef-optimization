import sys
sys.path.insert(0, 'pypfopt')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier
from pypfopt import plotting
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

df = pd.read_excel(r'cw2023AP.xlsx',index_col = 0)

df18 = df['2000-12-29' : '2018-10-01']
df23 = df['2018-09-28' : '2023-09-29']

stocks = ['BUNZL',
       'ROLLS-ROYCE HOLDINGS',
       'FULLER SMITH & TURNR.', 'BAE SYSTEMS',
       'SEVERN TRENT']

rs18 = ((df18[stocks]/df18[stocks].shift(1))-1).dropna()
rs23 = ((df23[stocks]/df23[stocks].shift(1))-1).dropna()

mu18 = rs18.mean(axis=0)
mu23 = rs23.mean(axis=0)

anmu18 = mu18 * 12
anmu23 = mu23 * 12

cov18 = rs18.cov()
cov23 = rs23.cov()

ukmid18 = df18['UK STERLING 1M DEPOSIT (FT/RFV) - MIDDLE RATE']
r18 = ukmid18.mean()/100
r18

ukmid23 = df23['UK STERLING 1M DEPOSIT (FT/RFV) - MIDDLE RATE']
r23 = ukmid23.mean()/100
r23

mu = anmu23
S = cov23 * np.sqrt(12)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe(risk_free_rate = r23)
ef.portfolio_performance(verbose=True, risk_free_rate = r23)

dicti = weights
opt23 = np.array(list(dicti.values()))
weights

mu = anmu23
S = cov23*np.sqrt(12)

ef = EfficientFrontier(mu, S)
weights = ef.min_volatility()
ef.portfolio_performance(verbose=True, risk_free_rate = r23)

dicti = weights
min23 = np.array(list(dicti.values()))
weights

mu = anmu23
S = cov23*np.sqrt(12)

rfr = r23

ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots()
ef_max_sharpe = ef.deepcopy()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find the tangency portfolio
ef_max_sharpe.max_sharpe()
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker=".", s=500, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 25000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Define the risk-free rate
risk_free_rate = rfr  # Adjust this to your actual risk-free rate

# Calculate the slope and intercept for the CML
slope_cml = (ret_tangent - risk_free_rate) / std_tangent
intercept_cml = risk_free_rate

# Plot the Capital Market Line
x_vals_cml = np.linspace(0, max(stds), 100)
y_vals_cml = slope_cml * x_vals_cml + intercept_cml
ax.plot(x_vals_cml, y_vals_cml, linestyle="--", color="blue", label="Capital Market Line")

# Output
ax.set_title("opt23")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=600)
plt.show()

# Calculate expected returns and sample covariance
mu = anmu18
S = cov18*np.sqrt(12)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe(risk_free_rate = r18)
ef.portfolio_performance(verbose=True, risk_free_rate = r18)

dicti = weights
opt18 = np.array(list(dicti.values()))
weights

mu = anmu18
S = cov18*np.sqrt(12)

ef = EfficientFrontier(mu, S)
weights = ef.min_volatility()
ef.portfolio_performance(verbose=True, risk_free_rate = r18)

dicti = weights
min18 = np.array(list(dicti.values()))
weights

mu = anmu18
S = cov18*np.sqrt(12)

rfr = r18

ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots()
ef_max_sharpe = ef.deepcopy()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find the tangency portfolio
ef_max_sharpe.max_sharpe()
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker=".", s=500, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 25000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Define the risk-free rate
risk_free_rate = rfr  # Adjust this to your actual risk-free rate

# Calculate the slope and intercept for the CML
slope_cml = (ret_tangent - risk_free_rate) / std_tangent
intercept_cml = risk_free_rate

# Plot the Capital Market Line
x_vals_cml = np.linspace(0, max(stds), 100)
y_vals_cml = slope_cml * x_vals_cml + intercept_cml
ax.plot(x_vals_cml, y_vals_cml, linestyle="--", color="blue", label="Capital Market Line")

# Output
ax.set_title("opt18")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=600)
plt.show()
