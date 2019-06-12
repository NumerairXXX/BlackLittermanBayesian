from pojo.portfolio import BlackLittermanBayesian
from utils import company_select
from utils import file_read
from backtest import simple_pnl

#portfolio inputs
df = file_read.input_file('dev')
num_2_select = 50
backtest_cutoff = 'input your date here'
method = 'cap' #for random selection, put random; cap means take top 50 market cap stocks
valid_company = company_select.valid_company_selector(df,num_2_select,method)

#initialize portfolio (both optimal and benchmark)
portfolio = BlackLittermanBayesian(df,backtest_cutoff,valid_company)
portfolio.set_ft()
portfolio.set_D()
portfolio.set_q_pred()
portfolio.set_sv_omega()

#generate backtest results
pnl_set, holdings_optimal = simple_pnl(portfolio)


