from collections import defaultdict


def calculate_pnl(portfolio):
    pnl = defaultdict(list)
    performance = defaultdict(list)
    for dates in portfolio.df['T'].unique()[2:]:
        hold_o, ret_o, name_o = portfolio.get_optimal_holding(dates)
        pnl_o = (hold_o @ ret_o)[0]
        pnl['optimal'].append(pnl_o)
        for i in range(len(name_o)):
            performance[name_o.values[i]].append(hold_o)

        hold_b, ret_b, name_b = portfolio.get_benchmark_holding(dates)
        pnl_b = (hold_b @ ret_b)[0]
        pnl['benchmark'].append(pnl_b)
    return pnl,performance