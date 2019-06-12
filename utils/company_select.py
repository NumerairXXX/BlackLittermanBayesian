import pandas as pd

def valid_company_selector(df,num,method = 'cap'):
    date_count = pd.DataFrame(df.groupby('NAME')['T'].count()).reset_index()
    valid_len = date_count['T'].value_counts().idxmax()
    total = date_count.equal('T',valid_len)
    if method == 'cap':
        cap_sorted = pd.DataFrame(df.groupby('NAME')['USDCAP'].mean()).reset_index().sort_values('USDCAP',ascending=False)
        valid_company = total.isin('NAME',cap_sorted['NAME'].values[:num])['NAME'].values
    if method == 'random':
        import random
        valid_company = random.choices(total['NAME'].values,k=num)
    return valid_company