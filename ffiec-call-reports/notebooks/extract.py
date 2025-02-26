import os
import pandas as pd
import numpy as np
import zipfile
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

os.getcwd()
os.path.abspath(os.path.join(os.getcwd(), os.pardir))

directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/0. Raw data'

timestamps = []
for year in range(2001, 2024):
    for quarter in ['0331', '0630', '0930', '1231']:
        ts = quarter + str(year)
        timestamps.append(ts)

data = pd.DataFrame()

for ts in tqdm(timestamps):
    quarter_data = pd.DataFrame()
    ts_folder_dir = directory + f'/FFIEC CDR Call Bulk All Schedules {ts}'

    # add finincial institution names
    names_dir = ts_folder_dir + f'/FFIEC CDR Call Bulk POR {ts}.txt'
    quarter_data = pd.read_table(names_dir, skiprows=range(1, 2))
    quarter_data["Quarter"] = "{}/{}/{}".format(ts[:-6], ts[-6:-4], ts[-4:])

    for schedule in ['RCB', 'RCO', 'RC', 'RCE', 'RI', 'RCCI', 'RCCII', 'RCEI', 'RCEII', 'RCK', 'RIBI', 'RCEI']:
        schedule_dir = ts_folder_dir + f'/FFIEC CDR Call Schedule {schedule} {ts}.txt'
        try:
            schedule_data = pd.read_table(schedule_dir, skiprows=range(1, 2))
        except:
            schedule_data = pd.read_table(schedule_dir.replace('.txt', '(1 of 2).txt'), skiprows=range(1, 2))

        schedule_data = schedule_data.dropna(axis=1, how='all')
        quarter_data = quarter_data.merge(schedule_data, on='IDRSSD', how='left', suffixes=('', '_drop'))

    quarter_data.drop(quarter_data.filter(regex='_drop$').columns, axis=1, inplace=True)
    data = pd.concat([data, quarter_data])

    data.drop(columns=['TEXTA545', 'Last Date/Time Submission Updated On'], inplace=True)

    data.to_csv("../1. Data/FFEIC_2001Q1_2024Q3.csv", index=False)

    with open('dashboard_codes.txt', 'r') as file:
        codes_to_pull = [code.strip() for code in file]

    data_selected = data[codes_to_pull]
    data_selected['Quarter'] = pd.to_datetime(data_selected['Quarter'])

    for code in tqdm(codes_to_pull):
        if code not in ['Quarter', 'IDRSSD', 'Financial Institution Name', 'Financial Institution State']:
            data_selected[code] = data_selected[code] * 1000
interest_expense_codes = [code for code in data_selected.columns if code[:4] == 'RIAD']

data_selected = data_selected.sort_values(by=['Quarter', 'IDRSSD'])

for code in tqdm(interest_expense_codes):
    data_selected[f'{code}_diff'] = data_selected[code] - data_selected.groupby(['IDRSSD'])[code].shift().fillna(0)
    data_selected[f'{code}_Q'] = np.where(data_selected['Quarter'].dt.quarter == 1, data_selected[code], data_selected[f'{code}_diff'])
    data_selected.drop(columns=[f'{code}_diff', code], inplace=True)

data_selected = data_selected.reset_index(drop=True)

deposit_coeff = data_selected.copy()

deposit_coeff['Retail Transaction deposits'] = deposit_coeff['RCONP754'] + deposit_coeff['RCONP753']
deposit_coeff['Retail Transaction share'] = deposit_coeff['Retail Transaction deposits'] / deposit_coeff['RCONB549']
deposit_coeff['Retail MMDA share']=  deposit_coeff['RCONP756'] / deposit_coeff['RCON6810']
deposit_coeff['Nonretail MMDA share'] = deposit_coeff['RCONP757'] / deposit_coeff['RCON6810']
deposit_coeff['Retail Savings share'] = deposit_coeff['RCONP758'] / deposit_coeff['RCON0352']
deposit_coeff['Nonretail Savings share'] = deposit_coeff['RCONP759'] / deposit_coeff['RCON0352']

deposit_coeff = deposit_coeff.groupby('Quarter')[[
    'Retail Transaction share',
    'Retail MMDA share',
    'Nonretail MMDA share',
    'Retail Savings share',
    'Nonretail Savings share'
]].mean().reset_index()

deposit_coeff = deposit_coeff.fillna(deposit_coeff.mean())

data_selected = data_selected.merge(deposit_coeff, on='Quarter', how='left')

#time deposits
data_selected['Time deposits'] = np.where(
    data_selected['RCONJ474'].isna(),
    data_selected['RCON6648'].fillna(0) + data_selected['RCON2604'].fillna(0),
    data_selected['RCON6648'].fillna(0) + data_selected['RCONJ473'].fillna(0) + data_selected['RCONJ474'].fillna(0),
)

data_selected['Retail Time deposits'] = np.where(
    data_selected['RCONJ474'].isna(),
    data_selected['RCON6648'].fillna(0) + data_selected['RCON2604'].fillna(0),
    data_selected['RCON6648'].fillna(0) + data_selected['RCONJ473'].fillna(0),
)

data_selected['Nonretail Time deposits'] = data_selected['RCONJ474'].fillna(0)

#transaction
data_selected['Retail Transaction deposits'] = np.where(
    data_selected['RCONP754'].isna(),
    data_selected['RCONB549'].fillna(0) * data_selected['Retail Transaction share'],
    data_selected['RCONP754'].fillna(0) + data_selected['RCONP753'].fillna(0)
)

data_selected['Nonretail Transaction deposits'] = (
    data_selected['RCONB549'].fillna(0) - data_selected['Retail Transaction deposits']
)

#MMDA
data_selected['Retail MMDA'] = np.where(
    data_selected['RCONP756'].isna(),
    data_selected['RCON6810'].fillna(0) * data_selected['Retail MMDA share'],
    data_selected['RCONP756'].fillna(0)
)

data_selected['Nonretail MMDA'] = np.where(
    data_selected['RCONP757'].isna(),
    data_selected['RCON6810'].fillna(0) * data_selected['Nonretail MMDA share'],
    data_selected['RCONP757'].fillna(0)
)

#Savings
data_selected['Retail Savings'] = np.where(
    data_selected['RCONP758'].isna(),
    data_selected['RCON0352'].fillna(0) * data_selected['Retail Savings share'],
    data_selected['RCONP758'].fillna(0)
)

data_selected['Nonretail Savings'] = np.where(
    data_selected['RCONP759'].isna(),
    data_selected['RCON0352'].fillna(0) * data_selected['Nonretail Savings share'],
    data_selected['RCONP759'].fillna(0)
)

#Retail/Nonretail
data_selected['Retail deposits'] = (
    data_selected['Retail Time deposits'] +
    data_selected['Retail Transaction deposits'] +
    data_selected['Retail MMDA'] +
    data_selected['Retail Savings']
)

data_selected['Non-retail deposits'] = (
    data_selected['Nonretail Time deposits'] +
    data_selected['Nonretail Transaction deposits'] +
    data_selected['Nonretail MMDA'] +
    data_selected['Nonretail Savings']
)

#Totals
data_selected['Total deposits'] = data_selected['RCON2200'].fillna(0) + data_selected['RCFN2200'].fillna(0)

data_selected['Interest-bearing deposits'] = data_selected['RCON6636'].fillna(0) + data_selected['RCFN6636'].fillna(0)
data_selected['Noninterest-bearing deposits'] = data_selected['RCON6631'].fillna(0) + data_selected['RCFN6631'].fillna(0)

#Government
data_selected['Government DDA'] = (
    data_selected['RCON2202'].fillna(0) +
    data_selected['RCON2203'].fillna(0) +
    data_selected['RCON2216'].fillna(0)
)

data_selected['Government non-DDA'] = (
    data_selected['RCON2520'].fillna(0) +
    data_selected['RCON2530'].fillna(0) +
    data_selected['RCON2377'].fillna(0)
)

data_selected['Government deposits'] = data_selected['Government DDA'] + data_selected['Government non-DDA']

data_selected['Other domestic deposits(goverment & institutions)'] = (
    data_selected['RCON2200'].fillna(0) -
    data_selected['Retail deposits'].fillna(0) -
    data_selected['Non-retail deposits'].fillna(0)
)

data_selected['Deposits of banks & institutions'] = (
    data_selected['Other domestic deposits(goverment & institutions)'] - data_selected['Government deposits']
)

data_selected['Loans to depository institutions and other banks'] = (
    data_selected['RCONB531'].fillna(0) +
    data_selected['RCFDB532'].fillna(0) +
    data_selected['RCFDB533'].fillna(0) +
    data_selected['RCFDB534'].fillna(data_selected['RCONB534']).fillna(0) +
    data_selected['RCONB535'].fillna(0) +
    data_selected['RCFDB536'].fillna(0) +
    data_selected['RCFDB537'].fillna(0)
)

data_selected['Commercial & Industrial Loans'] = (
    data_selected['RCFD1763'].fillna(data_selected['RCON1763']).fillna(0) +
    data_selected['RCFD1764'].fillna(data_selected['RCON1764']).fillna(0) +
    data_selected['RCFD1590'].fillna(data_selected['RCON1590']).fillna(0)
)

data_selected['CRE Loans'] = (
    data_selected['RCFDF160'].fillna(data_selected['RCONF160']).fillna(0) +
    data_selected['RCFDF161'].fillna(data_selected['RCONF161']).fillna(0) +
    data_selected['RCFDF159'].fillna(data_selected['RCONF159']).fillna(0) +
    data_selected['RCFD1420'].fillna(data_selected['RCON1420']).fillna(0) +
    data_selected['RCFD1460'].fillna(data_selected['RCON1460']).fillna(0)
)

data_selected['C&I loans to U.S. addressees'] = data_selected['RCFD1763'].fillna(data_selected['RCON1763'])
data_selected['C&I loans to non-U.S. addressees'] = data_selected['RCFD1764'].fillna(data_selected['RCON1764'])
data_selected['Agricultural Production Loans'] = data_selected['RCFD1590'].fillna(data_selected['RCON1590'])
data_selected['Loans - Other CRE'] = data_selected['RCFDF161'].fillna(data_selected['RCONF161'])
data_selected['Loans - Owner-Occupied CRE'] = data_selected['RCFDF160'].fillna(data_selected['RCONF160'])
data_selected['Other construction and land development loans'] = data_selected['RCFDF159'].fillna(data_selected['RCONF159'])
data_selected['Loans secured by farmland'] = data_selected['RCFD1420'].fillna(data_selected['RCON1420'])

data_selected['Mortgage'] = (
    data_selected['RCFD5367'].fillna(data_selected['RCON5367']).fillna(0) +
    data_selected['RCFD5368'].fillna(data_selected['RCON5368']).fillna(0) +
    data_selected['RCFDF158'].fillna(data_selected['RCONF158']).fillna(0)
)

data_selected['HELOC'] = data_selected['RCFD1797'].fillna(data_selected['RCON1797'])
data_selected['Credit Cards'] = data_selected['RCFDB538'].fillna(data_selected['RCONB538'])
data_selected['Other revolving Retail loans'] = data_selected['RCFDB539'].fillna(data_selected['RCONB539'])
data_selected['Auto Loans'] = data_selected['RCFDK137'].fillna(data_selected['RCONK137'])
data_selected['Personal Loans'] = data_selected['RCFDK207'].fillna(data_selected['RCONK207'])

data_selected['Total Loans'] = data_selected['RCFD2122'].fillna(data_selected['RCON2122'])

data_selected['Retail Loans'] = (
    data_selected['Mortgage'].fillna(0) +
    data_selected['HELOC'].fillna(0) +
    data_selected['Credit Cards'].fillna(0) +
    data_selected['Other revolving Retail loans'].fillna(0) +
    data_selected['Auto Loans'].fillna(0) +
    data_selected['Personal Loans'].fillna(0)
)

data_selected['Nonretail Loans'] = (
    data_selected['CRE Loans'].fillna(0) +
    data_selected['Commercial & Industrial Loans'].fillna(0)
)

data_selected['Held-to-maturity securities'] = np.where(
    data_selected['RCFDJJ34'].isna(), data_selected['RCFD1754'], data_selected['RCFDJJ34']
)

code_mapping = {
    'RCON2200': 'Total domestic deposits',
    'RCFN2200': 'Total foreign deposits',
    'RCON6631': 'Noninterest-bearing domestic deposits',
    'RCON6636': 'Interest-bearing domestic deposits',
    'RCON2215': 'Transaction deposits',
    'RCON2385': 'Nontransaction deposits',
    'RCFN6636': 'Interest-bearing foreign deposits',
    'RCFN6631': 'Noninterest-bearing foreign deposits',
    'RCONP756': 'Retail MDDA',
    'RCONP757': 'Nonretail MDDA',
    'RCONP758': 'Retail Savings deposits',
    'RCONP759': 'Nonretail Savings deposits',
    'RCON6648': 'Time deposits <$100K',
    'RCONJ473': 'Time deposits $100-$250K',
    'RCONJ474': 'Time deposits >$250K',
    'RCON2604': 'Time deposits >$100K',
    'RCONP753': 'Retail Noninterest-bearing Transaction deposits',
    'RCONP754': 'Retail Interest-bearing Transaction deposits',
    'RCONB549': 'Retail & Nonretail Transaction deposits',
    'RCON2210': 'DDA',
    'RCON6810': 'MMDA',
    'RCON0352': 'Savings deposits',
    'RCONB550': 'Retail & Nonretail nontransaction deposits',
    'RIAD4508_Q': 'Interest expense on Transaction accounts',
    'RIAD0093_Q': 'Interest expense on Savings deposits (incl. MMDA)',
    'RIADHK03_Q': 'Interest expense on Time deposits (<$250K)',
    'RIADHK04_Q': 'Interest expense on Time deposits (>$250K)',
    'RIADHK04_Q': 'Interest expense on Time deposits (>$250K)',
    'RIADA517_Q': 'Interest expense on Time deposits (<$100K)',
    'RIADA518_Q': 'Interest expense on Time deposits (>$100K)',
    'RIAD4172_Q': 'Interest expense on foreign deposits',
    'RIAD4073_Q': 'Total interest expense',
    'RCFD1410': 'Loans secured by real estate',
    #     'RCONB538': 'Credit Cards (Domestic Offices)',
    #     'RCON1797': 'HELOC (Domestic Offices)',
    #     'RCONK137': 'Auto Loans (Domestic Offices)',
    #     'RCONK207': 'Personal Loans (Domestic Offices)',
    #     'RCONB539': 'Other revolving Retail loans (Domestic Offices)',
    #     'RCON5368': 'Mortgages: Junior liens (Domestic Offices)',
    #     'RCON5367': 'Mortgages: First liens (Domestic Offices)',
    #     'RCONF158': 'Construction & land loans: Retail (1-4 family) (Domestic Offices)',
    #     'RCFDB538': 'Credit Cards (Consolidated Bank)',
    #     'RCFD1797': 'HELOC (Consolidated Bank)',
    #     'RCONK137': 'Auto Loans (Consolidated Bank)',
    #     'RCFDK207': 'Personal Loans (Consolidated Bank)',
    #     'RCFDB539': 'Other revolving Retail loans (Consolidated Bank)',
    #     'RCFD5368': 'Mortgages: Junior liens (Consolidated Bank)',
    #     'RCFD5367': 'Mortgages: First liens (Consolidated Bank)',
    #     'RCFDF158': 'Construction & land loans: Retail (1-4 family) (Consolidated Bank)',
    #     'RCON1460': 'Loans secured by multifamily (5+) res. prop.',
    #     'RCFD1410': 'Loans secured by real estate',
    #     'RCONF159': 'Other construction and land development loans (Domestic Offices)',
    #     'RCFDF159': 'Other construction and land development loans (Consolidated Bank)',
    #     'RCON1420': 'Loans secured by farmland (Domestic Offices)',
    #     'RCFD1420': 'Loans secured by farmland (Consolidated Bank)',
    #     'RCFDF160': 'Loans - Owner-Occupied CRE (Domestic Offices)',
    #     'RCONF160': 'Loans - Owner-Occupied CRE (Consolidated Bank)',
    #     'RCFDF161': 'Loans - Other CRE (Domestic Offices)',
    #     'RCONF161': 'Loans - Other CRE (Consolidated Bank)',
    #     'RCFD1590': 'Agricultural Production Loans (Consolidated Bank)',
    #     'RCON1590': 'Agricultural Production Loans (Domestic Offices)',
    #     'RCFD2123': 'Total loans (Consolidated Bank)',
    #     'RCON2123': 'Total loans (Domestic Offices)',

}

data_selected.rename(columns=code_mapping, inplace=True)

data_selected['Interest expense on Time deposits'] = np.where(
    data_selected['Interest expense on Time deposits (>$250K)'].notna(),
    data_selected['Interest expense on Time deposits (<$250K)'].fillna(0) +
    data_selected['Interest expense on Time deposits (>$250K)'].fillna(0),
    data_selected['Interest expense on Time deposits (<$100K)'].fillna(0) +
    data_selected['Interest expense on Time deposits (>$100K)'].fillna(0)
)

data_selected['Interest expense on domestic deposits'] = (
    data_selected['Interest expense on Time deposits'].fillna(0) +
    data_selected['Interest expense on Transaction accounts'].fillna(0) +
    data_selected['Interest expense on Savings deposits (incl. MMDA)'].fillna(0)
)

data_selected['Interest expense on deposits'] = (
    data_selected['Interest expense on domestic deposits'].fillna(0) +
    data_selected['Interest expense on foreign deposits'].fillna(0)
)

data_selected['Total Assets'] = np.where(
    data_selected['RCFD2170'].isna(), data_selected['RCON2170'], data_selected['RCFD2170'])

data_selected = data_selected.sort_values(by=['IDRSSD', 'Quarter'], ascending=[True, False])

data_last = data_selected.groupby('IDRSSD').first().reset_index()

data_last['Bank type'] = np.where(
    data_last['Total Assets'] > 10**12, "Mega (>$1T)", np.where(
        data_last['Total Assets'] > 100 * 10**9, "Super Regional ($100B-1T)", np.where(
            data_last['Total Assets'] > 10 * 10**9, "Mid-cap ($10-100B)", "Small (<$10B)"
        )
    )
)

data_selected = data_selected.merge(data_last[['IDRSSD', 'Bank type']], on='IDRSSD', how='left')

data_rank = data_selected[data_selected['Quarter'] == data_selected['Quarter'].max()]
data_rank = data_rank.sort_values(by=['Total Assets'], ascending=False)
data_rank['Rank'] = data_rank['Total Assets'].rank(ascending = False).astype(int)

data_selected = data_selected.merge(data_rank[['IDRSSD', 'Rank']], on='IDRSSD', how='left')
data_selected['Rank'] = data_selected['Rank'].fillna(0)

effective_rate = pd.read_csv('../3. Reference/FED rates/FEDFUNDS.csv', skiprows=0)
effective_rate['DATE'] = pd.to_datetime(effective_rate['DATE'])
data_selected['DATE'] = data_selected['Quarter'].to_numpy().astype('datetime64[M]')
effective_rate['FEDFUNDS'] = effective_rate['FEDFUNDS'] / 100

data_selected = data_selected.merge(effective_rate, on='DATE', how='left')

columns_to_drop = list(set(data_selected.columns) & set(codes_to_pull[4:]))
for column in data_selected.columns:
    if column[:4] == 'RIAD':
        columns_to_drop.append(column)
columns_to_drop.append('DATE')

data_selected.drop(columns=columns_to_drop, inplace=True)

data_selected.to_excel("../1. Data/FFEIC_2001Q1_2024Q3_dashboard.xlsx", index=False)

